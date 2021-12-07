import torch.nn as nn
import torch

def _single_conv(ch_in, ch_out, ks, stride=1, act=True, gammaZero=False, norm='batch', transpose=False, leaky=False):
        # do not reduce size due to ks mismatch
        padding = ks//2
        if not transpose:
            layers = [nn.Conv2d(ch_in, ch_out, ks, stride=stride, padding=padding)]
        else:
            layers = [nn.ConvTranspose2d(ch_in, ch_out, ks, stride=stride, padding=padding)]

        # add norm layer to prevent activations from getting too high
        if norm=='instance':
            norm_layer = nn.InstanceNorm2d(ch_out, affine=False, track_running_stats=False)
        elif norm=='batch':
            norm_layer = nn.BatchNorm2d(ch_out, affine=True, track_running_stats=True)
        else:
            raise Exception(f'Norm should be either "instance" or "batch" but {norm} was passed')

        if gammaZero and norm_layer=='batch':
            # init batch norm gamma param to zero to speed up training 
            nn.init.zeros_(norm_layer.weight.data)

        layers.append(norm_layer)
        # check if this layer should have an activation - yes unless the final layer
        if act and not leaky:
            layers.append(nn.ReLU())
        elif act and leaky:
            layers.append(nn.LeakyReLU(0.2))
        
        layers = nn.Sequential(*layers)
        return layers

class ResBlock(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1, leaky=False):
        super().__init__()
        self.conv = self._resblock_conv(ch_in, ch_out, leaky, stride=stride)
        self.pool = self._return if stride==1 else nn.AvgPool2d(stride, ceil_mode=True)
        self.id_conv = self._return if ch_in == ch_out else _single_conv(ch_in, ch_out, 1, stride=1, act=False)
        if leaky:
            self.relu = nn.LeakyReLU(0.2)
        else:
            self.relu = nn.ReLU()
    
    def _return(self, x):
        return x

    def _resblock_conv(self, ch_in, ch_out, leaky, stride=1, ks=3):
        # create the convolutional path of the resnet block following the bottleneck apporach
        conv_block = nn.Sequential(
            _single_conv(ch_in, ch_out//4, 1, leaky=leaky),
            _single_conv(ch_out//4, ch_out//4, ks, stride=stride, leaky=leaky), 
            _single_conv(ch_out//4, ch_out, 1, act=False, gammaZero=True)
        )
        return conv_block
    
    def forward(self, x):
        # apply a skip connection for resnet in the forward call
        return self.relu(self.conv(x) + self.id_conv(self.pool(x)))

class Generator(nn.Module):
    """
    Create a generator model (both encoder and decoder) which uses a resnet architecture as the encoder and transpoed 2d convolutions for decoder
    """
    def __init__(self, in_ch, out_ch, base_ch=64, n_blocks=6, n_downsamples=2):
        """
        Args:
            in_ch: Number of input channels in input tensor image
            out_ch: Number of output channels for output tensor image
            base_ch: Base number of channels throughout network
            n_blocks: Number of residual blocks to use
            n_downsamples: Number of downsamples to apply in the network stem
        """
        # network attributes
        super().__init__()
        self.n_downsamples = n_downsamples
        self.n_blocks = n_blocks
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.base_ch = base_ch

        # create resnet encoder stem to downsample data n_downsample times and reach initial filter param
        # below should be: [in_ch, base_ch//2, base_ch//2, base_ch, base_ch*2, base_ch*4]
        stem_sizes = self._create_stem_sizes()
        print(stem_sizes)
        self.stem = self._create_stem(stem_sizes)

        # Add residual layers for resnet encoder
        self.res_layers = self._create_res_layers()

        # create the encoder and decoder for the resnet
        self.encoder = nn.Sequential(*[self.stem, self.res_layers])
        # create upsampling layers for the decoder
        anti_stem_sizes = stem_sizes[::-1]
        self.decoder = self._create_decoder(anti_stem_sizes)
    
    def _create_stem_sizes(self):
        sizes = [self.in_ch, self.base_ch//2, self.base_ch//2, self.base_ch]
        # add the extra sizes as done in the following repo: https://github.com/taesungp/contrastive-unpaired-translation/blob/57430e99df041515c57a7ffd18bb7cbc3c1af0a9/models/networks.py#L914
        for i in range(self.n_downsamples):
            mult = 2 ** i
            sizes.append(self.base_ch*mult*2)
        return sizes
    
    
    def _create_stem(self, sizes):
        # apply n_downsample stride 2 convolutions 
        stem = [
            _single_conv(sizes[i], sizes[i+1], 3, stride = 2 if i < self.n_downsamples else 1) 
            for i in range(len(sizes)-1)
        ]
        return nn.Sequential(*stem)
    
    def _create_res_layers(self):
        # create multiplication factor
        mult = 2**self.n_downsamples
        layers = [ResBlock(self.base_ch*mult, self.base_ch*mult) for i in range(self.n_blocks)]
        return nn.Sequential(*layers)
    
    def _create_decoder(self, sizes):
        # print(sizes)
        lt_downsamples = lambda i: i < self.n_downsamples
        # create the decoder with transposed convolutions and a final Tanh layer
        decoder = [
            *[_single_conv(sizes[i], sizes[i+1], 3 if lt_downsamples(i) else 4, stride = 2 if lt_downsamples(i) else 1, transpose = True if lt_downsamples(i) else False)
            for i in range(len(sizes)-1)],
            nn.Tanh()
        ]
        return nn.Sequential(*decoder)
    

    def forward(self, x, encode_only=False):
        # first apply encoder
        enc_x = self.encoder(x)
        # print(enc_x.shape)
        # return only encoder results for patchNCELoss
        if encode_only:
            return enc_x
        
        # second apply decoder
        dec_x = self.decoder(enc_x)
        return dec_x


class Disciminator(nn.Module):
    """
    Create a discriminator model to tell the difference between real and fake images
    """
    def __init__(self, ch_in, base_ch=64, n_layers=3):
        super().__init__()
        self.ch_in = ch_in
        self.base_ch = base_ch
        self.n_layers = n_layers
        self.convs = self._create_conv_discriminator()
    
    def _create_conv_discriminator(self):
        # start with a 1x1 conv assuming  a 128*128 input image; convert to base_ch channels
        convs = [_single_conv(self.ch_in, self.base_ch, 1, stride=2, leaky=True)]

        # add multiple res blocks to reduce the 128*128 input
        ch_mult_prev = 1
        ch_mult = 1
        # first layer was alreday applied above; apply all others
        for i in range(1, self.n_layers):
            ch_mult_prev = ch_mult
            # set the multiplier to a max of 8 or 2**current layer
            ch_mult = min(2**i, 8)
            convs += [
                ResBlock(self.base_ch * ch_mult_prev, self.base_ch * ch_mult, stride=2, leaky=True)
            ]
        
        # output a single channel feature map of activations from the discriminator (from the Patch GAN paper)
        convs += [_single_conv(self.base_ch * ch_mult, 1, 3, leaky=True)]
        return nn.Sequential(*convs)
    
    def forward(self, x):
        # apply the convolutional discriminator
        out = self.convs(x)
        return out


# TODO: Define the discriminator loss (define and test vanilla, LSGAN, and non-saturating)


