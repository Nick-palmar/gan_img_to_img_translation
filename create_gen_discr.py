import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

# get an instance of a generator to use for numpy random calls
rng = np.random.default_rng()

class Normalize(nn.Module):
    """
    Normalization layer taken from https://github.com/taesungp/contrastive-unpaired-translation/blob/57430e99df041515c57a7ffd18bb7cbc3c1af0a9/models/networks.py#L449

    The CUT GAN paper states "We normalize vectors onto a unit sphere to prevent the space from collapsing or expanding"
    This layer is used to normalize vectors onto a unit sphere by using l2 norm
    """
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        # compute the l2 vector norm
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        # scale the input x by the norm
        out = x.div(norm + 1e-7)
        return out

def _single_conv(ch_in, ch_out, ks, stride=1, act=True, gammaZero=False, norm='batch', transpose=False, leaky=False):
    """
    Layer to perform a single convolution, normalization (batch or instance) and activation function (relu and leaky relu)
    """
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
    """
    Residual blocks to be used by both the generator and discriminator
    """
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

    Note that sometimes only certain layers will be taken from the encoder
    """
    def __init__(self, in_ch, out_ch, nce_layers, base_ch=64, n_blocks=6, n_downsamples=2):
        """
        Args:
            in_ch: Number of input channels in input tensor image
            out_ch: Number of output channels for output tensor image
            nce_layers: A list of layers that the patchNCE loss function will be using from the encoder
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
        # print(stem_sizes)
        self.stem = self._create_stem(stem_sizes)

        # Add residual layers for resnet encoder
        self.res_layers = self._create_res_layers()

        # create the encoder and decoder for the resnet
        self.encoder = nn.Sequential(*self.stem+self.res_layers)
        # print(self.encoder)
        # create upsampling layers for the decoder
        anti_stem_sizes = stem_sizes[::-1]
        self.decoder = self._create_decoder(anti_stem_sizes)

        # determine the layer channels for the Hl (feature extractor) network
        self.nce_layers = nce_layers
        self._determine_layer_channels(nce_layers)
    
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
        return stem
    
    def _create_res_layers(self):
        # create multiplication factor
        mult = 2**self.n_downsamples
        layers = [ResBlock(self.base_ch*mult, self.base_ch*mult) for i in range(self.n_blocks)]
        return layers
    
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
    
    def _determine_layer_channels(self, layers):
        """
        Determines the channels for the output layers that generator encoder will produce
        """
        # create a list of channels that will be used by the feature extractor
        self.feature_extractor_channels = []
        for layer_id, layer in enumerate(self.encoder):
            # print(layer, hasattr(layer, 'conv'))
            # only add the channels to the feature extractor channels if it is in the list of layers
            if layer_id in layers:
                if hasattr(layer, 'conv'):
                    try:
                        conv_out = layer.conv[2][0]
                    except:
                        print("The resblock has a different configuration as expected")
                else:
                    try:
                        if type(layer[0]) == nn.Conv2d:
                            conv_out = layer[0]
                    except:
                        print("The conv layer has a different configuration as expected")
                self.feature_extractor_channels.append(conv_out.out_channels)


    def forward(self, x, layers=[], encode_only=False):
        """
        Generator forward pass; only forward the specific layers if they were passed (if no layers, return entire encoder + decoder)
        """
        # TODO: Consider removing the layers input var
        layers = self.nce_layers
        # only output specific generator encoder layers
        if len(layers) > 0 and encode_only:
            encoder_layer_outs = [] # list of activation maps when one index corresponds to a layer of the encoder
            for layer_id, layer in enumerate(self.encoder):
                # compute the output for the layer
                x = layer(x)
                # only add the layer activation map to the output if in the list of layers (this will be used as one of the layers by PatchNCELoss)
                if layer_id in layers:
                    encoder_layer_outs.append(x)
            return encoder_layer_outs

        # first apply encoder - only reaches this part if layers is empty
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
    Create a discriminator model to tell the difference between real and fake images (assumes 128*128 input images)
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



class EncoderFeatureExtractor(nn.Module):
    """
    Create a MLP (multilayer perceptron) to transform the patch features from output and input into shared feature space
    Approach is taken from SimCLR: https://arxiv.org/pdf/2002.05709.pdf
    """
    def __init__(self, nce_layer_channels, n_features=256):
        """
        Creates the MLP network (H sub l in the paper) that will transform input encoder patches to a shared embedding space

        Args: 
            nce_layer_channels: A list containing the size of the channels for each of the layers that will be used by nce loss
            n_features: An integer which is the number of features that the transformed space will have
        """
        super().__init__()
        self.norm = Normalize(2)
        # self.gpu = gpu
        self.n_features = n_features
        # create the mlp for each layer that will be used for nce (based on the channels)
        self.mlps = nn.ModuleList(self._create_mlp(nce_layer_channels)) # define this as a module list for pytorch to treat each module in the list on it's own
    
    def _create_mlp(self, layer_channels):
        """
        Create a two layer fully connected network for each of the layers with the corresponding channel sizes
        """
        mlps = []
        for ch_in in layer_channels:
            mlps.append(nn.Sequential(*[
                nn.Linear(ch_in, self.n_features), 
                nn.ReLU(), 
                nn.Linear(self.n_features, self.n_features)
                ]))
        return mlps


    def forward(self, feats, num_patches, patch_ids=None):
        """
        Performs a forward pass for an EncoderFeatureExtractor (called Hl in this paper: https://arxiv.org/pdf/2007.15651.pdf)

        Args: 
            feats: A list containing tensor features passed from specific layer of the generator encoder (assume tensors of size bs*channels*H*W)
            num_patches: The number of patches to sample from feats
            patch_ids: The indexes of patches to select from each layer of feats (this is != None when a forward call has been used on the source images and we want to take the same patches from the target images)
        
        Returns: 
            A tuple of lists, the first list being 
        """
        return_ids = []
        return_feats = []

        # go through each of the feature layers while grabbing corresponding mlp
        for feat_id, (mlp, feat) in enumerate(zip(self.mlps, feats)):
            # reshape the feature tensor to be (bs*img_locs*channels)
            feat_reshaped = feat.flatten(2, 3).permute(0, 2, 1)
            # print('feats', feat.shape, feat_reshaped.shape)

            if num_patches > 0:
                # get the patch id for the current layer if the patch_ids exist
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                # create the patches if the patch ids DNE
                else:
                    # get random permutation of all indices from 0 to max img loc (axis=1)
                    patch_id = rng.permutation(feat_reshaped.shape[1])
                    # index the patch_id to extract first num_patch locations 
                    if num_patches < len(patch_id):
                        patch_id = patch_id[:num_patches]
                # index only the patches that will be used from feats_reshaped (axis 1 is the img_loc axis)
                # note that in practice, bs=1 so that we only compare batches in the same image (internal patches outperforms external patches)
                # TODO: Consider removing the .view() at the end of the next line
                feat_patch_sampled = feat_reshaped[:, patch_id, :].view(-1, feat_reshaped.shape[2]) # flatten the tensor to be of shape (patch_loc, channels). The PatchNCE loss will be done regardless of batch (this makes it by patches and not by image)
            else:
                # the number of patches is zero or negative; take all patches 
                feat_patch_sampled = feat_reshaped
                patch_id = []
             
            # apply the mlp (H sub l) to select patches
            feat_sampled_emb_space = mlp(feat_patch_sampled)
            # "We normalize vectors onto a unit sphere to prevent the space from collapsing or expanding"
            feat_norm = self.norm(feat_sampled_emb_space)
            return_ids.append(patch_id)

            # if zero patches, return in as similar a shape as possible (but in dimensions of shared embedding space after passing through mlp)
            if num_patches == 0:
                batch, _, h, w = feat.shape
                n_emb_feat = feat_norm.shape[-1]
                # shape is (bs*img_loc*emb_features) -> (bs*emb_features*img_loc)
                feat_norm = feat_norm.permute(0, 2, 1).view(batch, n_emb_feat, h, w)
            return_feats.append(feat_norm)
        
        return return_feats, return_ids

        
    