import torch
import torch.nn as nn
import torch.optim as optim
from create_gen_discr import Generator, Disciminator, EncoderFeatureExtractor
from losses import DGANLoss, PatchNCELoss
from data_utility import set_requires_grad

class CUT_gan(nn.Module):
    def __init__(self, lambda_gan, lambda_nce, nce_layers, device, lr, nce_idt=True, encoder_net_features=256, nce_tau=0.07, num_patches=256, train=True, gan_l_type='non-saturating', bs=1):
        """
        Creates a CUT model which is a type of GAN for image to image translation

        Args: 
            lambda_gan: The weight for the GAN loss for the generator (since generator loss depends on gan and nce loss)
            lambda_nce: The weight for the NCE loss for the generator (since generator loss depends on gan and nce loss)
            nce_layers: A list of layers that the generator encoder will return for the nce_loss from convolutional layer (can also be residual blocks) activations
            device: torch.Device('cpu') if a cpu is used to train on otherwise torch.Device('cuda:0') if gpu is avaiable
            lr: The learning rate for stepping the weights of all of the optimizers
            nce_idt: True if the loss consists of the identity loss NCE(Y, X-tilde), False otherwise
            encoder_net_features: The number of features that the EncoderFeatureExtractor will produce in it's new space
            nce_tau: A constant that the nce loss will use to scale the matrices by in the loss
            num_patches: The number of pathces that will be used by nce to compute the loss
            train: True if training False if evaluating/inferencing
            gan_l_type: The type of dgan loss to be used (either 'non-saturating', 'vanilla', or 'lsgan')
            bs: The batch size that is going to be used in training
        """
        super().__init__()
        # keep relevant attirbutes for the training loop
        self.device = device
        self.lambda_gan = lambda_gan
        self.lambda_nce = lambda_nce
        self.nce_idt = nce_idt
        self.num_patches = num_patches
        # definte the generator for the CUT model to go from rgb -> rgb image
        self.gen = Generator(3, 3, nce_layers).to(self.device)
 
        if train:
            # define a discriminator to take 3 input channels with 4 residual blocks
            self.disc = Disciminator(3, n_layers=4).to(self.device)
            # define a feature extractor network H sub l to transform generator encoder features to a new embedding space for nce loss
            self.feat_net = EncoderFeatureExtractor(self.gen.feature_extractor_channels, n_features=encoder_net_features).to(self.device)

            # define loss functions
            self.dgan_loss = DGANLoss(gan_l_type).to(self.device)
            self.nce_losses = []
            for _ in nce_layers:
                self.nce_losses.append(PatchNCELoss(nce_tau, bs).to(self.device))

            # create adam optimizers
            self.gen_optim = optim.Adam(self.gen.parameters(), lr=lr)
            self.disc_optim = optim.Adam(self.disc.parameters(), lr=lr)
            self.feat_net_optim = optim.Adam(self.feat_net.parameters(), lr=lr)
    

    def train(self):
        """
        Set all 3 networks to training mode
        """
        self.gen.train()
        self.disc.train()
        self.feat_net.train()
    

    def eval(self):
        """
        Depending on the mode, set the networks to eval mode
        """
        if self.train:
            self.gen.eval()
            self.disc.eval()
            self.feat_net.eval()
        else:
            self.gen.eval()
        

    def forward(self, real_src, real_targ):
        """
        Does a forward pass of the generator for training and inference.

        Saves the real source, real target, and fake target images.
        Also, if nce_idt and train are True, saves the fake source images. 
        """
        # save the current real src and targ images to pass to other functions
        self.real_src = real_src
        self.real_targ = real_targ
        if self.train and self.nce_idt:
            # put the real source and target images if in training and using identity nce loss
            real = torch.cat((real_src, real_targ), dim=0)
        else:
            real = real_src
        
        # use the generator on real images
        fake = self.gen(real)
        # get fake target images (y hat)
        self.fake_targ = fake[:real_src.shape[0]]
        # if possible, get fake source images for identity loss (x tilde)
        if self.train and self.nce_idt:
            self.fake_src = fake[real_src.shape[0]:]


    def optimize_params(self, real_src, real_targ):
        """
        Forward pass, loss, back propagate, and step for all 3 networks to optmize all the params
        """
        # forward pass
        self.forward(real_src, real_targ)

        # discriminator param update
        set_requires_grad(self.disc, True)
        self.disc_optim.zero_grad()
        loss_d = self.calc_d_loss()
        loss_d.backward()
        self.disc_optim.step()

        # generator and encoder feature extractor param update
        set_requires_grad(self.disc, False)
        self.gen_optim.zero_grad()
        self.feat_net_optim.zero_grad()
        loss_g = self.calc_g_loss()
        loss_g.backward()
        self.gen_optim.step()
        self.feat_net_optim.step()


    # TODO: ensure that keeping the loss as variables and not class attibutes does not affect back propagation
    def calc_d_loss(self):
        """
        Calculates discriminator loss
        """
        # prevent generator from updating
        fake_targ = self.fake_targ.detach()
        # fake target loss
        fake_pred = self.disc(fake_targ)
        fake_loss = self.dgan_loss(fake_pred, False)
        # real target loss
        real_pred = self.disc(self.real_targ)
        real_loss = self.dgan_loss(real_pred, True)
        # combine both fake and real target loss
        return (fake_loss + real_loss) * 0.5


    def calc_g_loss(self):
        """
        Calculates generator loss
        """
        # check normal GAN loss on discriminator with fake generator images
        pred_fake = self.disc(self.fake_targ)
        gan_loss = self.dgan_loss(pred_fake, False) * self.lambda_gan

        # use patch NCE loss for src -> fake targ
        nce_loss = self.calc_nce_loss(self.real_src, self.real_targ)
        # use patch NCE loss for targ -> fake source (identity loss)
        nce_identity_loss = self.calc_nce_loss(self.real_targ, self.fake_src)
        # get total nce loss
        nce_loss_total = (nce_loss + nce_identity_loss) * 0.5
        # get total loss (Lgan + NCE loss + identity NCE loss)
        loss_total = nce_loss_total + gan_loss
        return loss_total


    def calc_nce_loss(self, src, targ):
        """
        Calculates the NCE loss using patches to associate similar locations and dissociate different locations
        """
        # get the pathces for source after doing H sub l(G enc(x))
        src_feats = self.gen(src, encode_only=True)
        transformed_src_feats, patch_ids = self.feat_net(src_feats, self.num_patches)

        # get the patches for target after doing H sub l(G enc(G(x))) 
        targ_feats = self.gen(targ, encode_only=True)
        transformed_targ_feats, _ = self.feat_net(targ_feats, self.num_patches, patch_ids=patch_ids)

        total_loss = 0
        # calculate the loss for each layer in the transformed returned features
        for src_feat, targ_feat, nce_loss in zip(transformed_src_feats, transformed_targ_feats, self.nce_losses):
            # TODO: Consider switching src_feats and targ_feats if training is not working well
            total_loss += (nce_loss(src_feat, targ_feat) * self.lambda_nce).mean()
        
        return total_loss/len(self.gen.nce_layers)
