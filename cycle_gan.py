from create_gen_discr import Generator, Disciminator
from data_utility import ImagePool, set_requires_grad
from losses import DGANLoss
import torch.nn as nn
import torch.optim as optim
import itertools
import os
import torch

class CycleGan:
    def __init__(self, device, lr, train=True, gan_l_type='lsgan', pool_size=50):
        self.device = device
        self.lr = lr
        self.is_train = train

        # define the two generator networks
        self.gen_src_targ = Generator(3, 3, []).to(self.device)
        self.gen_targ_src = Generator(3, 3, []).to(self.device)

        if self.is_train:
            # define both discriminators
            self.disc_targ = Disciminator(3, True).to(self.device)
            self.disc_src = Disciminator(3, True).to(self.device)
            # define buffers to store generated images
            self.fake_src_pool = ImagePool(pool_size)
            self.fake_targ_pool = ImagePool(pool_size)
            # define losses
            self.gan_loss = DGANLoss(gan_l_type, self.device).to(self.device)
            self.cycle_loss = nn.L1Loss()
            self.identity_loss = nn.L1Loss()
            # define optimizers; use itertools.chain to put both next parameters together in a single optimizer
            self.gen_optim = optim.Adam(itertools.chain(self.gen_src_targ.parameters(), self.gen_targ_src.parameters()), lr=lr)
            self.disc_optim = optim.Adam(itertools.chain(self.disc_targ.parameters(), self.disc_src.parameters()), lr=lr)


    def train(self):
        self.gen_src_targ.train()
        self.gen_targ_src.train()
        self.disc_src.train()
        self.disc_targ.train()


  
    def eval(self):
        self.gen_src_targ.eval()
        self.gen_targ_src.eval()
        if self.is_train:
            self.disc_src.eval()
            self.disc_targ.eval()
    

    def save_nets(self, epoch, folder='models'):
        """
        Save the network params on the cpu for all 3 networks
        """
        gen1_checkpoint = {'state_dict': self.gen_src_targ.cpu().state_dict()}
        gen2_checkpoint = {'state_dict': self.gen_targ_src.cpu().state_dict()}
        disc1_checkpoint = {'state_dict': self.disc_targ.cpu().state_dict()}
        disc2_checkpoint = {'state_dict': self.disc_src.cpu().state_dict()}
        
        # move the nets back to the current device
        self.gen_src_targ.to(self.device)
        self.gen_targ_src.to(self.device)
        self.disc_targ.to(self.device)
        self.disc_src.to(self.device)

        os.makedirs(folder, exist_ok=True)

        get_path = lambda model_name: os.path.join(folder, f"{epoch}_{model_name}.pth")
        torch.save(gen1_checkpoint, get_path('gen1'))
        torch.save(gen2_checkpoint, get_path('gen2'))
        torch.save(disc1_checkpoint, get_path('disc1'))
        torch.save(disc2_checkpoint, get_path('disc2'))


    def load_gen(self, epoch, folder='models'):
        checkpoint = torch.load(os.path.join(folder, f"{epoch}_gen1.pth"), map_location=torch.device('cpu'))
        gen = Generator(3, 3, [])
        gen.load_state_dict(checkpoint['state_dict'])
        # print(next(self.gen.parameters()))
        return gen


    def forward(self, real_src, real_targ):
        """
        Does a forward pass of the generator for training and inference.
        Note that in this implementation both real source and real target images are expected
        """
        # save the current real src and targ images to pass to other functions
        self.real_src = real_src
        self.real_targ = real_targ
        # use the generator on real source images, and then reconsturct that image
        self.fake_targ = self.gen_src_targ(self.real_src)
        self.src_reconstruct = self.gen_targ_src(self.fake_targ)
        # for the identity loss, use the generator on real target images, and then reconsturct that image
        self.fake_src = self.gen_targ_src(self.real_targ)
        self.targ_reconstruct = self.gen_src_targ(self.fake_src)
        
    
    def optimize_params(self, real_src, real_targ, discriminator_train=1, gen_train=1):
        """
        Forward pass, loss, back propagate, and step for all 3 networks to optmize all the params
        """
        # forward pass to compute images by generators
        self.forward(real_src, real_targ)
        # generator param update
        for _ in range(gen_train):
            set_requires_grad(self.disc_targ, False)
            set_requires_grad(self.disc_src, False)
            self.gen_optim.zero_grad()
            self.g_loss = self.calc_g_loss()
            self.g_loss.backward()
            self.gen_optim.step()
        # discriminator param update
        for _ in range(discriminator_train):
            set_requires_grad(self.disc_targ, True)
            set_requires_grad(self.disc_src, True)
            self.disc_optim.zero_grad()
            self.disc_targ_loss = self.calc_d_loss(self.disc_targ, self.real_targ, self.fake_targ)
            self.disc_src_loss = self.calc_d_loss(self.disc_src, self.real_src, self.fake_src)
            self.disc_targ_loss.backward()
            self.disc_src_loss.backward()
            self.disc_optim.step()
            


    def calc_d_loss(self, disc, real, fake):
        """
        Calculates discriminator loss
        """
        pass


    def calc_g_loss(self):
        """
        Calculates generator loss
        """
        pass

    