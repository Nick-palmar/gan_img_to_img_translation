from create_gen_discr import Generator, Disciminator
from data_utility import ImagePool, set_requires_grad
from losses import DGANLoss
import torch.nn as nn
import torch.optim as optim
import itertools
import os
import torch

class CycleGan(nn.Module):
    def __init__(self, device, lr, train=True, gan_l_type='lsgan', pool_size=50, lambda_src=10, lambda_targ=10, lambda_identity=0.5):
        super().__init__()
        self.device = device
        self.lr = lr
        self.is_train = train

        # set lambda scaling factors
        self.lambda_src = lambda_src
        self.lambda_targ = lambda_targ
        self.lambda_identity = lambda_identity

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


    def forward(self, real_src, real_targ=None):
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
        if self.real_targ != None:
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
            self.disc_targ_loss = self.calc_d_loss(self.disc_targ, self.real_targ, self.fake_targ, self.fake_targ_pool)
            self.disc_src_loss = self.calc_d_loss(self.disc_src, self.real_src, self.fake_src, self.fake_src_pool)
            self.disc_targ_loss.backward()
            self.disc_src_loss.backward()
            self.disc_optim.step()


    def calc_d_loss(self, disc, real, fake, fake_pool):
        """
        Calculates discriminator loss
        """
        queried_fake = fake_pool.query(fake)
        # real image discriminator GAN loss
        d_pred_real = disc(real)
        real_d_loss = self.gan_loss(d_pred_real, True)
        # fake image disciminator GAN loss
        d_pred_fake = disc(queried_fake.detach())
        fake_d_loss = self.gan_loss(d_pred_fake, False)
        return (real_d_loss + fake_d_loss) * 0.5


    def calc_g_loss(self):
        """
        Calculates generator loss (identity, GAN discriminator, and cycle loss)
        """
        # calculate the identity loss
        if self.lambda_identity > 0:
            # assume x -G-> y and y -F-> x
            # take identity loss || G(y) - y||
            identity_targ = self.gen_src_targ(self.real_targ)
            self.loss_idt_targ = self.identity_loss(identity_targ, self.real_targ) * self.lambda_targ * self.lambda_identity
            # take identity loss || G(x) - x||
            identity_src = self.gen_targ_src(self.real_src)
            self.loss_idt_src = self.identity_loss(identity_src, self.real_src) * self.lambda_src * self.lambda_identity
        else:
            self.loss_idt_targ = 0
            self.loss_idt_src = 0
        
        # take the discriminator loss for fake target images to make them look real (loss of target domain discriminator on fake target images to look 'real'; GANLoss(D_Y(G(X)), Make these look real))
        self.loss_gan_src_targ = self.gan_loss(self.disc_targ(self.fake_targ), True)
        # discriminator loss for fake soruce images to make them look real. GANLoss(D_X(F(Y)), Make these look real)
        self.loss_gan_targ_src = self.gan_loss(self.disc_src(self.fake_src), True)
        # get the forward cycle loss as || F(G(x)) - x|| where x -G-> y and y -F-> x
        self.fwd_cycle = self.cycle_loss(self.src_reconstruct, self.real_src) * self.lambda_src
        # get the backward cycle loss as || G(F(y)) - y|| where x -G-> y and y -F-> x
        self.bwd_cycle = self.cycle_loss(self.targ_reconstruct, self.real_targ) * self.lambda_targ
        # return the combined loss
        return self.loss_gan_src_targ + self.loss_gan_targ_src + self.fwd_cycle + self.bwd_cycle + self.loss_idt_src + self.loss_idt_targ


    def get_losses(self):
        self.new_losses = [
            self.g_loss.item(), 
            self.loss_gan_src_targ.item(),
            self.loss_gan_targ_src.item(),
            self.fwd_cycle.item(),
            self.bwd_cycle.item(),
            self.loss_idt_src.item(),
            self.loss_idt_targ.item(),
            self.disc_targ_loss.item(),
            self.disc_src_loss.item(),
        ]    
        return self.new_losses
    
    
    def update_losses(self, loss_list):
        # update the loss list through pass by reference
        for i, new_loss in enumerate(self.new_losses):
            loss_list[i] += new_loss


    def zero_losses(self):
        """
        Return a list of 9 zeros for the 9 losses being outputted
        """
        return [0] * 9

    