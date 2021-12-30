from data_utility import Data, save_images, report_losses
from cut_model import CUT_gan
import torch
import time
from tqdm import tqdm

# define the params to pass to the cut gan model
lambda_gan = 1
lambda_nce = 1
nce_layers = [0, 2, 4, 5, 7, 9, 10]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
enc_net_feats = 32
num_patches = 128
print(f'Device: {device}')
lr = 2e-3 # use the lr as recommended by the paper
gan_l_type='lsgan' # consider switching to lsgan as used in the paper
bs = 1

save_every = 100 # save generator image every x images in a batch
# define the number of epochs for training
epochs = 100
loss_names = ['DLoss', 'FakeDLoss', 'RealDLoss', 'GLoss', 'GANGLoss', 'NCELoss', 'NCEIdentityLoss']

def main():
    data = Data('apples', 'oranges', False, bs, (128, 128))
    data.get_loaders('apples_and_oranges')
    # show_batch(data, 9)

    # define the CUT gan model which has all 3 nets and training loop for the 3 nets
    cut_model = CUT_gan(lambda_gan, lambda_nce, nce_layers, device, lr, gan_l_type=gan_l_type, bs=bs, num_patches=num_patches, encoder_net_features=enc_net_feats)
    # print(next(cut_model.gen.parameters()).device)

    for epoch in range(epochs):
        # discriminator losses
        loss_d = 0
        fake_d_loss = 0
        real_d_loss = 0
        # generator losses
        loss_g = 0
        gan_g_loss = 0
        nce_loss = 0
        nce_identity_loss = 0
        start_ep = time.time()
        x_check = []
        with tqdm(total=len(data.dlSourceTrain)) as pbar:
            for i, ((x, _), (y, _)) in enumerate(zip(data.dlSourceTrain, data.dlTargetTrain)):
                # set model to train
                cut_model.train()
                # move the image tensors onto the correct device
                x = x.to(device)
                y = y.to(device)
                # train model
                cut_model.optimize_params(x, y, discriminator_train=1)
                # update losses
                # with torch.no_grad():
                loss_d += cut_model.loss_d.item()
                fake_d_loss += cut_model.fake_d_loss.item()
                real_d_loss += cut_model.real_d_loss.item()
                loss_g += cut_model.loss_g.item()
                gan_g_loss += cut_model.gan_g_loss.item()
                nce_loss += cut_model.nce_loss.item()
                nce_identity_loss += cut_model.nce_identity_loss.item()
                # save image to show in results at end of epoch
                if i % save_every == 0:
                    x_check.append(x)
                # update progress
                pbar.update(bs)
    
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))
        ep_time = time.time() - start_ep
        cut_model.save_nets(epoch) # save all 3 networks
        # output training loss and time
        loss_list = [loss_d, fake_d_loss, real_d_loss, loss_g, gan_g_loss, nce_loss, nce_identity_loss]
        report_losses(loss_list, loss_names, epoch, ep_time, i+1)
        # output visuals
        cut_model.eval()
        x_check = torch.cat(x_check, dim=0)
        cut_model(x_check)
        x_fake = cut_model.fake_targ
        # put the images together to save them
        ims = torch.cat((x_check, x_fake), dim=3)
        save_images(ims, f'ep{epoch}.png')

main()
