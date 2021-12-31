from data_utility import Data, save_images, report_losses
from cut_model import CUTGan
from cycle_gan import CycleGan
import torch
import time
from tqdm import tqdm

# define the model to use
model_type = 'cycle'

# define the params to pass to the cut gan model
lambda_gan = 1
lambda_nce = 1
nce_layers = [0, 2, 4, 5, 7, 9, 10]
enc_net_feats = 32
num_patches = 128

# define the params to pass to the cycle GAN model
lambda_src = 10
lambda_targ = 10
lambda_identity = 0.5

save_every = 100 # save generator image every x images in a batch
# define training parameters common to both architectures
gan_l_type='lsgan' # consider switching to lsgan as used in the paper
epochs = 100
lr = 2e-4 
bs = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
print(f'Device: {device}')


def main():
    data = Data('apples', 'oranges', False, bs, (128, 128))
    data.get_loaders('apples_and_oranges')
    # show_batch(data, 9)

    # define the CUT gan model which has all 3 nets and training loop for the 3 nets
    if model_type == 'CUT':
        model = CUTGan(lambda_gan, lambda_nce, nce_layers, device, lr, gan_l_type=gan_l_type, bs=bs, num_patches=num_patches, encoder_net_features=enc_net_feats)
        loss_names = ['DLoss', 'FakeDLoss', 'RealDLoss', 'GLoss', 'GANGLoss', 'NCELoss', 'NCEIdentityLoss']
    elif model_type == 'cycle':
        model = CycleGan(device, lr, gan_l_type=gan_l_type, lambda_src=lambda_src, lambda_targ=lambda_targ, lambda_identity=lambda_identity)
        loss_names = ['GLoss', 'GAN_x_to_y', 'GAN_y_to_x', 'FwdCycle', 'BwdCycle', 'IdtSrc', 'IdtTarg', 'DTarg', 'DSrc']
    else:
        raise ValueError(f'Model must be of type "CUT" or "cycle" not {model_type}')

    for epoch in range(epochs):
        # reset discriminator losses
        loss_list = model.zero_losses()
        start_ep = time.time()
        x_check = []
        with tqdm(total=len(data.dlSourceTrain)) as pbar:
            for i, ((x, _), (y, _)) in enumerate(zip(data.dlSourceTrain, data.dlTargetTrain)):
                # set model to train
                model.train()
                # move the image tensors onto the correct device
                x = x.to(device)
                y = y.to(device)
                # train model
                model.optimize_params(x, y, discriminator_train=1)
                # update losses
                model.get_losses()
                model.update_losses(loss_list)
                # save image to show in results at end of epoch
                if i % save_every == 0:
                    x_check.append(x)
                # update progress
                pbar.update(bs)
    
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))
        ep_time = time.time() - start_ep
        model.save_nets(epoch) # save all 3 networks
        # output training loss and time
        # loss_list = [loss_d, fake_d_loss, real_d_loss, loss_g, gan_g_loss, nce_loss, nce_identity_loss]
        report_losses(loss_list, loss_names, epoch, ep_time, i+1)
        # output visuals
        model.eval()
        x_check = torch.cat(x_check, dim=0)
        model(x_check)
        x_fake = model.fake_targ
        # put the images together to save them
        ims = torch.cat((x_check, x_fake), dim=3)
        save_images(ims, f'ep{epoch}.png')

main()
