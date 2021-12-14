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
print(f'Device: {device}')
lr = 2e-3 # use the lr as recommended by the paper
gan_l_type='non-saturating' # consider switching to lsgan as used in the paper
bs = 1
save_every = 20 # save generator image every 50 images

# define the number of epochs for training
epochs = 100
loss_names = ['DLoss', 'FakeDLoss', 'RealDLoss', 'GLoss', 'GANGLoss', 'NCELoss', 'NCEIdentityLoss']

def main():
    data = Data('apples', 'oranges', False, bs, (128, 128))
    data.get_loaders('apples_and_oranges')
    
    # show_batch(data, 9)

    # define the CUT gan model which has all 3 nets and training loop for the 3 nets
    cut_model = CUT_gan(lambda_gan, lambda_nce, nce_layers, device, lr, gan_l_type=gan_l_type, bs=bs)

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
        # print(f"Start epoch {epoch}, total {len(data.dlSourceTrain)}")
        with tqdm(total=len(data.dlSourceTrain)) as pbar:
            for i, ((x, _), (y, _)) in enumerate(zip(data.dlSourceTrain, data.dlTargetTrain)):
                # print(x.shape, y.shape, i)
                # ims = torch.cat((x, y), dim=0)
                # save_images(ims, 'x_y_dl_test.png')

                # set model to train
                cut_model.train()
                # move the image tensors onto the correct device
                x = x.to(device)
                y = y.to(device)
                # train model
                cut_model.optimize_params(x, y)
                # update losses
                loss_d += cut_model.loss_d
                fake_d_loss += cut_model.fake_d_loss
                real_d_loss += cut_model.real_d_loss
                loss_g += cut_model.loss_g
                gan_g_loss += cut_model.gan_g_loss
                nce_loss += cut_model.nce_loss
                nce_identity_loss += cut_model.nce_identity_loss
                # save image to show in results at end of epoch
                if i % save_every == 0:
                    x_check.append(x)
                # update progress
                pbar.update(bs)

                if i == 5:
                    break
    
        ep_time = time.time() - start_ep
        cut_model.save_nets(epoch)
        # output training loss and time
        loss_list = [loss_d, fake_d_loss, real_d_loss, loss_g, gan_g_loss, nce_loss, nce_identity_loss]
        report_losses(loss_list, loss_names, ep_time, i+1)
        # output visuals
        cut_model.eval()
        x_check = torch.cat(x_check, dim=0)
        cut_model(x_check)
        x_fake = cut_model.fake_targ
        # put the images together to save them
        ims = torch.cat((x_check, x_fake), dim=3)
        save_images(ims, f'ep{epoch}.png')

main()
