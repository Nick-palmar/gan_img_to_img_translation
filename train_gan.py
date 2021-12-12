
from data_utility import Data, save_images
from cut_model import CUT_gan
import torch

# define the params to pass to the cut gan model
lambda_gan = 1
lambda_nce = 1
nce_layers = [0, 2, 4, 5, 7, 9, 10]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
print(f'Device: {device}')
lr = 2e-3 # use the lr as recommended by the paper
gan_l_type='non-saturating' # consider switching to lsgan as used in the paper
bs = 1

# define the number of epochs for training
epochs = 1

def main():
    data = Data('apples', 'oranges', False, bs, (128, 128))
    data.get_loaders('apples_and_oranges')
    # show_batch(data, 9)

    # define the CUT gan model which has all 3 nets and training loop for the 3 nets
    cut_model = CUT_gan(lambda_gan, lambda_nce, nce_layers, device, lr, gan_l_type=gan_l_type, bs=bs)

    for epoch in range(epochs):
        for i, ((x, _), (y, _)) in enumerate(zip(data.dlSourceTrain, data.dlTargetTrain)):
            # print(x.shape, y.shape, i)
            # ims = torch.cat((x, y), dim=0)
            # save_images(ims, 'x_y_dl_test.png')
            # set model to train
            cut_model.train()
            # move the image tensors onto the correct device
            x = x.to(device)
            y = y.to(device)
            
            cut_model.optimize_params(x, y)
            print(f"Done training {epoch}.{i}")
        
        # output training loss and time

        # output validation loss and visuals
        cut_model.eval()
        for j, ((x, _), (y, _)) in enumerate(zip(data.dlSourceTest, data.dlTargetTest)):
            # only see outputs of 5 images
            if j > 5:
                break
            cut_model(x, y)
            y_hat = cut_model.fake_targ
            res_cat = torch.cat((x, y_hat), dim=0)
            save_images(res_cat, f'ep{epoch}_{j}.png')

main()
