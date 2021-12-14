import torch
from data_utility import set_requires_grad
from create_gen_discr import Disciminator, Generator, EncoderFeatureExtractor
from losses import DGANLoss
from data_utility import Data
from cut_model import CUT_gan

bs = 1
n_patches=64
nce_layers = [0, 2, 4, 5, 7, 9, 10]
data = Data('apples', 'oranges', False, bs, (128, 128))
data.get_loaders('apples_and_oranges')    
# Test full generator and only encoder part of generator on batch of images
generator = Generator(3, 3, nce_layers)
nce_layer_channels = generator.feature_extractor_channels
# change the nlayers parameter to change the output map size from the discriminator (128//2**n_layers)
discriminator = Disciminator(3, n_layers=4)
# create the feature encoder mlp
encoder_network = EncoderFeatureExtractor(nce_layer_channels)

def test_nets():
    for i, (x, _) in enumerate(data.dlSourceTrain):
        # print(x.shape)
        full_gen_x = generator(x)
        enc_gen_x = generator(x, layers=nce_layers, encode_only=True)
        print('gen', full_gen_x.shape, len(enc_gen_x))
        for layer, layer_idx in zip(enc_gen_x, generator.nce_layers):
            print(f"layer{layer_idx}: {layer.shape}")
        # save_images(full_gen_x, f'img_generator_{i}.png')
        discr_real = discriminator(x)
        discr_fake = discriminator(full_gen_x)
        assert(discr_real.shape == discr_fake.shape)
        break


def test_mlp_network():
    for i, (x, _) in enumerate(data.dlSourceTrain):
        enc_gen_x = generator(x, encode_only=True)
        new_space_x, ids = encoder_network(enc_gen_x, n_patches)
        new_space_x_copy, _ = encoder_network(enc_gen_x, n_patches, patch_ids=ids)

        # assertions to tet working mlp
        # the end sizes are (bs*num_patches, self.n_features)
        for layer_1, layer_2 in zip(new_space_x, new_space_x_copy):
            # since encoder layers and ids are the same, both should be the same
            assert(torch.equal(layer_1, layer_2))
            # test the shape of the tensors are as expected 
            assert(len(layer_1.shape) == 2)
            assert(layer_1.shape[0] ==  bs*n_patches)
            assert(layer_1.shape[1] == encoder_network.n_features)
            print(layer_1.shape)
            # break
        break

    print("all assertions passed")

def make_param_assert(net, exp_requires_grad):
    for param in net.parameters():
        assert(param.requires_grad == exp_requires_grad)

def test_set_requires_grad():
    """
    This assertion confirmed that the pytorch nn.Module network is pass by reference
    """
    d = Disciminator(3)
    set_requires_grad(d, False)
    make_param_assert(d, False)
    set_requires_grad(d, True)
    make_param_assert(d, True)
    print("all assertions passed")

def test_d_loss():
    for loss_type in ['non-saturating', 'lsgan', 'vanilla']:
        dloss = DGANLoss(loss_type)
        data = Data('apples', 'oranges', False, 1, (128, 128))
        data.get_loaders('apples_and_oranges')
        generator = Generator(3, 3, nce_layers)
        discriminator = Disciminator(3, n_layers=4)

        for i, (x, _) in enumerate(data.dlSourceTrain):
            fake_x = generator(x)
            pred_fake = discriminator(fake_x)
            loss = dloss(pred_fake, False)
            # loss.backward()
            print(loss_type, loss.item())
            break

def test_mat_mul():
    tens1 = torch.randn((10, 3, 6))
    tens2 = torch.randn((10, 3, 6))

    t_mm = torch.matmul(tens1.transpose(2,1), tens2)
    t_m_sum = (tens1*tens2).sum(dim=1)[:, :, None]
    t_bmm = torch.bmm(tens1.transpose(2, 1), tens2)
    print(t_mm.shape, t_m_sum.shape, t_bmm.shape)

def test_load_gen():
    lambda_gan = 1
    lambda_nce = 1
    nce_layers = [0, 2, 4, 5, 7, 9, 10]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    print(f'Device: {device}')
    lr = 2e-3 # use the lr as recommended by the paper
    gan_l_type='non-saturating' # consider switching to lsgan as used in the paper
    bs = 1
    # define the CUT gan model which has all 3 nets and training loop for the 3 nets
    cut_model = CUT_gan(lambda_gan, lambda_nce, nce_layers, device, lr, gan_l_type=gan_l_type, bs=bs)
    gen = cut_model.load_gan(0)

def main():
    # test_set_requires_grad()
    # test_d_loss()
    # test_mat_mul()
    # test_mlp_network()
    test_nets()


main()