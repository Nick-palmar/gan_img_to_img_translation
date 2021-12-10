
from data_utility import show_batch, Data, save_images
import os
from create_gen_discr import Generator, Disciminator, EncoderFeatureExtractor

from testing import test_mlp_network

# layers to look at for nce
nce_layers = [0, 2, 4, 5, 7, 9, 10]

def main():
    data = Data('apples', 'oranges', False, 64, (128, 128))
    data.get_loaders('apples_and_oranges')
    # show_batch(data, 9)
    
    # Test full generator and only encoder part of generator on batch of images
    generator = Generator(3, 3, nce_layers)
    nce_layer_channels = generator.feature_extractor_channels
    # change the nlayers parameter to change the output map size from the discriminator (128//2**n_layers)
    discriminator = Disciminator(3, n_layers=4)

    # create the feature encoder mlp
    feature_extractor = EncoderFeatureExtractor(None, nce_layer_channels)

    for i, (x, _) in enumerate(data.dlSourceTrain):
        # take a single image
        x_1 = x[:10]
        test_mlp_network(x_1, generator, feature_extractor, x_1.shape[0])
        # print(x_1.shape)
        # print(x_1.shape)
        full_gen_x = generator(x_1)
        enc_gen_x = generator(x_1, layers=nce_layers, encode_only=True)
        print('gen', full_gen_x.shape, len(enc_gen_x), enc_gen_x[0].shape, enc_gen_x[-1].shape)
        # print('Output shape', full_gen_x.shape)
        # save_images(full_gen_x, f'img_generator_{i}.png')
        discr_real = discriminator(x_1)
        discr_fake = discriminator(full_gen_x)
        print(discr_real.shape, discr_fake.shape)
        break



    


main()
