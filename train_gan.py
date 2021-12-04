
from data_utility import show_batch, Data, save_images
import os
from create_gen_discr import Generator


def main():
    data = Data('apples', 'oranges', False, 64, (128, 128))
    data.get_loaders('apples_and_oranges')
    # show_batch(data, 9)
    
    # Test full generator and only encoder part of generator on batch of images
    generator = Generator(3, 3)

    for i, (x, _) in enumerate(data.dlSourceTrain):
        # take a single image
        x_1 = x[0].unsqueeze(0)
        # print(x_1.shape)
        full_gen_x = generator(x_1)
        print('Output shape', full_gen_x.shape)
        save_images(full_gen_x, f'img_generator_{i}.png')



    


main()
