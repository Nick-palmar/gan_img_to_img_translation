import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from typing import Tuple, List
import matplotlib.pyplot as plt
import matplotlib
import os
import random

class Data:
    def __init__(self, source: str, target: str, flip: bool, bs: int, im_size: Tuple[int], make_valid: bool = True):
        if not flip:
            self.source=source
            self.target=target
        else:
            self.source=target
            self.target=source
        self.flip=flip
        self.bs=bs
        self.im_size = im_size
        self.dlSourceTrain=None
        self.dlTargetTrain=None
        self.dlSourceTest=None
        self.dlTargetTest=None
        # seems like it is not necessary to create validation set
        # self.dlSourceValid=None
        # self.dlTargetValid=None
        # self.make_valid = make_valid

    def get_loaders(self, data_dir_name: str) -> List[DataLoader]:
        data_dir_path = os.path.join('data', data_dir_name)
        for dir in os.listdir(data_dir_path):
            # print(dir)
            loader = create_dataloader(os.path.join(data_dir_path, dir), self.im_size, self.bs)
            # set the data class with corresponding loader
            if dir == f'{self.source}_test':
                self.dlSourceTest=loader
            elif dir == f'{self.target}_test':
                self.dlTargetTest=loader
            elif dir == f'{self.source}_train':
                self.dlSourceTrain=loader
            elif dir == f'{self.target}_train':
                self.dlTargetTrain=loader


def create_dataloader(root: str, im_size: Tuple, bs: int) -> DataLoader:
    dataset = datasets.ImageFolder(
        root=root,
        transform=transforms.Compose([
            # get the images in desired dimensions
            transforms.Resize(im_size),
            transforms.CenterCrop(im_size),
            # turn images to tensors
            transforms.ToTensor(),
            # apply normalization to mean of 0 and standard deviation of 1
            transforms.Normalize((0, 0, 0), (1, 1, 1))
        ])
    )
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)
    return dataloader


class ImagePool:
    """
    Image buffer class for cycle GAN
    """
    def __init__(self, pool_size):
        self.ps = pool_size
        self.n_imgs = 0
        self.images = []
    
    def query(self, images):
        """
        Return images from buffer; 50% that returned images come from images in input and 50% chance that they come from the buffer
        """
        ret_imgs = []
        for img in images:
            # remove axis with size of image (1)
            img = torch.unsqueeze(img.data, 0)
            
            # add image to buffer and return if not full
            if len(self.images) < self.ps:
                self.images.append(img)
                ret_imgs.append(img)
            else:
                random_prob = random.uniform(0, 1)
                # by 50% chance, return and object from the buffer, insert the new one
                if random_prob < 0.5:
                    rand_id = random.randint(0, self.ps-1)
                    # create a deep copy to not overwrite the value
                    ret_img = self.images[rand_id].clone()
                    self.images[rand_id] = img
                    ret_imgs.append(ret_img)
                # by 50% chance, just return the image
                else:
                    ret_imgs.append(img)
            # return a tensor of the selected images
            return torch.cat(ret_imgs, 0)


def save_images(img_tensors: torch.Tensor, file_name: str, folder: str='output'):
    fig, axs = plt.subplots(img_tensors.shape[0], figsize=(5, img_tensors.shape[0]))
    img_idx = 0

    if img_tensors.shape[0] == 1:
        axs.imshow(img_tensors[img_idx].permute(1, 2, 0).detach().cpu())
        axs.set_xticks([])
        axs.set_yticks([])
    else:
        for ax in axs:
            # plt.axis('off')
            ax.imshow(img_tensors[img_idx].permute(1, 2, 0).detach().cpu())
            ax.set_xticks([])
            ax.set_yticks([])
            img_idx += 1
    
    # save the figure results
    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, file_name))
    plt.close(fig)

def show_batch(data: Data, n: int, folder: str='output') -> matplotlib.image.AxesImage:
    if n <= 0:
        raise ValueError(f"Expected the number of samples to be >= 1 but got {n} instead")
    elif n > data.bs:
        print(f"Will only show max of {data.bs} samples")
    
    sqrt_num_res = n ** (1/2)
    if sqrt_num_res - int(sqrt_num_res) != 0:
        raise Exception('Num res should be a perfect square (eg. 1, 4, 9, 16, 25, 36...)')
    sqrt_num_res = int(sqrt_num_res)
    os.makedirs(folder, exist_ok=True)

    for i, loader in enumerate([data.dlSourceTrain, data.dlTargetTrain]):
        for x, _ in loader:
            fig, axs = plt.subplots(sqrt_num_res, sqrt_num_res, figsize=(5, 5))
            img_idx = 0

            for row in axs:
                for ax in row:
                    # plt.axis('off')
                    ax.imshow(x[img_idx].squeeze(0).permute(1, 2, 0).detach())
                    ax.set_xticks([])
                    ax.set_yticks([])
                    img_idx += 1
            
            # save the figure results
            if i == 0:
                file = data.source+'.png'
            else:
                file = data.target+'.png'
            plt.savefig(os.path.join(folder, file))
            break
    
def set_requires_grad(net, requires_grad):
    """
    Freezes or unfreezes a network (for the weight to change or not during training)
    Note that this is pass by reference so returns None
    """
    for param in net.parameters():
        param.requires_grad = requires_grad


def report_losses(losses, loss_names, epoch, time, scale=1):
    """
    Formats losses in a list
    """
    print(f'\nEpoch: {epoch}', end=" | ")
    for loss, name in zip(losses, loss_names):
        print(f"{name}: {round(loss/scale, 2)}", end=" | ")
    print(f"Time: {round(time, 2)}s\n")