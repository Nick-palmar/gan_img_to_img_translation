import torch.nn as nn
import torch.nn.functional as F
import torch

class DGANLoss(nn.Module):
    """
    Defines the GAN loss function for the discriminator's predictions of real or fake on data
    """
    def __init__(self, mode):
        self.mode = mode

        # define the loss to be used when receiving a grid of activaitons (predictions) for an image
        if self.mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.mode == 'non-saturating':
            self.loss = None
        else:
            raise NotImplementedError(f"The mode {mode} for DGANLoss is not implemented")
    
    def create_targ_tensor(inp, is_real):
        if is_real:
            targ_tensor = torch.Tensor([1])
        else:
            targ_tensor = torch.Tensor([0])
        # returns the target tensor in the same shape as the input (since it will be a grid of activations from the discriminator)
        return targ_tensor.expand_as(inp)
        
    
    def forward(self, x, is_real):
        if self.mode in ['lsgan', 'vanilla']:
            # create an equal shaped target tensor and compute the loss
            targ_tens = self.create_targ_tensor(x, is_real)
            loss = self.loss(x, targ_tens)
        # non-saturating loss is being used
        else:
            if is_real:
                # minimize the loss by passing softplus(-x) = ln(1+e**-x) as x -> +inf (real prediction get predicted more real) => e**-x -> 0 => softplus(-x) -> 0+
                loss = F.softplus(-x)
            else:
                # minimize the loss by passing softplus(x) = ln(1+e**x) as x -> -inf (fake prediction get predicted more fake) => e**x -> 0 => softplus(x) -> 0+
                loss = F.softplus(x)

            # since the discriminator is giving a grid of activations, group the loss by batch and take the mean along the activation dimension
            loss = loss.view(x.shape[0], -1).mean(1)
        return loss

class PatchNCELoss(nn.Module):
    """
    
    """
