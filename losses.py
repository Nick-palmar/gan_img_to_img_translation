import torch.nn as nn
import torch.nn.functional as F
import torch

class DGANLoss(nn.Module):
    """
    Defines the GAN loss function for the discriminator's predictions of real or fake on data
    """
    def __init__(self, mode):
        super().__init__()
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
    
    def create_targ_tensor(self, inp, is_real):
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

# TODO: Ask prof for help understanding math behind this loss function
class PatchNCELoss(nn.Module):
    """
    The patch NCE loss to associate similar sections in source and target images
    """
    def __init__(self, tau, bs):
        self.loss = nn.CrossEntropyLoss(reduction='none')
        # assume bs=1 by default
        self.bs = bs
        # division factor for scaling outputs
        self.tau = tau
    
    def forward(self, real_feats, fake_feats):
        n_patches = real_feats.shape[0]
        n_transformed_space = real_feats.shape[1]
        # detach generator for the real features
        fake_feats = fake_feats.detach()

        # create the positive feature results by doing (1, bs*n_transformed_space) * (bs*n_transformed_space, 1) = (1, 1) for each patch in the group
        l_pos = torch.bmm(real_feats.view(n_patches, 1, -1), fake_feats.view(n_patches, -1, 1)).view(n_patches, 1)

        real_feats = real_feats.view(self.bs, -1, n_transformed_space)
        fake_feats = fake_feats.view(self.bs, -1, n_transformed_space)
        # the new number of patches is the number of patches by image
        n_patches = real_feats.shape[1]
        # create the negative feature results by doing (n_patches, n_transformed_space) * (n_transformed_space, n_patches) = (n_patches, n_patches)
        l_neg_batch = torch.bmm(real_feats, fake_feats.transpose(2, 1))
        # remove meaningless diagonal entries by masking the l_neg_batch with an identity matrix
        diag = torch.eye(n_patches, device=real_feats.device)
        l_neg_batch.masked_fill_(diag, -10.0)
        l_neg = l_neg_batch.view(n_patches, -1) # NOTICE: this line is different from the paper

        # concat over patch dimension (1 pos and n_patch neg patches)
        out = torch.cat([l_pos, l_neg], dim=1) / self.tau
        # the target feature is alway the l_pos feature of out which is at index 0
        targs = torch.zeros(out.shape[0], device=real_feats.device).long()
        loss = self.loss(out, targs)
        return loss


