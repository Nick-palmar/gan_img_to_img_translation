import torch
from data_utility import set_requires_grad
from create_gen_discr import Disciminator
n_patches=64

def test_mlp_network(x, generator, encoder_network, bs):
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

def main():
    test_set_requires_grad()


main()