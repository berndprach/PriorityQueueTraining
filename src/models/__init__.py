from . import lipschitz_mlp, lipschitz_conv_net

MODELS = {
    "AOL-MLP": lipschitz_mlp.get_aol_mlp,
    "AOL-ConvNet": lipschitz_conv_net.get_aol_lsc,
    # "CPL-ConvNet": conv_net.get_cpl_conv_net,
}


names = list(MODELS.keys())
get = MODELS.get


def load(name, **kwargs):
    return MODELS[name](**kwargs)
