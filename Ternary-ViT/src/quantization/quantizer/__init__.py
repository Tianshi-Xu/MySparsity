import copy

from .lsq import LsqQuantizer
from .quantizer import IdentityQuantizer
from .twn import TwnQuantizer


def build_quantizer(cfg):
    if cfg is None:
        return IdentityQuantizer()

    cfg = copy.deepcopy(cfg)
    print(cfg)
    if cfg['mode'] == "Identity":
        quant = IdentityQuantizer
    elif cfg['mode'] == "LSQ":
        quant = LsqQuantizer
    elif cfg['mode'] == "TWN":
        quant = TwnQuantizer
    # elif cfg['mode'] == "BWN":
    #     quant = BwnQuantizer
    else:
        raise NotImplementedError

    cfg.pop('mode')
    return quant(**cfg)
