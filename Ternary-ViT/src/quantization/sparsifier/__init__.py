import copy

from .sparsifier import IdentitySparsifier, PerChlMPLnSparsifier


def build_sparsifier(cfg):
    if cfg is None:
        return IdentitySparsifier()

    cfg = copy.deepcopy(cfg)
    if cfg['mode'] == "Identity":
        sp_fn = IdentitySparsifier
    elif cfg['mode'] == "PerChlMPLn":
        sp_fn = PerChlMPLnSparsifier
    else:
        raise NotImplementedError

    cfg.pop('mode')
    return sp_fn(**cfg)
