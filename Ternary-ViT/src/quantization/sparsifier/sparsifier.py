import math
import numpy as np
import torch
import torch.nn.functional as F


class Sparsifier(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        
    def forward(self, x):
        raise NotImplementedError

    def init_from(self, x, *args, **kwargs):
        pass

    def update_mask(self, x, sparsity=None):
        pass

    
class IdentitySparsifier(Sparsifier):
    def forward(self, weight, base_weight=None):
        return weight


class PerChlMPLnSparsifier(Sparsifier):
    """
    Per channel multi-pattern ln-based sparsifier
    """
    def __init__(self, patterns, norm="l1", dim=1, **kwargs):
        super().__init__()
        assert norm in ["l1", "l2"]
        assert dim == 1
        self.norm = norm
        self.dim = dim
        self.freeze = False
        self.mask = None
        self.patterns = patterns
        self.check_mask = None
        self.check = True

    def freeze_mask(self):
        self.freeze = True

    def unfreeze_mask(self):
        self.freeze = False

    def init_from(self, x, *args, **kwargs):
        self.mask = torch.nn.Parameter(torch.ones_like(x)) 

    def forward(self, weight, base_weight=None):
        if base_weight is None:
            base_weight = weight
        if self.freeze:
            mask = self.mask
        else:
            mask = self.compute_mask(base_weight).detach()
            self.mask = mask
        self.freeze_mask()
        # if self.check:
        #     self.check_mask = self.mask
        #     self.check=False
        # print(self.mask == self.check_mask)
        # if self.freeze:
        #     if self.mask is not None:
        #         self.mask.data.copy_(mask.data)
        #     else:
        #         self.mask = torch.nn.Parameter(mask)
        return weight * mask

    def compute_mask(self, weight, sparsity=None):
        weight_norm = self.compute_weight_norm(weight)
        mask = torch.zeros_like(weight_norm)
        N, C, H, W = weight.shape

        for pattern in self.patterns.split(","):
            m, n = pattern.split(":")
            m, n = int(m), int(n)
            masked_weight = ((1 - mask) * weight_norm).reshape(N, C // m, m, H, W)
            _, idx = torch.topk(masked_weight, n, dim=2, largest=True)
            mask = mask.view(N, C // m, m, H, W)
            mask.scatter_(2, idx, 1)
            mask = mask.view(N, C, H, W)
        # print(torch.sum(mask)*1.0/mask.flatten().shape[0])
        return mask

    def compute_weight_norm(self, weight):
        if self.norm == "l1":
            weight_norm = torch.abs(weight)
        elif self.norm == "l2":
            weight_norm = weight ** 2
        else:
            raise NotImplementedError
        return weight_norm +1e-7

    def extra_repr(self):
        return (
            f"patterns={self.patterns}, norm={self.norm}, dim={self.dim}"
        )
