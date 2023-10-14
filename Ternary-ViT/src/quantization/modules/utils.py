import os
import torch
import torch.nn as nn

from .conv import QConv2d, QConvBn2d
from .linear import QLinear
from .attention import QAttention
from ..quantizer import build_quantizer
from ..sparsifier import build_sparsifier

from src.utils import Attention


QMODULE_MAPPINGS = {
    torch.nn.Conv2d: QConv2d,
    torch.nn.Linear: QLinear,
    Attention: QAttention,
    # ConvBn2d: QConvBn2d,
    torch.nn.intrinsic.modules.fused.ConvBn2d: QConvBn2d,
}


def get_module_by_name(model, module_name):
    names = module_name.split(".")
    module = model
    for name in names:
        module = getattr(module, name)
    return module


def set_module_by_name(model, module_name, module):
    names = module_name.split(".")
    if len(names) == 1:
        parent = model
    else:
        parent = get_module_by_name(model, ".".join(names[:-1]))
    setattr(parent, names[-1], module)


def replace_module_by_qmodule(model, qconfigs):
    for name, cfg in qconfigs.items():
        module = get_module_by_name(model, name)
        print(cfg)
        if "down" in name or "xts" in name:
             qmodule = QMODULE_MAPPINGS[type(module)](
            module,
            quan_w_fn=build_quantizer(cfg["weight_q"]),
            quan_a_fn=build_quantizer(cfg["act_q"]),
            quan_attn_fn=build_quantizer(getattr(cfg, "attn_q", cfg["act_q"])),
            sparse_w_fn=build_sparsifier(None),
            sparse_a_fn=build_sparsifier(cfg["act_s"]),
        )
        else:
            qmodule = QMODULE_MAPPINGS[type(module)](
                module,
                quan_w_fn=build_quantizer(cfg["weight_q"]),
                quan_a_fn=build_quantizer(cfg["act_q"]),
                quan_attn_fn=build_quantizer(getattr(cfg, "attn_q", cfg["act_q"])),
                sparse_w_fn=build_sparsifier(cfg["weight_s"]),
                sparse_a_fn=build_sparsifier(cfg["act_s"]),
            )
        set_module_by_name(model, name, qmodule)
    return model


def register_act_quant_hook(model, qconfigs):

    def quant_act_hook(self, input, output):
        return self.quan_a_fn(output)

    for name, cfg in qconfigs.items():
        if cfg is not None:
            module = get_module_by_name(model, name)
            print(cfg.get("act_q", cfg))
            quan_a_fn = build_quantizer(cfg.get("act_q", cfg))
            module.quan_a_fn = quan_a_fn
            module.register_forward_hook(quant_act_hook)

    return model


def replace_relu_by_prelu(module):
    module_output = module
    if isinstance(module, (nn.ReLU, nn.ReLU6)):
        module_output = nn.PReLU()
        module_output.training = module.training
    for name, child in module.named_children():
        module_output.add_module(name, replace_relu_by_prelu(child))
    del module
    return module_output
