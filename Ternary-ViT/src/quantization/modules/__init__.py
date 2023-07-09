from .conv import  QConv2d
from .linear import QLinear
from .utils import (
    register_act_quant_hook,
    replace_module_by_qmodule,
    replace_relu_by_prelu,
)
