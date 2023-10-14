import torch

from .quantizer import Quantizer


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad


def clip(x, eps):
    x_clip = torch.where(x > eps, x, eps)
    return x - x.detach() + x_clip.detach()


def round_p2(x):
    y = torch.log2(x).round()
    y = 2 ** y
    y_grad = x
    return (y - y_grad).detach() + y_grad


class LsqQuantizer(Quantizer):
    def __init__(
        self,
        bit=None,
        thd_pos=None,
        thd_neg=None,
        all_positive=False,
        symmetric=False,
        per_channel=True,
        normalize_first=False,
        p2_round_scale=False,
        **kwargs,
    ):
        super().__init__(
            bit, thd_pos, thd_neg, all_positive, symmetric, per_channel, normalize_first
        )
        self.s = torch.nn.Parameter(torch.ones(1))
        self.p2_round_scale = p2_round_scale
        # self.eps = self.register_buffer("eps", torch.tensor(1e-5).float())

    def init_from(self, x, *args, **kwargs):
        x = self.normalize(x)
        if self.per_channel:
            self.s = torch.nn.Parameter(
                x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True) * 2 / (self.thd_pos ** 0.5))
        else:
            self.s = torch.nn.Parameter(x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5))

    def forward(self, x):
        if self.per_channel:
            # s_grad_scale = 1.0 / ((self.thd_pos * x[0].numel()) ** 0.5)
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        else:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        s_scale = grad_scale(clip(self.s, torch.tensor(self.eps).float().to(self.s.device)), s_grad_scale)
        # s_scale = grad_scale(self.s, s_grad_scale)

        # round to power-of-2
        if self.p2_round_scale:
            s_scale = round_p2(s_scale)

        x = x / s_scale
        if self.bit == 1 and not self.all_positive:
            x = torch.sign(x)
        else:
            x = torch.clamp(x, self.thd_neg, self.thd_pos)
            x = round_pass(x)
        x = x * s_scale
        return x

    def extra_repr(self):
        return (
            f"bit={self.bit}, "
            f"pos={self.thd_pos}, neg={self.thd_neg}, "
            f"norm=({self.normalize_first}, {self.eps}, {self.gamma}), "
            f"all_positive={self.all_positive}, "
            f"symmetric={self.symmetric}, "
            f"per_channel={self.per_channel}"
        )
