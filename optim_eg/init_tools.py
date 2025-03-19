import math
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional as _Optional



class SplitBias(nn.Module):
    def __init__(self, base_layer):
        super().__init__()
        self.base_layer = base_layer
        self.has_bias = base_layer.bias is not None
        if self.has_bias:
            self.bias_pos = nn.Parameter(torch.empty_like(self.base_layer.bias))
            self.bias_neg = nn.Parameter(torch.empty_like(self.base_layer.bias))
            del self.base_layer.bias
            self.base_layer.bias = 0

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.has_bias:
            if len(self.base_layer.weight.shape) == 1:
                # norm layers, biases usually initialized with zeros
                bound = 1e-4
                # could test if there is a big difference for eg initializing with [+1, -1] or 1e-4
            else:
                # conv layers, biases usually initialized with uniform,")
                # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
                # same variance as for non-split bias
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.base_layer.weight)
                if fan_in > 0: bound = 1 / math.sqrt(fan_in)
                else: raise
            nn.init.uniform_(self.bias_pos, 0, math.sqrt(2) * bound)
            nn.init.uniform_(self.bias_neg, -bound * math.sqrt(2), 0)

    def forward(self, x):
        self.base_layer.bias = self.bias_pos + self.bias_neg
        return self.base_layer(x)


def set_split_bias(module):
    attr_to_change = dict()
    for name, child in module.named_children():
        if len(list(child.children())) > 0:
            set_split_bias(child)
        else:
            if hasattr(child, 'bias') and child.bias is not None:
                attr_to_change[name] = SplitBias(child)
    for name, value in attr_to_change.items():
        setattr(module, name, value)


def calc_ln_mu_sigma(mean, var):
    "Given desired mean and var returns ln mu and sigma"
    mu_ln = math.log(mean ** 2 / math.sqrt(mean ** 2 + var))
    sigma_ln = math.sqrt(math.log(1 + (var / mean ** 2)))
    return mu_ln, sigma_ln


def lognormal_(
    tensor: Tensor,
    gain: float = 1.0,
    mode: str = "fan_in",
    generator: _Optional[torch.Generator] = None,
    mean_std_ratio: float = 1.5,
    **ignored
):
    """
    Initializes the tensor with a log normal distribution * {1,-1}.

    Arguments:
        tensor: torch.Tensor, the tensor to initialize
        gain: float, the gain to use for the initialization stddev calulation.
        mode: str, the mode to use for the initialization. Options are 'fan_in', 'fan_out'
        generator: optional torch.Generator, the random number generator to use.
        mean_std_ratio: float, the ratio of the mean to std for log_normal initialization.
        conv2d_patch_same_sign: bool, assigns the same sign to each patch

    Note this function draws from a log normal distribution with mean = mean_std_ratio * std
    and then multiplies the tensor by a random Rademacher dist. variable (impl. with bernoulli).
    This induces the need to correct the ln std dev, as the final symmetrical distribution
    will have variance = mu^2 + sigma^2 = (1 + mean_std_ratio^2) * sigma^2. Where sigma, mu are
    the log normal distribution parameters.
    """
    with torch.no_grad():
        fan = torch.nn.init._calculate_correct_fan(tensor, mode)
        std = gain / math.sqrt(fan)
        std /= (1 + mean_std_ratio ** 2) ** 0.5
        mu, sigma = calc_ln_mu_sigma(std * mean_std_ratio, std ** 2)

        tensor.log_normal_(mu, sigma, generator=generator)
        tensor.mul_(2 * torch.bernoulli(0.5 * torch.ones_like(tensor), generator=generator) - 1)


def re_init_network(net, nonlinearity='relu',
                    mean_std_ratio:_Optional[float]=None,
                    generator:_Optional[float]=None):
    """
    Arguments:
        net: torch.nn.Module, the network to re-init
        nonlinearity: str, to calculate the gain to use for the initialization stddev calulation.
        mean_std_ratio: float, the ratio of the mean to std for log_normal initialization.
        generator: optional torch.Generator, the random number generator to use.
        conv2d_patch_same_sign: bool, assigns the same sign to each patch

    Note this function does not change biases.

    Pytorch default initialization bounds are based on a "good bug", where the uniform bounds are set
    to be the value of the std dev of weights for a given fan in/out. Therefore to replicate pytorch
    default init use gain = 1/root(3).
    """
    init_func = lognormal_
    gain = torch.nn.init.calculate_gain(nonlinearity)

    for m in net.modules():
        if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
            init_func(m.weight, gain=gain, mode="fan_out", mean_std_ratio=mean_std_ratio, generator=generator)
