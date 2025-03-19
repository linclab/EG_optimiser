import torch
import torch.nn as nn
import math


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
