import torch
from torch import Tensor
from torch.optim.optimizer import (Optimizer, required)
from typing import List, Optional


class SGD(Optimizer):
    """
    This is a slightly stripped down & modified version of torch.optim.SGD

    See https://github.com/pytorch/pytorch/blob/main/torch/optim/sgd.py

    update_alg can choose between gradient descent ('gd') and exponentiated gradient ('eg')
    For 'gd', freeze_gd_signs=True implements sign-constrained gradient descent.
    To avoid storing signs for each parameter, updates that cross 0 fix the weight
    at +-freeze_gd_signs_th instead (so the next param.sign() returns the same sign).
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, update_alg="gd",
                 freeze_gd_signs=False, freeze_gd_signs_th=1e-18):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if update_alg not in ["gd", "eg", "gd_sign"]:
            raise ValueError("Invalid update_alg value: {}".format(update_alg))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        update_alg=update_alg, freeze_gd_signs=freeze_gd_signs,
                        freeze_gd_signs_th=freeze_gd_signs_th)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []

            # this is from optim.sgd._init_group
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            sgd(params_with_grad,
                d_p_list,
                momentum_buffer_list,
                weight_decay=group['weight_decay'],
                momentum=group['momentum'],
                lr=group['lr'],
                dampening=group['dampening'],
                nesterov=group['nesterov'],
                update_alg=group['update_alg'],
                freeze_gd_signs=group['freeze_gd_signs'],
                freeze_gd_signs_th=group['freeze_gd_signs_th'])

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss


def sgd(params: List[Tensor],
        d_p_list: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool,
        update_alg: str,
        freeze_gd_signs: bool,
        freeze_gd_signs_th: float
        ):
    for i, param in enumerate(params):

        d_p = d_p_list[i]
        if weight_decay != 0:
            if update_alg == "gd":
                d_p = d_p.add(param, alpha=weight_decay)
            elif update_alg == "eg":
                # param.sign added bec. in the update we need to multiply by sign
                # which induces an error here (neg weights will grow with w.d)
                d_p = d_p.add(param.sign(), alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        if update_alg == "gd" and freeze_gd_signs:
            s = param.sign()
            param.add_(d_p, alpha=-lr)
            # if s and param signs are different, s * param becomes negative
            flip_idx = (s * param) < 0
            param[flip_idx] = freeze_gd_signs_th * s[flip_idx] # setting to a non-zero value so the weight keep the sign
        elif update_alg == 'gd':
            param.add_(d_p, alpha=-lr)
        elif update_alg == "eg":
            # multiply by sign to ensure that the update is in the correct direction
            # this occurs because eg is not compatible with negative weights
            # if weight is neg, and grad is neg (so pos update) instead the weight will become more negative
            param.mul_(torch.exp(param.sign() * d_p * -lr))
