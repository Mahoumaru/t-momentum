# coding:utf-8

import math
import torch
from torch.optim.optimizer import Optimizer
from . import _temafunctional as tF

class TDiffGrad(Optimizer):
    r"""Implements a Robust version of DiffGrad.

    .. _DiffGrad: An Optimization Method for
    Convolutional Neural Networks:
          https://arxiv.org/abs/1909.11015
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.0, amsgrad=False,
                 k_dof=1.0, beta_dof=0.999):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= amsgrad <= 1.0:
            raise ValueError("Invalid amsgrad parameter: {}".format(amsgrad))
        if not (0.0 < k_dof or math.inf == k_dof):
            raise ValueError("Invalid degrees of freedom scale factor: {}".format(k_dof))
        if not 0.0 <= beta_dof <= 1.0:
            raise ValueError("Invalid beta parameter for dof optimisation: {}".format(beta_dof))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        k_dof=k_dof, beta_dof=beta_dof, optim_dof=beta_dof < 1.0)
        super(TDiffGrad, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(TDiffGrad, self).__setstate__(state)
        # for group in self.param_groups:
        #     group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        r"""Performs a single optimization step.
        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('TDiffGrad, just as Adam, does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']
                beta1, beta2 = group['betas']
                beta_dof = group["beta_dof"]
                optim_dof = group["optim_dof"]

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Previous gradient
                    state['previous_grad'] = grad.clone()

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                previous_grad = state['previous_grad']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq'].mul_(amsgrad)

                state['step'] += 1
                bias_correction1 = 1.0 - beta1 ** state['step']
                bias_correction2 = 1.0 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # diffgrad
                # first-order momentum
                betaw = tF.get_tema_decay_factor(grad=grad,
                                                 state=state,
                                                 group=group,
                                                 exp_avg=exp_avg,
                                                 exp_var=exp_avg_sq,
                                                 beta=beta1,
                                                 beta_dof=beta_dof,
                                                 optim_dof=optim_dof)
                dfc = previous_grad.sub(grad).abs_().neg_().exp_().add_(1.0).reciprocal_()
                exp_avg.mul_(betaw).add_(grad, alpha=1.0 - betaw).mul_(dfc)
                previous_grad.copy_(grad)

                # second-order momentum
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                # update parameter
                step_size = group['lr'] / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
