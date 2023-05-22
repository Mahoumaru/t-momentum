# coding:utf-8

import math
import torch
from torch.optim.optimizer import Optimizer
from . import _temafunctional as tF

class TAdamW(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0, warmup=0,
                 k_dof=1.0, beta_dof=0.999):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not (0.0 < k_dof or math.inf == k_dof):
            raise ValueError("Invalid degrees of freedom scale factor: {}".format(k_dof))
        if not 0.0 <= beta_dof <= 1.0:
            raise ValueError("Invalid beta parameter for dof optimisation: {}".format(beta_dof))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, warmup=warmup,
                        k_dof=k_dof, beta_dof=beta_dof, optim_dof=beta_dof < 1.0)
        super(TAdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(TAdamW, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('TAdamW does not support sparse gradients, please consider SparseAdam instead')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                beta_dof = group["beta_dof"]
                optim_dof = group["optim_dof"]

                state['step'] += 1

                # Weights computation
                betaw = tF.get_tema_decay_factor(grad=grad,
                                                 state=state,
                                                 group=group,
                                                 exp_avg=exp_avg,
                                                 exp_var=exp_avg_sq,
                                                 beta=beta1,
                                                 beta_dof=beta_dof,
                                                 optim_dof=optim_dof)
                ###
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                exp_avg.mul_(betaw).add_(1 - betaw, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['warmup'] > state['step']:
                    scheduled_lr = 1e-8 + state['step'] * group['lr'] / group['warmup']
                else:
                    scheduled_lr = group['lr']

                step_size = scheduled_lr * math.sqrt(bias_correction2) / bias_correction1

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * scheduled_lr, p_data_fp32)

                p_data_fp32.addcdiv_(exp_avg, denom, value=-step_size)

                p.data.copy_(p_data_fp32)

        return loss
