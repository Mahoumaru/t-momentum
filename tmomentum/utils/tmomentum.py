import torch
import math

"""
A separate class for the t-momentum algorithm.
GOAL: Make easier the integration of the t-momentum algorithm to new optimizers.
### WORK IN PROGRESS... ###
"""
class TMomentum(object):
    def __init__(self, k_dof=math.inf, eps=1e-6):#, estimate_kdof=False, kdof_estim_type="avg"):
        super(TMomentum, self).__init__(k_dof)
        self.k_dof = k_dof
        self.eps = eps
        """ ## kdof_estimator: TO BE IMPLEMENTED ##
        self.estimate_kdof = estimate_kdof
        if estimate_kdof == True:
            self.kdof_estimator = KdofEstimator(type=kdof_estim_type)
        """

    def tmomentum(self, gradient, state, exp_avg, beta, exp_var=None, kdof_beta=None):
        k_dof = self.k_dof
        if k_dof == math.inf:
            betaw = beta
        else:
            grad_diff_square = gradient.sub(exp_avg).square_()
            #########
            try:
                Wt = state['Wt']
                dof = state['dof']
                dim = state['dim']
            except KeyError:
                state['Wt'] = torch.tensor(0.0).type_as(exp_avg) + beta / (1.0 - beta)
                state['dim'] = float(gradient.numel())
                dim = state['dim']
                state['dof'] = torch.tensor(0.0).type_as(exp_avg) + k_dof * dim
                Wt = state['Wt']
                dof = state['dof']
            #########
            """ ## kdof_estimator: TO BE IMPLEMENTED ##
            if self.estimate_kdof == True:
                k_dof = self.kdof_estimator.get_kdof(state, grad_diff_square, kdof_beta, self.eps)
                dof = k_dof * dim
            """
            #########
            estim_var = False
            if exp_var is None:
                estim_var = True
                try:
                    exp_var = state['exp_var']
                except KeyError:
                    state['exp_var'] = torch.tensor(0.0).type_as(exp_avg)
                    exp_var = state['exp_var']
            #########
            wt = grad_diff_square.div(exp_var.add(self.eps)).sum()
            wt.add_(dof).reciprocal_().mul_(dim + dof)
            betaw = Wt.div(Wt.add(wt)).item()
            Wt.add_(wt).mul_(beta)
            if estim_var:
                exp_var.mul_(beta).add_(grad_diff_square.mul_(wt), alpha=1.0 - beta)
        return betaw
