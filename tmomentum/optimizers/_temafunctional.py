import math
import torch

def get_tema_decay_factor(grad, state, group, exp_avg, exp_var, beta, beta_dof, optim_dof=False):
    if group["k_dof"] == math.inf:
        return beta
    if state['step'] == 1:
        state['W_t'] = beta / (1.0 - beta)
        # Dimension d of the parameters
        state['dim'] = float(exp_avg.numel())
        # Degrees of freedom, initialized to the parameters dimension or to the user specified value
        if not optim_dof:
            state['dof'] = group['k_dof'] * state['dim']
        else:
            state['trigamma_value'] = torch.polygamma(1, torch.tensor(0.5 * state['dim'])).item()
            state['z_ave'] = 0.0
            state['z_var'] = 2.0 * (group['k_dof'] + 2.0) / group['k_dof']**2 + state['trigamma_value']
    Wt = state['W_t']
    #z_ave, z_var = state["z_ave"], state["z_var"]
    D_ = grad.sub(exp_avg).square_().div_(exp_var.add(group['eps'])).sum().item()
    if optim_dof:
        dof = update_dof(z_ave=state['z_ave'], z_var=state['z_var'], mahal_dist=D_, beta_dof=beta_dof,
                         trigamma_value=state['trigamma_value'], eps=group['eps'])
        state['dof'] = dof * state['dim']
    wt = (state['dim'] + state['dof']) / (D_ + state['dof'])
    betaw = Wt / (Wt + wt)
    Wt += wt
    Wt *= beta
    return betaw

def update_dof(z_ave, z_var, mahal_dist, beta_dof, trigamma_value, eps):
    z_k = math.log(mahal_dist + eps)
    z_var += (1.0 - beta_dof) * (z_k - z_ave)**2
    z_var *= beta_dof
    z_ave *= beta_dof
    z_ave += (1.0 - beta_dof) * z_k
    b_k = max(z_var - trigamma_value, eps)
    return (1.0 + math.sqrt(1.0 + 4.0 * b_k)) / b_k
