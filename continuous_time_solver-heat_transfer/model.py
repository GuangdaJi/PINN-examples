import torch
from torch.autograd import grad


def d(f, x):

    return grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True, only_inputs=True)[0]


def EoM(u, x, t, model_params):

    kappa, = model_params

    return d(u, t) - kappa*d(d(u, x), x)


# def initial_condition(x, model_params):

#     kappa, = model_params

#     return (x > 0).to(dtype=float, device=x.device).clone().detach()


def initial_condition(x, model_params):

    kappa, = model_params

    return torch.exp(-torch.pow(x/0.1, 2)).to(dtype=torch.float, device=x.device).clone().detach()
