# from torch import pow, ones_like, cat, sqrt
import torch
from torch.autograd import grad


def d(f, x):
    return grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True, only_inputs=True)[0]


def Hamiltonian(x, p, t, model_params):
    omega, = model_params
    return torch.pow(p, 2)/2 + omega*omega*torch.pow(x, 2)/2
