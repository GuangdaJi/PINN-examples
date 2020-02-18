# from torch import pow, ones_like, cat, sqrt
import torch
from torch.autograd import grad


def d(f, x):
    # Due to the fact that PyTorch can only be done w.r.t scalar, if we want f and x to be batch of data (instead of single data), we must make following modification, and also assume that data cross batchs are independent, that is, D[f[i], x[j]] === 0, if i!=j, and f must be an Nx1 vector, instead of matrix.
    return grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True, only_inputs=True)[0]


def EoM(data, t, model_params):

    # calculate the right hand side of equation of motion
    k, = model_params

    x = data[:, 0:1]

    ddx = d(d(x, t), t)

    return ddx + k*x


def IC(data, t, model_params):

    k, = model_params

    x = data[:, 0:1]

    dx = d(x, t)

    return torch.cat([x, dx], dim=1)
