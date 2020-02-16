from torch import pow, ones_like, cat, sqrt
from torch.autograd import grad


def d(f, x):
    # Due to the fact that PyTorch can only be done w.r.t scalar, if we want f and x to be batch of data (instead of single data), we must make following modification, and also assume that data cross batchs are independent, that is, D[f[i], x[j]] === 0, if i!=j, and f must be an Nx1 vector, instead of matrix.
    return grad(f, x, grad_outputs=ones_like(f), create_graph=True, only_inputs=True)[0]


def EoM_right(data, model_params):

    # calculate the right hand side of equation of motion
    f, e = model_params

    x, y, dx, dy = data[:, 0:1], data[:, 1:2], data[:, 2:3], data[:, 3:4]

    r = sqrt(x*x + y*y)

    ddx = -f*x*pow(r, -3) - e*x*pow(r, -4)
    ddy = -f*y*pow(r, -3) - e*y*pow(r, -4)

    return cat([dx, dy, ddx, ddy], dim=1)


def data_to_txy(data, t):

    x, y = data[:, 0:1], data[:, 1:2]

    return cat([t, x, y], dim=1).detach().cpu().numpy()
