from torch import cos, sin, pow, ones_like, cat
from torch.autograd import grad


def d(f, x):
    # Due to the fact that PyTorch can only be done w.r.t scalar, if we want f and x to be batch of data (instead of single data), we must make following modification, and also assume that data cross batchs are independent, that is, D[f[i], x[j]] === 0, if i!=j, and f must be an Nx1 vector, instead of matrix.
    return grad(f, x, grad_outputs=ones_like(f), create_graph=True, only_inputs=True)[0]


def EoM(data, model_params):

    a, b, c, d, e = model_params

    a1, a2, da1, da2 = data[:, 0:1], data[:, 1:2], data[:, 2:3], data[:, 3:4]

    dominator = a*b - c*c*pow(cos(a1 - a2), 2)

    dda1 = -b*d*sin(a1) + c*e*cos(a1 - a2)*sin(a2) - c*c*cos(a1 - a2)*sin(a1 - a2)*pow(
        da1, 2
    ) - b*c*sin(a1 - a2)*pow(da2, 2)
    dda1 = dda1/dominator

    dda2 = -a*e*sin(a2) + c*d*cos(a1 - a2)*sin(a1) + c*c*cos(a1 - a2)*sin(a1 - a2)*pow(
        da2, 2
    ) + a*c*sin(a1 - a2)*pow(da1, 2)
    dda2 = dda2/dominator

    return cat([da1, da2, dda1, dda2], dim=1)


def real_params_to_eff_params(real_params):

    m1, m2, l1, l2, lc1, lc2, Ic1, Ic2, g = real_params

    a = Ic1 + m1*lc1*lc1 + m2*l1*l1
    b = Ic2 + m2*lc2*lc2
    c = m2*l1*lc2
    d = (m1*lc1 + m2*l1)*g
    e = m2*lc2*g

    return a, b, c, d, e


def angle_to_txy(data, t, l1, l2):
    a1 = data[:, 0:1]
    a2 = data[:, 1:2]

    x_1 = l1*sin(a1)
    y_1 = -l1*cos(a2)

    x_2 = l1*sin(a1) + l2*sin(a2)
    y_2 = -l1*cos(a1) - l2*cos(a2)

    return cat([t, x_1, y_1], dim=1).detach().cpu().numpy(), cat([t, x_2, y_2],
                                                                 dim=1).detach().cpu().numpy()
