from torch import cos, sin, pow, ones_like, cat
from torch.autograd import grad


class model_parameters(object):

    def __init__(self, a, b, c, d, e):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e


def real_params_to_eff_params(m1, m2, l1, l2, lc1, lc2, Ic1, Ic2, g):
    a = Ic1 + m1*lc1*lc1 + m2*l1*l1
    b = Ic2 + m2*lc2*lc2
    c = m2*l1*lc2
    d = (m1*lc1 + m2*l1)*g
    e = m2*lc2*g

    return model_parameters(a, b, c, d, e)


class init_parameters(object):

    def __init__(self, theta_1, d_theta_1, theta_2, d_theta_2):

        self.theta_1 = theta_1
        self.d_theta_1 = d_theta_1
        self.theta_2 = theta_2
        self.d_theta_2 = d_theta_2


def d(f, x):
    # Due to the fact that PyTorch can only be done w.r.t scalar, if we want f and x to be batch of data (instead of single data), we must make following modification, and also assume that data cross batchs are independent, that is, D[f[i], x[j]] === 0, if i!=j.
    return grad(f, x, grad_outputs=ones_like(f), create_graph=True, only_inputs=True)[0]


def equation_of_motion(theta_1, theta_2, t, model_params):

    d_theta_1 = d(theta_1, t)
    d_theta_2 = d(theta_2, t)
    dd_theta_1 = d(d_theta_1, t)
    dd_theta_2 = d(d_theta_2, t)

    EoM_1 = model_params.a*dd_theta_1 + model_params.c*cos(
        theta_1 - theta_2
    )*dd_theta_2 + model_params.c*sin(theta_1 - theta_2)*pow(d_theta_2,
                                                             2) + model_params.d*sin(theta_1)

    EoM_2 = model_params.b*dd_theta_2 + model_params.c*cos(
        theta_1 - theta_2
    )*dd_theta_1 - model_params.c*sin(theta_1 - theta_2)*pow(d_theta_1,
                                                             2) + model_params.e*sin(theta_2)

    return cat([EoM_1, EoM_2], dim=1)


def initial_condition(theta_1, theta_2, t, model_params, init_params):
    # model_params is not used in here.
    d_theta_1 = d(theta_1, t)
    d_theta_2 = d(theta_2, t)

    return cat([
        theta_1 - init_params.theta_1,
        d_theta_1 - init_params.d_theta_1,
        theta_2 - init_params.theta_2,
        d_theta_2 - init_params.d_theta_1,
    ],
               dim=1)


def angle_to_xy(theta_1, theta_2, l1, l2):
    x_1 = l1*sin(theta_1)
    y_1 = -l1*cos(theta_1)

    x_2 = l1*sin(theta_1) + l2*sin(theta_2)
    y_2 = -l1*cos(theta_1) - l2*cos(theta_2)

    return cat([x_1, y_1], dim=1), cat([x_2, y_2], dim=1)
