from torch import cos, sin, tensor, pow
from torch.autograd import grad


class model_parameters(object):

    def __init__(self, a, b, c, d, e):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e


class init_parameters(object):

    def __init__(self, theta_1, d_theta_1, theta_2, d_theta_2):

        self.theta_1 = theta_1
        self.d_theta_1 = d_theta_1
        self.theta_2 = theta_2
        self.d_theta_2 = d_theta_2


def d(f, x):
    return grad(f, x, create_graph=True, only_inputs=True)[0]


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

    return tensor([EoM_1, EoM_2])


def initial_condition(theta_1, theta_2, t, model_params, init_params):
    # model_params is not used in here.
    d_theta_1 = d(theta_1, t)
    d_theta_2 = d(theta_2, t)

    return tensor([
        theta_1 - init_params.theta_1,
        d_theta_1 - init_params.d_theta_1,
        theta_2 - init_params.theta_2,
        d_theta_2 - init_params.d_theta_1,
    ])
