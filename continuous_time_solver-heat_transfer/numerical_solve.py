import numpy as np
from scipy.sparse import dia_matrix, eye
from scipy.sparse.linalg import spsolve
import pandas as pd


def Dxx(n):
    offsets = np.array([0, -1, 1, n - 1, 1 - n])
    data = np.array([-2*np.ones(n), np.ones(n), np.ones(n), np.ones(n), np.ones(n)])
    return dia_matrix((data, offsets), shape=(n, n))


if __name__ == "__main__":

    x_left = -1.0
    x_right = 1.0
    t_start = 0.0
    t_end = 0.3

    time_display_step = 100
    space_display_step = 100

    kappa = 0.1

    n = 100

    space_cal_step = space_display_step*n
    time_cal_step = time_display_step*n

    idx_display = np.arange(0, space_cal_step, n)

    dx = (x_right-x_left)/space_cal_step
    dt = (t_end-t_start)/time_cal_step

    x = np.linspace(x_left, x_right, space_cal_step, endpoint=False)
    t = np.linspace(t_start, t_end, time_cal_step + 1)
    u = np.empty(shape=(time_cal_step + 1, space_cal_step))

    ep = kappa*dt/(dx*dx*2)
    dxx = Dxx(space_cal_step)
    identity = eye(space_cal_step)
    A_l = identity - ep*dxx
    A_r = identity + ep*dxx

    # initialization
    u[0, :] = np.exp(-np.power(x/0.1, 2))

    display_points = []

    # solving
    for i in range(0, time_cal_step - 1):
        u[i + 1, :] = spsolve(A=A_l, b=A_r.dot(u[i, :]))

        if i % n == 0:
            print(t[i])

            u_d = u[i, idx_display].reshape(-1, 1)
            x_d = x[idx_display].reshape(-1, 1)
            t_d = t[i]*np.ones_like(x_d)

            display_points.append(np.concatenate([t_d, x_d, u_d], axis=1))

    display_points = np.concatenate(display_points)

    pd.DataFrame(display_points).to_csv('./result/txu_NUMERICAL.csv', index=False)
