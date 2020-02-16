from model import d, EoM, real_params_to_eff_params, angle_to_txy
from net import Net
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F


def PINN_train(
    t_start=0.0,
    t_end=5.0,
    real_params=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 10.0],
    init_cond=[np.pi/2, -np.pi/2, 0.0, 0.0],
    is_load=False,
    epoch=50000,
    batch_size=5000,
    EoM_panelty=1.0,
    IC_panelty=10.0,
    lr=0.001
):
    device = torch.device('cuda:0')
    model_params = real_params_to_eff_params(real_params)

    initial_condition = torch.tensor([init_cond]).to(device)

    min_loss = 100.0

    if not is_load:
        PINN = Net(1, 4).to(device)
    else:
        PINN = torch.load('PINN.pth', map_location=device)

    optimizer = optim.Adam(PINN.parameters(), lr=lr)
    scheducler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5000, verbose=True, threshold=1e-3
    )

    for i in range(epoch):

        t = (t_start+t_end)/2 + 1.1*(t_end-t_start)*(
            torch.rand(size=(batch_size, 1), dtype=torch.float, device=device, requires_grad=True) -
            0.5
        )

        optimizer.zero_grad()

        data = PINN(t)

        # left hand side of EoM
        left = torch.cat([
            d(data[:, 0:1], t),
            d(data[:, 1:2], t),
            d(data[:, 2:3], t),
            d(data[:, 3:4], t)
        ],
                         dim=1)

        # right hand side of EoM
        right = EoM(data, model_params)

        # equation of motion mse
        mse_EoM = F.mse_loss(left, right)

        t_init = torch.tensor([[t_start]], dtype=torch.float, device=device, requires_grad=True)

        # initial condition mse
        mse_IC = F.mse_loss(PINN(t_init), initial_condition)

        # the initial condition is harder to converge.
        loss = EoM_panelty*mse_EoM + IC_panelty*mse_IC

        loss.backward()
        optimizer.step()
        scheducler.step(loss)

        if i % 100 == 0:
            print(
                'epoch:{:05d}, EoM: {:.08e}, IC: {:.08e}, loss: {:.08e}'.format(
                    i, mse_EoM.item(), mse_IC.item(), loss.item()
                )
            )

            if min_loss > loss.item():
                min_loss = loss.item()
                torch.save(PINN, 'PINN.pth')


def PINN_show(
    t_start=0.0,
    t_end=5.0,
    real_params=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 10.0],
    time_step=1e-2
):

    device = torch.device('cuda:0')
    PINN = torch.load('PINN.pth', map_location=device)

    t = torch.linspace(
        t_start, t_end, int((t_end-t_start)/time_step + 1), dtype=torch.float, device=device
    ).reshape(-1, 1)

    data = PINN(t)

    txy_1, txy_2 = angle_to_txy(data, t, real_params[2], real_params[3])

    pd.DataFrame(txy_1).to_csv('txy_1_PINN.csv', index=False)
    pd.DataFrame(txy_2).to_csv('txy_2_PINN.csv', index=False)


def RK4_solve(
    t_start=0.0,
    t_end=5.0,
    real_params=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 10.0],
    init_cond=[np.pi/2, -np.pi/2, 0.0, 0.0],
    time_step=1e-2,
    precision=1e-5
):
    model_params = real_params_to_eff_params(real_params)

    y = torch.tensor([init_cond])
    t = torch.tensor([[t_start]])

    txy_1, txy_2 = angle_to_txy(y, t, real_params[2], real_params[3])

    with open('txy_1_RK4.csv', 'w') as f:
        f.write('t,x,y\n')
        f.write('{:.10f},{:.10f},{:.10f}\n'.format(txy_1[0, 0], txy_1[0, 1], txy_1[0, 2]))

    with open('txy_2_RK4.csv', 'w') as f:
        f.write('t,x,y\n')
        f.write('{:.10f},{:.10f},{:.10f}\n'.format(txy_2[0, 0], txy_2[0, 1], txy_2[0, 2]))

    step_per_record = int(time_step/precision)

    n = 0
    while t <= t_end + 1e-8:

        with torch.no_grad():
            k1 = precision*EoM(y, model_params)
            k2 = precision*EoM(y + 0.5*k1, model_params)
            k3 = precision*EoM(y + 0.5*k2, model_params)
            k4 = precision*EoM(y + k3, model_params)
            y = y + k1/6 + k2/3 + k3/3 + k4/6
            t = t + precision

        if n % step_per_record == 0:

            print(t.item())

            txy_1, txy_2 = angle_to_txy(y, t, real_params[2], real_params[3])

            with open('txy_1_RK4.csv', 'a') as f:
                f.write('{:.10f},{:.10f},{:.10f}\n'.format(txy_1[0, 0], txy_1[0, 1], txy_1[0, 2]))

            with open('txy_2_RK4.csv', 'a') as f:
                f.write('{:.10f},{:.10f},{:.10f}\n'.format(txy_2[0, 0], txy_2[0, 1], txy_2[0, 2]))

        n = n + 1


if __name__ == "__main__":

    # PINN_train()
    # PINN_show()
    RK4_solve()
