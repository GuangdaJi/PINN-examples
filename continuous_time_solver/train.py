import model
import net
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F


def train(
    t_start=0.0,
    t_end=10.0,
    m1=1.0,
    m2=1.0,
    l1=1.0,
    l2=1.0,
    lc1=1.0,
    lc2=1.0,
    Ic1=0.0,
    Ic2=0.0,
    g=10.0,
    theta_10=np.pi/2,
    d_theta_10=0.0001,
    theta_20=np.pi/2,
    d_theta_20=0.0001
):
    device = torch.device('cuda:0')
    model_params = model.real_params_to_eff_params(m1, m2, l1, l2, lc1, lc2, Ic1, Ic2, g)
    init_params = model.init_parameters(theta_10, d_theta_10, theta_20, d_theta_20)

    epoch = 10000
    batch_size = 10000

    PINN = net.Net(1, 2).to(device)

    optimizer = optim.Adam(PINN.parameters())

    for i in range(epoch):

        t = t_start + (t_end-t_start)*torch.rand(
            size=(batch_size, 1), dtype=torch.float, device=device, requires_grad=True
        )
        t_init = torch.tensor([[t_start]], dtype=torch.float, device=device, requires_grad=True)

        data = PINN(t)

        theta_1 = data[:, 0:1]
        theta_2 = data[:, 1:2]

        EoM = model.equation_of_motion(theta_1, theta_2, t, model_params)

        data_init = PINN(t_init)

        theta_10 = data_init[:, 0:1]
        theta_20 = data_init[:, 1:2]

        IC = model.initial_condition(theta_10, theta_20, t_init, model_params, init_params)

        mse_eom = F.mse_loss(EoM, torch.zeros_like(EoM))

        mse_ic = F.mse_loss(IC, torch.zeros_like(IC))

        loss = mse_eom + mse_ic

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 50 == 0:
            print(
                'epoch:{:04d}, EoM: {:.08e}, IC: {:.08e}, loss: {:.08e}'.format(
                    i, mse_eom.item(), mse_ic.item(), loss.item()
                )
            )

    time_step = 1e-3

    t = torch.linspace(
        t_start,
        t_end,
        int((t_end-t_start)/time_step),
        dtype=torch.float,
        device=device,
        requires_grad=True
    )

    data = PINN(t)

    theta_1 = data[:, 0:1]
    theta_2 = data[:, 1:2]

    xy_1, xy_2 = model.angle_to_xy(theta_1, theta_2, l1, l2)

    txy_1 = torch.cat([t, xy_1], dim=1).detach().cpu().numpy()
    txy_2 = torch.cat([t, xy_2], dim=1).detach().cpu().numpy()

    pd.DataFrame(txy_1).to_csv('txy_1.csv', index=False)
    pd.DataFrame(txy_2).to_csv('txy_2.csv', index=False)

    torch.save(PINN, 'PINN.pth')


if __name__ == "__main__":
    train()
