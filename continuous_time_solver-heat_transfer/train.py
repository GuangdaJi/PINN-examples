from model import EoM, initial_condition
from net import Net
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F


def PINN_train(
    t_start=0.0,
    t_end=0.30,
    x_left=-1.0,
    x_right=1.0,
    model_params=(0.1, ),
    is_load=False,
    epoch=100000,
    n_b=1000,
    n_i=1000,
    n_f=10000,
    EoM_panelty=1.0,
    IC_panelty=10.0,
    BC_panelty=1.0,
    lr=0.001
):

    device = torch.device('cuda:0')

    min_loss = 100.0

    if not is_load:
        PINN = Net(2, 1, hidden_layers=10, width=20).to(device)
    else:
        PINN = torch.load('./result/PINN.pth', map_location=device)

    optimizer = optim.Adam(PINN.parameters(), lr=lr)
    # scheducler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.5, patience=1000, verbose=True, threshold=1e-4
    # )

    for i in range(epoch):
        optimizer.zero_grad()

        # inside
        t_f = ((t_start+t_end)/2 + (t_end-t_start)*
               (torch.rand(size=(n_f, 1), dtype=torch.float, device=device) - 0.5)
               ).requires_grad_(True)
        x_f = ((x_left+x_right)/2 + (x_right-x_left)*
               (torch.rand(size=(n_f, 1), dtype=torch.float, device=device) - 0.5)
               ).requires_grad_(True)
        u_f = PINN(torch.cat([x_f, t_f], dim=1))
        EoM_ = EoM(u_f, x_f, t_f, model_params)
        mse_EoM = F.mse_loss(EoM_, torch.zeros_like(EoM_))

        # # boundary
        # t_b = ((t_start+t_end)/2 + (t_end-t_start)*
        #        (torch.rand(size=(n_b, 1), dtype=torch.float, device=device) - 0.5)
        #        ).requires_grad_(True)
        # x_b = ((x_left+x_right)/2 + (x_right-x_left)*
        #        ((torch.rand(size=(n_b, 1), device=device) > 0.5).to(dtype=torch.float))
        #        ).requires_grad_(True)
        # u_b = PINN(torch.cat([x_b, t_b], dim=1))
        # u_b_target = (x_b > 0).to(dtype=torch.float).clone().detach()

        t_b = ((t_start+t_end)/2 + (t_end-t_start)*
               (torch.rand(size=(n_b, 1), dtype=torch.float, device=device) - 0.5)
               ).requires_grad_(True)

        x_b_l = (x_left*torch.ones_like(t_b)).requires_grad_(True)
        x_b_r = (x_right*torch.ones_like(t_b)).requires_grad_(True)
        u_b_l = PINN(torch.cat([x_b_l, t_b], dim=1))
        u_b_r = PINN(torch.cat([x_b_r, t_b], dim=1))

        mse_BC = F.mse_loss(u_b_l, u_b_r)

        # initial condition
        x_i = ((x_left+x_right)/2 + (x_right-x_left)*
               (torch.rand(size=(n_i, 1), dtype=torch.float, device=device) - 0.5)
               ).requires_grad_(True)
        t_i = (t_start*torch.ones_like(x_i)).requires_grad_(True)
        u_i = PINN(torch.cat([x_i, t_i], dim=1))
        u_i_target = initial_condition(x_i, model_params)
        mse_IC = F.mse_loss(u_i, u_i_target)

        loss = EoM_panelty*mse_EoM + IC_panelty*mse_IC + BC_panelty*mse_BC

        if i % 100 == 0:
            print(
                'epoch:{:05d}, EoM: {:.08e}, IC: {:.08e}, BC: {:.08e}, loss: {:.08e}'.format(
                    i, mse_EoM.item(), mse_IC.item(), mse_BC.item(), loss.item()
                )
            )

            if min_loss > loss.item():
                min_loss = loss.item()
                torch.save(PINN, './result/PINN.pth')

        loss.backward()
        optimizer.step()
        # scheducler.step(loss)


def PINN_show(t_start=0.0, t_end=0.3, x_left=-1.0, x_right=1.0, n_f=10000):

    # this funcion samples randomly
    device = torch.device('cuda:0')
    PINN = torch.load('./result/PINN.pth', map_location=device)

    with torch.no_grad():
        t_f = ((t_start+t_end)/2 + (t_end-t_start)*
               (torch.rand(size=(n_f, 1), dtype=torch.float, device=device) - 0.5))
        x_f = ((x_left+x_right)/2 + (x_right-x_left)*
               (torch.rand(size=(n_f, 1), dtype=torch.float, device=device) - 0.5))
        u_f = PINN(torch.cat([x_f, t_f], dim=1))

    print(torch.min(u_f))

    txu = torch.cat([t_f, x_f, u_f], dim=1).detach().cpu().numpy()

    pd.DataFrame(txu).to_csv('./result/txu_PINN.csv', index=False)


def PINN_show_grid(
    t_start=0.0, t_end=0.3, x_left=-1.0, x_right=1.0, time_display_step=100, space_display_step=100
):

    # this funcion samples randomly
    device = torch.device('cuda:0')
    PINN = torch.load('./result/PINN.pth', map_location=device)

    x = np.linspace(x_left, x_right, space_display_step, endpoint=False)
    t = np.linspace(t_start, t_end, time_display_step + 1)

    t_list, x_list = [], []

    # solving
    for i in range(time_display_step + 1):
        x_d = x.reshape(-1, 1)
        t_d = t[i]*np.ones_like(x_d)
        x_list.append(x_d)
        t_list.append(t_d)

    x_f = torch.tensor(np.concatenate(x_list), dtype=torch.float, device=device)
    t_f = torch.tensor(np.concatenate(t_list), dtype=torch.float, device=device)

    with torch.no_grad():

        u_f = PINN(torch.cat([x_f, t_f], dim=1))

    print(torch.min(u_f))

    txu = torch.cat([t_f, x_f, u_f], dim=1).detach().cpu().numpy()

    pd.DataFrame(txu).to_csv('./result/txu_PINN.csv', index=False)


if __name__ == "__main__":

    # PINN_train()
    # PINN_show()
    PINN_show_grid()
