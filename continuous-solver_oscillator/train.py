from model import d, EoM, IC
from net import Net
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F


def PINN_train(
    t_start=0.0,
    t_end=13.0,
    model_params=(1.0, ),
    init_cond=[1.0, 0.0],
    is_load=False,
    epoch=100000,
    batch_size=5000,
    lr=0.005
):
    device = torch.device('cuda:0')

    # initial condition
    t_init = torch.tensor([[t_start]], dtype=torch.float, device=device, requires_grad=True)
    initial_condition = torch.tensor([init_cond], dtype=torch.float, device=device)

    min_loss = 100.0

    if not is_load:
        PINN = Net(1, 1).to(device)
    else:
        PINN = torch.load('./result/PINN.pth', map_location=device)

    optimizer = optim.Adam(PINN.parameters(), lr=lr)

    for i in range(epoch):

        optimizer.zero_grad()

        t = ((t_start+t_end)/2 + (t_end-t_start)*
             (torch.rand(size=(batch_size, 1), dtype=torch.float, device=device) - 0.5)
             ).requires_grad_(True)

        data = PINN(t)

        # 0-order diravative of equation of motion mse
        EoM_0 = EoM(data, t, model_params)
        mse_EoM_0 = F.mse_loss(EoM_0, torch.zeros_like(EoM_0))

        # 1-order diravative of equation of motion mse
        EoM_1 = d(EoM_0, t)
        mse_EoM_1 = F.mse_loss(EoM_1, torch.zeros_like(EoM_1))

        # 2-order diravative of equation of motion mse
        EoM_2 = d(EoM_1, t)
        mse_EoM_2 = F.mse_loss(EoM_2, torch.zeros_like(EoM_2))

        # initial condition
        mse_IC = F.mse_loss(IC(PINN(t_init), t_init, model_params), initial_condition)

        # the initial condition is harder to converge.
        # loss = 1.0*mse_EoM_0 + 1.0*mse_EoM_1 + 1.0*mse_EoM_2 + 1.0*mse_IC
        loss = 1.0*mse_EoM_0 + 1.0*mse_EoM_1 + 1.0*mse_EoM_2 + 1.0*mse_IC

        if i % 100 == 0:
            print(
                'epoch:{:05d}, EoM_0: {:.08e}, EoM_1: {:.08e}, EoM_2: {:.08e}, IC: {:.08e}, loss: {:.08e}'
                .format(
                    i, mse_EoM_0.item(), mse_EoM_1.item(), mse_EoM_2.item(), mse_IC.item(),
                    loss.item()
                )
            )

            if min_loss > loss.item():
                min_loss = loss.item()
                torch.save(PINN, './result/PINN.pth')

        loss.backward()
        optimizer.step()


def PINN_show(t_start=0.0, t_end=13.0, model_params=[1.0, 0.01], time_step=1e-2):

    device = torch.device('cuda:0')
    PINN = torch.load('./result/PINN.pth', map_location=device)

    t = torch.linspace(
        t_start, t_end, int((t_end-t_start)/time_step + 1), dtype=torch.float, device=device
    ).reshape(-1, 1)

    x = PINN(t)

    tx = torch.cat([t, x], dim=1).detach().cpu().numpy()

    pd.DataFrame(tx).to_csv('./result/tx_PINN.csv', index=False)


if __name__ == "__main__":

    t_start = 0.0
    t_end = 50.0
    model_params = [1.0]
    init_cond = [1.0, 0.0]

    PINN_train(
        t_start=t_start,
        t_end=t_end,
        model_params=model_params,
        init_cond=init_cond,
        epoch=5000,
        is_load=False
    )
    PINN_show(t_start=t_start, t_end=t_end, model_params=model_params)
