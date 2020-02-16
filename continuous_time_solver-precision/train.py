from model import d, EoM_right, data_to_txy
from net import Net
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F


def PINN_train(
    t_start=0.0,
    t_end=50.0,
    model_params=[1.0, 0.01],
    init_cond=[1.0, 0.0, 0.0, 0.3],
    is_load=False,
    epoch=100000,
    batch_size=5000,
    EoM_panelty=1.0,
    IC_panelty=1.0,
    lr=0.005
):
    device = torch.device('cuda:0')

    initial_condition = torch.tensor([init_cond]).to(device)

    min_loss = 100.0

    if not is_load:
        PINN = Net(1, 4, hidden_layers=10).to(device)
    else:
        PINN = torch.load('./result/PINN.pth', map_location=device)

    optimizer = optim.Adam(PINN.parameters(), lr=lr)
    scheducler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1000, verbose=True, threshold=1e-3
    )

    for i in range(epoch):

        t = (t_start+t_end)/2 + 1.1*(t_end-t_start)*(
            torch.rand(size=(batch_size, 1), dtype=torch.float, device=device, requires_grad=True) -
            0.5
        )

        optimizer.zero_grad()

        data = PINN(t)

        EoM = torch.cat([
            d(data[:, 0:1], t),
            d(data[:, 1:2], t),
            d(data[:, 2:3], t),
            d(data[:, 3:4], t)
        ],
                        dim=1) - EoM_right(data, model_params)

        # equation of motion mse
        mse_EoM = F.mse_loss(EoM, torch.zeros_like(EoM))

        t_init = torch.tensor([[t_start]], dtype=torch.float, device=device, requires_grad=True)

        # initial condition mse
        mse_IC = F.mse_loss(PINN(t_init), initial_condition)

        # the initial condition is harder to converge.
        loss = EoM_panelty*mse_EoM + IC_panelty*mse_IC

        if i % 100 == 0:
            print(
                'epoch:{:05d}, EoM: {:.08e}, IC: {:.08e}, loss: {:.08e}'.format(
                    i, mse_EoM.item(), mse_IC.item(), loss.item()
                )
            )

            if min_loss > loss.item():
                min_loss = loss.item()
                torch.save(PINN, './result/PINN.pth')

        loss.backward()
        optimizer.step()
        scheducler.step(loss)


def PINN_show(t_start=0.0, t_end=50.0, model_params=[1.0, 0.01], time_step=1e-2):

    device = torch.device('cuda:0')
    PINN = torch.load('./result/PINN.pth', map_location=device)

    t = torch.linspace(
        t_start, t_end, int((t_end-t_start)/time_step + 1), dtype=torch.float, device=device
    ).reshape(-1, 1)

    data = PINN(t)

    txy = data_to_txy(data, t)

    pd.DataFrame(txy).to_csv('./result/txy_PINN.csv', index=False)


def RK4_solve(
    t_start=0.0,
    t_end=50.0,
    model_params=[1.0, 0.01],
    init_cond=[1.0, 0.0, 0.0, 0.3],
    time_step=1e-2,
    precision=1e-4
):

    t_list = torch.linspace(t_start, t_end, int((t_end-t_start)/precision + 1),
                            dtype=torch.float).reshape(-1, 1)

    f = torch.tensor([init_cond])

    print(f.detach().cpu().numpy())

    with open('./result/txy_RK4.csv', 'w') as file:
        file.write('t,x,y\n')

    step_per_record = int(time_step/precision)

    for i in range(len(t_list)):

        if i % step_per_record == 0:

            txy = data_to_txy(f, t_list[i:i + 1, :])

            print(t_list[i, 0].item())

            with open('./result/txy_RK4.csv', 'a') as file:
                file.write('{:.10f},{:.10f},{:.10f}\n'.format(txy[0, 0], txy[0, 1], txy[0, 2]))

        with torch.no_grad():
            k1 = precision*EoM_right(f, model_params)
            k2 = precision*EoM_right(f + 0.5*k1, model_params)
            k3 = precision*EoM_right(f + 0.5*k2, model_params)
            k4 = precision*EoM_right(f + k3, model_params)
            f = f + k1/6 + k2/3 + k3/3 + k4/6


if __name__ == "__main__":

    t_start = 0.0
    t_end = 4.0
    model_params = [1.0, 0.01]
    init_cond = [1.0, 0.0, 0.0, 0.3]

    # PINN_train(
    #     t_start=t_start, t_end=t_end, model_params=model_params, init_cond=init_cond, is_load=False
    # )
    PINN_show(t_start=t_start, t_end=t_end, model_params=model_params)
    RK4_solve(t_start=t_start, t_end=t_end, model_params=model_params, init_cond=init_cond)
