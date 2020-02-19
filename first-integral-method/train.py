from model import d, Hamiltonian
from net import Net
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
import itertools


def HJE_Solve(
    t_mean=6.0,
    t_std=12.0,
    x_mean=0.0,
    x_std=2.0,
    p_mean=0.0,
    p_std=2.0,
    model_params=(1.0, ),
    is_load=False,
    epoch=50000,
    batch_size=50000,
    lr=0.001
):
    device = torch.device('cuda:0')

    min_loss = 10000000.0

    if is_load:
        alpha_net = torch.load('./result/alpha_net.pth', map_location=device)
        S_net = torch.load('./result/S_net.pth', map_location=device)
    else:
        alpha_net = Net(3, 1).to(device)
        S_net = Net(3, 1).to(device)

    params = itertools.chain(alpha_net.parameters(), S_net.parameters())
    optimizer = optim.Adam(params, lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=1000, threshold=1e-3, factor=0.5, verbose=True, min_lr=1e-5
    )
    # alpha_optimizer = optim.Adam(alpha_net.parameters(), lr=lr)
    # S_optimizer = optim.Adam(S_net.parameters(), lr=lr)

    for i in range(epoch):

        # t = (t_mean + t_std*torch.randn(size=(batch_size, 1), dtype=torch.float, device=device)
        #      ).requires_grad_(True)
        # x = (x_mean + x_std*torch.randn(size=(batch_size, 1), dtype=torch.float, device=device)
        #      ).requires_grad_(True)
        # p = (p_mean + p_std*torch.randn(size=(batch_size, 1), dtype=torch.float, device=device)
        #      ).requires_grad_(True)

        t = (
            t_mean + 2*t_std*torch.rand(size=(batch_size, 1), dtype=torch.float, device=device) -
            t_std
        ).requires_grad_(True)
        x = (
            x_mean + 2*x_std*torch.rand(size=(batch_size, 1), dtype=torch.float, device=device) -
            x_std
        ).requires_grad_(True)
        p = (
            p_mean + 2*p_std*torch.rand(size=(batch_size, 1), dtype=torch.float, device=device) -
            p_std
        ).requires_grad_(True)

        alpha = alpha_net(torch.cat([x, p, t], dim=1))

        x_eff = x + 0.0
        t_eff = t + 0.0

        S = S_net(torch.cat([x_eff, alpha, t_eff], dim=1))

        partial_x_S = d(S, x_eff)
        partial_t_S = d(S, t_eff)

        HJE = Hamiltonian(x, partial_x_S, t, model_params) + partial_t_S

        loss_hamiltonian = F.mse_loss(HJE, torch.zeros_like(HJE))

        loss_p = F.mse_loss(partial_x_S, p)

        init = torch.tensor([[x_mean, p_mean, t_mean]], dtype=torch.float, device=device)
        S_init = S_net(init)

        loss_init = F.mse_loss(S_init, torch.zeros_like(S_init))

        loss = 1.0*loss_hamiltonian + 1.0*loss_p + 1.0*loss_init

        if i % 100 == 0:

            print(
                'epoch:{:05d}, loss_hamiltonian: {:.08e}, loss_p: {:.08e}, loss_init: {:.08e}, loss: {:.08e}'
                .format(i, loss_hamiltonian.item(), loss_p.item(), loss_init.item(), loss.item())
            )

            if min_loss > loss.item():
                min_loss = loss.item()
                torch.save(alpha_net, './result/alpha_net.pth')
                torch.save(S_net, './result/S_net.pth')

        # alpha_optimizer.zero_grad()
        # loss_p.backward(retain_graph=True)
        # # loss_p.backward()
        # alpha_optimizer.step()

        # S_optimizer.zero_grad()
        # # loss_p.backward(retain_graph=True)
        # loss_hamiltonian.backward()
        # S_optimizer.step()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)


def Orbit_Solve(
    t_start=0.0,
    t_end=25.0,
    x_start=1.0,
    p_start=0.0,
    model_params=(1.0, ),
    epoch=80000,
    batch_size=5000,
    lr=0.001
):
    device = torch.device('cuda:0')

    min_loss = 10000000.0

    # alpha_net = torch.load('./result/alpha_net.pth', map_location=device)
    # S_net = torch.load('./result/S_net.pth', map_location=device)

    orbit_net = Net(input_dim=1, output_dim=2).to(device)

    optimizer = optim.Adam(orbit_net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=1000, threshold=1e-3, factor=0.8, verbose=True, min_lr=1e-5
    )

    for i in range(epoch):

        t = ((t_start+t_end)/2 + (t_end-t_start)*
             (torch.rand(size=(batch_size, 1), dtype=torch.float, device=device) - 0.5)
             ).requires_grad_(True)
        output = orbit_net(t)
        x = output[:, 0:1]
        p = output[:, 1:2]
        # a = alpha_net(torch.cat([x, p, t], dim=1))
        # S = S_net(torch.cat([x, a, t], dim=1))
        # b = d(S, a)
        H = Hamiltonian(x, p, t, model_params)
        dx_l = d(x, t)
        dx_r = d(H, p)
        dp_l = d(p, t)
        dp_r = -d(H, x)
        loss_h = F.mse_loss(dx_l, dx_r) + F.mse_loss(dp_l, dp_r)

        # t_init = t_start*torch.ones_like(t)
        # x_init = x_start*torch.ones_like(x)
        # p_init = p_start*torch.ones_like(p)
        # a_init = alpha_net(torch.cat([x_init, p_init, t_init], dim=1))
        # S_init = S_net(torch.cat([x_init, a_init, t_init], dim=1))
        # b_init = d(S_init, a_init)

        # loss_a = F.mse_loss(a, a_init)
        # loss_b = F.mse_loss(b, b_init)

        t_init = torch.tensor([[t_start]], dtype=torch.float, device=device)
        x_init = torch.tensor([[x_start]], dtype=torch.float, device=device)
        p_init = torch.tensor([[p_start]], dtype=torch.float, device=device)
        output_init = orbit_net(t_init)
        x_out = output_init[:, 0:1]
        p_out = output_init[:, 1:2]
        loss_init = F.mse_loss(x_out, x_init) + F.mse_loss(p_out, p_init)
        H_init = Hamiltonian(x_init, p_init, t_init, model_params)[0, 0]
        loss_energy = F.mse_loss(H, H_init*torch.ones_like(H))

        loss = loss_h + loss_energy + loss_init

        if i % 100 == 0:

            print(
                'epoch:{:05d}, loss_h: {:.08e}, loss_e: {:.08e}, loss_i: {:.08e}, loss: {:.08e}'
                .format(i, loss_h.item(), loss_energy.item(), loss_init.item(), loss.item())
            )

            if min_loss > loss.item():
                min_loss = loss.item()
                torch.save(orbit_net, './result/orbit_net.pth')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)


def show(t_start=0.0, t_end=25.0, model_params=(1.0, ), time_step=1e-2):

    device = torch.device('cuda:0')
    orbit_net = torch.load('./result/orbit_net.pth', map_location=device)

    t = torch.linspace(
        t_start, t_end, int((t_end-t_start)/time_step + 1), dtype=torch.float, device=device
    ).reshape(-1, 1)

    output = orbit_net(t)
    x = output[:, 0:1]
    print(output.shape)

    tx = torch.cat([t, x], dim=1).detach().cpu().numpy()

    pd.DataFrame(tx).to_csv('./result/tx_PINN.csv', index=False)


if __name__ == "__main__":

    # HJE_Solve()
    Orbit_Solve()
    show()
