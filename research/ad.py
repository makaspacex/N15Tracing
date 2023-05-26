import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from argparse import Namespace

args = Namespace(method='dopri5', data_size=1000, batch_time=10, batch_size=20, niters=2000, test_freq=20, viz=False,
                 gpu=0, adjoint=False, device='cpu')

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

true_y0 = torch.tensor([[2., 0.]]).to(args.device)
t = torch.linspace(0., 25., args.data_size).to(args.device)
true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).to(args.device)


class Lambda(nn.Module):
    def forward(self, t, y):
        return torch.mm(y ** 3, true_A)


with torch.no_grad():
    true_y = odeint(Lambda(), true_y0, t, method='dopri5')


def get_batch(device):
    s = torch.from_numpy(
        np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)


class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 2),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y ** 3)


func = ODEFunc().to(args.device)

optimizer = optim.RMSprop(func.parameters(), lr=1e-3)

for itr in range(1, args.niters + 1):
    optimizer.zero_grad()
    batch_y0, batch_t, batch_y = get_batch(args.device)
    pred_y = odeint(func, batch_y0, batch_t).to(args.device)
    loss = torch.mean(torch.abs(pred_y - batch_y))
    loss.backward()
    optimizer.step()

    if itr % args.test_freq == 0:
        with torch.no_grad():
            pred_y = odeint(func, true_y0, t)
            loss = torch.mean(torch.abs(pred_y - true_y))
            print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))

    end = time.time()
