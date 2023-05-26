#! /usr/bin/python
# -*- coding: utf-8 -*-
# @author izhangxm
# @date 2023/3/6
# @fileName torchdiff.py
# Copyright 2017 izhangxm@gmail.com. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
import argparse
import time
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from argparse import Namespace
from scipy.integrate import odeint as scipy_odeint
from torchdiffeq import odeint_adjoint as tf_odeint_adj
from torchdiffeq import odeint as tf_odeint
from tqdm import tqdm


def get_observed_data(a=1, b=0.1, c=1.5, d=0.75):
    # Definition of parameters
    # a, b, c, d = 1.0, 0.1, 1.5, 0.75

    # initial population of rabbits and foxes
    y0 = [10.0, 5.0]
    # size of data
    size = 200
    # time lapse
    time = 15
    t = np.linspace(0, time, size)

    # Lotka - Volterra equation
    def dX_dt(y, t, a, b, c, d):
        """Return the growth rate of fox and rabbit populations."""
        return np.array([a * y[0] - b * y[0] * y[1], -c * y[1] + d * b * y[0] * y[1]])

    observed = scipy_odeint(dX_dt, y0=y0, t=t, rtol=0.01, args=(a, b, c, d))
    noise = np.random.normal(size=(size, 2))
    observed += noise

    return t, y0, observed


t, y0, observed = get_observed_data()

def plot_dataset(t, observed, pred=None):
    _, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, observed[:, 0], "x", label="prey")
    ax.plot(t, observed[:, 1], "x", label="predator")

    if pred is not None:
        ax.plot(t, pred[:, 0], c="C0", label="prey")
        ax.plot(t, pred[:, 1], c="C1", label="predator")

    ax.set_xlabel("time")
    ax.set_ylabel("population")
    ax.set_title("Observed data")
    ax.legend()
    plt.show()


class CompetitionModel(torch.nn.Module):
    def __init__(self):
        super(CompetitionModel, self).__init__()
        # a:1.0, b:0.1, c:1.5, d:0.75
        self.a = nn.Parameter(torch.rand(1)[0])
        setattr(self.a, 'constrain', [0, 1.5])

        self.b = nn.Parameter(torch.rand(1)[0])
        setattr(self.b, 'constrain', [0, 2])

        self.c = nn.Parameter(torch.rand(1)[0])
        setattr(self.c, 'constrain', [0, 2])

        self.d = nn.Parameter(torch.rand(1)[0])
        setattr(self.d, 'constrain', [0, 2])

        self.ys = nn.Parameter(torch.tensor([10., 5.]))

    def forward(self, t, y):
        # np.array([a * y[0] - b * y[0] * y[1], -c * y[1] + d * b * y[0] * y[1]])
        # w = self.w.type_as(y).to(y.device).clone().detach()

        if t == 0:
            y = self.ys

        y0, y1 = y

        # a * y[0] - b * y[0] * y[1]
        _y0 = self.a * y0 - self.b * y0 * y1

        # -c * y[1] + d * b * y[0] * y[1]
        _y1 = -1 * self.c * y1 + self.d * self.b * y0 * y1

        _y = torch.stack([_y0, _y1])
        return _y



class N15Loss(nn.Module):

    def __init__(self):
        super(N15Loss, self).__init__()

    def forward(self, pred, target):
        target_mean = torch.mean(target, dim=0)
        ss_tot = torch.sum(torch.abs(target - target_mean)**2, dim=0)
        ss_res = torch.sum(torch.abs(pred - target)**2, dim=0)

        p = ss_res / ss_tot  # 0-1, 1 is best
        r2_loss = torch.abs(1 - p)  # so loss of 1 is 0

        _l = torch.mean(r2_loss)
        # _l = torch.mean(torch.abs(pred - target))
        return _l

    # def forward(self, pred, target):
    #     target_mean = torch.mean(target, dim=0)
    #     ss_tot = torch.sum(torch.abs(target - target_mean), dim=0)
    #     ss_res = torch.sum(torch.abs(pred - target), dim=0)
    #
    #     p = ss_res / ss_tot  # 0-1, 1 is best
    #     r2_loss = torch.abs(1 - p)  # so loss of 1 is 0
    #
    #     # torch.mean(torch.abs(pred - target) * self.epsilon)/100
    #     return torch.mean(r2_loss)




ode_func = CompetitionModel()
device = "cpu"
params = ode_func.parameters()

batch_y0, batch_t, batch_y = torch.tensor(y0, dtype=torch.float32).to(device), torch.tensor(t, dtype=torch.float32).to(
    device), torch.tensor(observed, dtype=torch.float32).to(device)

# optimizer = optim.Adam(params, lr=0.001, betas=(0,0.99))

optimizer = optim.Adam(params, lr=0.001)

# loss_func = nn.MSELoss()
# loss_func = N15Loss()
loss_func = nn.HuberLoss()

epoch = 100000
pbar = tqdm(total=epoch, ascii=True)

vis = False

for itr in range(epoch):
    optimizer.zero_grad()
    pred_y = tf_odeint(ode_func, batch_y0, batch_t, method="dopri5").to(device)
    loss = loss_func(pred_y, batch_y)
    # loss = torch.mean(torch.abs(pred_y - batch_y))
    loss.backward()
    optimizer.step()

    for p in ode_func.parameters():
        if not hasattr(p, 'constrain'):
            continue
        cons = getattr(p, 'constrain')
        if p.data is None:
            p.data = torch.rand(1)[0]
        p.data.clamp_(cons[0], cons[1])

    with torch.no_grad():
        _real = f"a:1.0, b:0.1, c:1.5, d:0.75"
        _now = ", ".join([f"{n}:{p.detach().numpy()}" for n, p in ode_func.named_parameters()])
        pbar.set_description(f'Iter {itr:04d} | Total Loss {loss.item():.6f} real:{_real} now:{_now}')
        pbar.update(1)
        pbar.refresh()
        if itr % 100 == 0:
            plot_dataset(t, observed, pred_y)

_now = ", ".join([f"{n}:{p.detach().numpy()}" for n, p in ode_func.named_parameters()])
print(_now)
