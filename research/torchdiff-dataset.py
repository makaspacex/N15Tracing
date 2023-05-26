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
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import  matplotlib
import matplotlib.pyplot as plt
from argparse import Namespace
from scipy.integrate import odeint as scipy_odeint
from torchdiffeq import odeint_adjoint as tf_odeint_adj
from torchdiffeq import odeint as tf_odeint
from tqdm import tqdm
import arviz as az
import pandas as pd
import pymc as pm
import pickle
import core
from torchdiffeq import odeint_event
from torchvision import models
import torchsummary
from core import DynamicShowPlot

db_csv_path = "dataset/data.csv"
idata_save_path = "odes-exp04-idata-4-number-1core-c0number-halfnormks-from-core.py-success.dt"

dataset_ori = core.MyDataset(db_csv_path)
df_ori = dataset_ori.get_df()
df = dataset_ori.get_df()
# t_eval = np.array([0.5, 48, 96, 144])

k_kinetics = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).astype(np.uint8)
dataset = dataset_ori
t_eval = df['time'].values
cct_names, _, _ = dataset.get_var_col_names()
c0 = df[cct_names].iloc[0].values
# ------------ simulate data -----------------------


# ks_true = np.array(
#     [0.00071942, 0.00269696, 0.00498945, 0.00444931, 0.00571299, 0.00801272, 0.00131931, 0.00319959, 0.00415571,
#      0.00228432, 0.00177611])
# dataset = core.MyDataset(db_csv_path)
# dataset.set_as_sim_dataset(t_eval, c0, t0=0.5, args=(ks_true, k_kinetics))

# --------------------------------
df = dataset.get_df()
ccts = df[cct_names].values
core.plot_dataset(dataset, dataset)



#
#
# resnet18 = models.resnet18(pretrained=True)
# num_ftrs = resnet18.fc.in_features
# resnet18.fc = nn.Linear(num_ftrs, 11)
# reset_input = torch.as_tensor(ccts, dtype=torch.float32).unsqueeze(0).repeat(3,1,1).unsqueeze(0)


class KNet(torch.nn.Module):
    def __init__(self):
        super(KNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(40, 256),
            nn.Tanh(),
            nn.Linear(256, 11),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, ccts):
        x = self.net(ccts)
        return x


class N15TracingModel_V2(torch.nn.Module):

    def __init__(self):
        super(N15TracingModel_V2, self).__init__()

        self.ks = nn.Parameter(torch.rand(11))
        setattr(self.ks, 'constrain', [1e-6, 0.01])

        self.kk = nn.Parameter(torch.rand(11))
        setattr(self.kk, 'constrain', [0, 1])

        self.post_opti()
        self.t_eval = t_eval

        self.eval_times = 0

    def post_opti(self, reset=False):

        def _constrain_data(p,cons):
            if torch.isnan(p.data):
                p.data = torch.rand(1)[0]
            if p.data < 0:
                p.data = torch.tensor(cons[0])
            if p.data > cons[1] or p.data < cons[0]:
                p.data.uniform_(cons[0],  cons[1])
            return p

        with torch.no_grad():
            cons = getattr(self.ks, 'constrain')
            for i, k in enumerate(self.ks):
                self.ks[i] = _constrain_data(k, cons)

            cons = getattr(self.kk, 'constrain')
            for i, k in enumerate(self.kk):
                self.kk[i] = _constrain_data(k, cons)

        self.eval_times = 0


    def forward(self, t, c):
        self.eval_times += 1

        ks, kk = self.ks, self.kk

        c_xNH3, c_xNO3, c_xNO2, c_xNOrg, c_xN2, c_ANH3, c_ANO3, c_ANO2, c_ANOrg, c_AN2 = c


        r1 = kk[0] * (ks[0] * c_xN2) + (1 - kk[0])* ks[0]
        r2 = kk[1] * (ks[1] * c_xNH3) + (1 - kk[1]) * ks[1]
        r3 = kk[2] * (ks[2] * c_xNO2) + (1 - kk[2])* ks[2]
        r4 = kk[3] * (ks[3] * c_xNO3) + (1 - kk[3])* ks[3]
        r5 = kk[4] * (ks[4] * c_xNO2) + (1 - kk[4])* ks[4]
        r6 = kk[5] * (ks[5] * c_xNO2 * c_xNO3) + (1 - kk[5])* ks[5]
        r7 = kk[6] * (ks[6] * c_xNO3) + (1 - kk[6])* ks[6]
        r8 = kk[7] * (ks[7] * c_xNO3) + (1 - kk[7])* ks[7]
        r9 = kk[8] * (ks[8] * c_xNH3) + (1 - kk[8])* ks[8]
        r10 = kk[9] * (ks[9] * c_xNOrg) + (1 - kk[9])* ks[9]
        r11 = kk[10] * (ks[10] * c_xNOrg) + (1 - kk[10])* ks[10]


        dc_xNH3 = 2 * r1 + r7 + r10 - r2 - r6 - r9
        dc_xNO3 = r3 - r7 - r4 - r8 + r11
        dc_xNO2 = r2 + r4 - r3 - r6 - 2 * r5
        dc_xNOrg = r8 + r9 - r10 - r11
        dc_xN2 = r5 + r6 - r1
        dc_ANH3 = (2 * r1 * (c_AN2 - c_ANH3) + (c_ANO3 - c_ANH3) * r7 + (c_ANOrg - c_ANH3) * r10) / c_xNH3
        dc_ANO3 = ((c_ANO2 - c_ANO3) * r2 + (c_ANOrg - c_ANO3) * r11) / c_xNO3
        dc_ANO2 = ((c_ANH3 - c_ANO2) * r2 + (c_ANO3 - c_ANO2) * r4) / c_xNO2
        dc_ANOrg = ((c_ANO3 - c_ANOrg) * r8 + (c_ANH3 - c_ANOrg) * r9) / c_xNOrg
        dc_AN2 = ((c_ANO2 - c_AN2) * r5 + (c_ANO2 * c_ANH3 - c_AN2) * r6) / c_xN2

        dcdts = [dc_xNH3, dc_xNO3, dc_xNO2, dc_xNOrg, dc_xN2, dc_ANH3, dc_ANO3, dc_ANO2, dc_ANOrg, dc_AN2]
        _y = torch.stack(dcdts)
        return _y

class N15TracingModel_V1(torch.nn.Module):

    def __init__(self, k_kinetics, t_eval):
        super(N15TracingModel_V1, self).__init__()

        self.ks = nn.Parameter(torch.rand(11))
        setattr(self.ks, 'constrain', [1e-6, 0.01])

        self.k_kinetics = k_kinetics
        self.post_opti()
        self.t_eval = t_eval

        self.eval_times = 0

    def post_opti(self, reset=False):
        with torch.no_grad():
            cons = getattr(self.ks, 'constrain')
            for i, k in enumerate(self.ks):
                if torch.isnan(k.data):
                    k.data = torch.rand(1)[0]
                if k.data < 0:
                    k.data = torch.tensor(cons[0])
                if k.data > cons[1] or k.data < cons[0]:
                    k.data.uniform_(cons[0], cons[1])

                self.ks[i] = k

        self.eval_times = 0

    def event_fn(self, t, y):

        _d_max = y.max()
        _d_min = y.min()
        if t > self.t_eval[-1] or _d_max > 2000 or _d_min < 0:
            return 0

        return torch.tensor(1.0)

    def forward(self, t, c):
        # np.array([a * y[0] - b * y[0] * y[1], -c * y[1] + d * b * y[0] * y[1]])
        # w = self.w.type_as(y).to(y.device).clone().detach()
        # print(c, t, ks, k_kinetics)
        # print()
        self.eval_times += 1

        ks, k_kinetics = self.ks, self.k_kinetics

        c_xNH3, c_xNO3, c_xNO2, c_xNOrg, c_xN2, c_ANH3, c_ANO3, c_ANO2, c_ANOrg, c_AN2 = c

        r1 = ks[0] * c_xN2 if k_kinetics[0] == 1 else ks[0]
        r2 = ks[1] * c_xNH3 if k_kinetics[1] == 1 else ks[1]
        r3 = ks[2] * c_xNO2 if k_kinetics[2] == 1 else ks[2]
        r4 = ks[3] * c_xNO3 if k_kinetics[3] == 1 else ks[3]
        r5 = ks[4] * c_xNO2 if k_kinetics[4] == 1 else ks[4]
        r6 = ks[5] * c_xNO2 * c_xNO3 if k_kinetics[5] == 1 else ks[5]
        r7 = ks[6] * c_xNO3 if k_kinetics[6] == 1 else ks[6]
        r8 = ks[7] * c_xNO3 if k_kinetics[7] == 1 else ks[7]
        r9 = ks[8] * c_xNH3 if k_kinetics[8] == 1 else ks[8]
        r10 = ks[9] * c_xNOrg if k_kinetics[9] == 1 else ks[9]
        r11 = ks[10] * c_xNOrg if k_kinetics[10] == 1 else ks[10]

        dc_xNH3 = 2 * r1 + r7 + r10 - r2 - r6 - r9
        dc_xNO3 = r3 - r7 - r4 - r8 + r11
        dc_xNO2 = r2 + r4 - r3 - r6 - 2 * r5
        dc_xNOrg = r8 + r9 - r10 - r11
        dc_xN2 = r5 + r6 - r1
        dc_ANH3 = (2 * r1 * (c_AN2 - c_ANH3) + (c_ANO3 - c_ANH3) * r7 + (c_ANOrg - c_ANH3) * r10) / c_xNH3
        dc_ANO3 = ((c_ANO2 - c_ANO3) * r2 + (c_ANOrg - c_ANO3) * r11) / c_xNO3
        dc_ANO2 = ((c_ANH3 - c_ANO2) * r2 + (c_ANO3 - c_ANO2) * r4) / c_xNO2
        dc_ANOrg = ((c_ANO3 - c_ANOrg) * r8 + (c_ANH3 - c_ANOrg) * r9) / c_xNOrg
        dc_AN2 = ((c_ANO2 - c_AN2) * r5 + (c_ANO2 * c_ANH3 - c_AN2) * r6) / c_xN2

        dcdts = [dc_xNH3, dc_xNO3, dc_xNO2, dc_xNOrg, dc_xN2, dc_ANH3, dc_ANO3, dc_ANO2, dc_ANOrg, dc_AN2]
        _y = torch.stack(dcdts)
        return _y


class N15TracingModel(torch.nn.Module):

    def __init__(self, ccts):
        super(N15TracingModel, self).__init__()

        self.ks_net = nn.Sequential(
            nn.Linear(40,256),
            nn.Linear(256,11),
            nn.Softmax()
        )

        self.kk_net = nn.Sequential(
            nn.Linear(40, 256),
            nn.Linear(256, 11),
            nn.Softmax()
        )
        self.ccts = torch.as_tensor(ccts)


        self.post_opti()
        self.t_eval = t_eval

        self.eval_times = 0

    def pre_opti(self):
        self.ks = self.ks_net(self.ccts)
        self.kk = self.kk_net(self.ccts)


    def post_opti(self, reset=False):

        def _constrain_data(p,cons):
            if torch.isnan(p.data):
                p.data = torch.rand(1)[0]
            if p.data < 0:
                p.data = torch.tensor(cons[0])
            if p.data > cons[1] or p.data < cons[0]:
                p.data.uniform_(cons[0],  cons[1])
            return p

        with torch.no_grad():
            cons = getattr(self.ks, 'constrain')
            for i, k in enumerate(self.ks):
                self.ks[i] = _constrain_data(k, cons)

            cons = getattr(self.kk, 'constrain')
            for i, k in enumerate(self.kk):
                self.kk[i] = _constrain_data(k, cons)

        self.eval_times = 0


    def forward(self, t, c):
        self.eval_times += 1

        ks, kk = self.ks, self.kk

        c_xNH3, c_xNO3, c_xNO2, c_xNOrg, c_xN2, c_ANH3, c_ANO3, c_ANO2, c_ANOrg, c_AN2 = c


        r1 = kk[0] * (ks[0] * c_xN2) + (1 - kk[0])* ks[0]
        r2 = kk[1] * (ks[1] * c_xNH3) + (1 - kk[1]) * ks[1]
        r3 = kk[2] * (ks[2] * c_xNO2) + (1 - kk[2])* ks[2]
        r4 = kk[3] * (ks[3] * c_xNO3) + (1 - kk[3])* ks[3]
        r5 = kk[4] * (ks[4] * c_xNO2) + (1 - kk[4])* ks[4]
        r6 = kk[5] * (ks[5] * c_xNO2 * c_xNO3) + (1 - kk[5])* ks[5]
        r7 = kk[6] * (ks[6] * c_xNO3) + (1 - kk[6])* ks[6]
        r8 = kk[7] * (ks[7] * c_xNO3) + (1 - kk[7])* ks[7]
        r9 = kk[8] * (ks[8] * c_xNH3) + (1 - kk[8])* ks[8]
        r10 = kk[9] * (ks[9] * c_xNOrg) + (1 - kk[9])* ks[9]
        r11 = kk[10] * (ks[10] * c_xNOrg) + (1 - kk[10])* ks[10]


        dc_xNH3 = 2 * r1 + r7 + r10 - r2 - r6 - r9
        dc_xNO3 = r3 - r7 - r4 - r8 + r11
        dc_xNO2 = r2 + r4 - r3 - r6 - 2 * r5
        dc_xNOrg = r8 + r9 - r10 - r11
        dc_xN2 = r5 + r6 - r1
        dc_ANH3 = (2 * r1 * (c_AN2 - c_ANH3) + (c_ANO3 - c_ANH3) * r7 + (c_ANOrg - c_ANH3) * r10) / c_xNH3
        dc_ANO3 = ((c_ANO2 - c_ANO3) * r2 + (c_ANOrg - c_ANO3) * r11) / c_xNO3
        dc_ANO2 = ((c_ANH3 - c_ANO2) * r2 + (c_ANO3 - c_ANO2) * r4) / c_xNO2
        dc_ANOrg = ((c_ANO3 - c_ANOrg) * r8 + (c_ANH3 - c_ANOrg) * r9) / c_xNOrg
        dc_AN2 = ((c_ANO2 - c_AN2) * r5 + (c_ANO2 * c_ANH3 - c_AN2) * r6) / c_xN2

        dcdts = [dc_xNH3, dc_xNO3, dc_xNO2, dc_xNOrg, dc_xN2, dc_ANH3, dc_ANO3, dc_ANO2, dc_ANOrg, dc_AN2]
        _y = torch.stack(dcdts)
        return _y
class N15Loss(nn.Module):

    def __init__(self, method="r2l1", delta=0.1):
        super(N15Loss, self).__init__()
        self.method = method
        self.delta = delta

    def forward(self, pred, target):

        _l = torch.tensor(0.)

        if self.method == 'r2':
            target_mean = torch.mean(target, dim=0)
            ss_true = torch.sum(torch.abs(target - target_mean), dim=0)
            ss_pred = torch.sum(torch.abs(pred - target_mean), dim=0)

            p = ss_true / ss_pred  # 0-1, 1 is best
            r2_loss = torch.abs(1 - p)  # so loss of 1 is 0
            _l = torch.mean(r2_loss)

        if self.method == 'r2l1':
            l1_loss = torch.mean(torch.abs(pred - target), dim=0)
            _l = (_l + torch.mean(l1_loss))/2

        if self.method == 'huber':
            delta = self.delta
            huber_loss = torch.where(torch.abs(target - pred) < delta, 0.5 * ((target - pred) ** 2),
                     delta * torch.abs(target - pred) - 0.5 * (delta ** 2))
            _l = huber_loss.mean()

        if self.method == 'phuber':
            delta = self.delta
            percent_huber_loss = torch.where(torch.abs(target - pred) / target < delta, 0.5 * ((target - pred) ** 2),
                                             delta * torch.abs(target - pred) - 0.5 * (delta ** 2))
            _l = percent_huber_loss.mean()

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


lr = 1e-3
vis_it = 2
check_interval = 10


atol, rtol = 1e-10, 1e-8
restore_model = False


options = None

# method = "explicit_adams"
# options = dict(step_size=0.01)
# scipy_
from scipy.integrate import solve_ivp


method = "dopri5"
loss_method = "r2l1"
# options = dict(step_size=0.1)

# nn.MSELoss
# List of ODE Solvers:
# Adaptive-step:
#
# dopri8 Runge-Kutta of order 8 of Dormand-Prince-Shampine.
# dopri5 Runge-Kutta of order 5 of Dormand-Prince-Shampine [default].
# bosh3 Runge-Kutta of order 3 of Bogacki-Shampine.
# fehlberg2 Runge-Kutta-Fehlberg of order 2.
# adaptive_heun Runge-Kutta of order 2.
#
# Fixed-step:
# euler Euler method.
# midpoint Midpoint method.
# rk4 Fourth-order Runge-Kutta with 3/8 rule.
# explicit_adams Explicit Adams-Bashforth.
# implicit_adams Implicit Adams-Bashforth-Moulton.
# Additionally, all solvers available through SciPy are wrapped for use with scipy_solver.


model_save_path = f"test/model-best-{method}.pth"

ode_func = N15TracingModel(k_kinetics)
if os.path.exists(model_save_path) and restore_model:
    print(f"loadding from {model_save_path}")
    ode_func.load_state_dict(torch.load(model_save_path))
    ode_func.eval()

device = "cpu"
params = ode_func.parameters()
batch_y0, batch_t, batch_y = torch.tensor(c0, dtype=torch.float64).to(device), \
    torch.tensor(t_eval, dtype=torch.float64).to(device), \
    torch.tensor(ccts, dtype=torch.float64).to(device)

optimizer = optim.RAdam(params, lr=lr)
loss_func = N15Loss(method=loss_method)
# loss_func = nn.HuberLoss(delta=0.1)


epoch = 100000
pbar = tqdm(total=epoch, ascii=True, ncols=300)


with torch.no_grad():
    pred_y = tf_odeint(ode_func, batch_y0, batch_t, method=method, atol=atol, rtol=rtol, options=options).to(device)
    # min_loss = loss_func(pred_y, batch_y)
    min_loss = torch.inf

    dataset_pred = deepcopy(dataset)
    dataset_pred.set_as_sim_dataset(t_eval=t_eval, y0=None, args=None, nowy=pred_y.detach().numpy())
    # core.plot_dataset(dataset, dataset_pred)


with DynamicShowPlot() as fig:
    for itr in range(epoch):
        optimizer.zero_grad()
        # e_time, _ = odeint_event(ode_func,
        #                          batch_y0,
        #                          batch_t[0],
        #                          event_fn=ode_func.event_fn,
        #                          reverse_time=False,
        #                          atol=atol,
        #                          rtol=rtol,
        #                          odeint_interface=tf_odeint_adj,
        #                          method=method,
        #                          options=options
        #                          )
        # if e_time < t_eval[-1]:
        #     pass
        pred_y = tf_odeint_adj(ode_func, batch_y0, batch_t, method=method, atol=atol, rtol=rtol, options=options).to(device)
        loss = loss_func(pred_y, batch_y)
        # loss2 = loss_func2(pred_y, batch_y)
        # loss = torch.mean(torch.abs(pred_y - batch_y))
        loss.backward()
        optimizer.step()

        if not hasattr(ode_func, 'post_opti'):
            continue
        eval_times = ode_func.eval_times
        ode_func.post_opti()

        with torch.no_grad():
            _now = ", ".join([f"k{i + 1}:{k.detach().numpy():.8f}" for i, k in enumerate(ode_func.ks)])
            pbar.set_description(f'Iter {itr:04d} | Total Loss {loss.item():.4f} {eval_times} min_loss:{min_loss:.4f} {_now}')
            pbar.update(1)
            pbar.refresh()

            if itr % check_interval == 0:
                if loss < min_loss:
                    min_loss = loss
                    torch.save(ode_func.state_dict(), model_save_path)

            if itr % vis_it == 0:
                dataset_pred = deepcopy(dataset)
                dataset_pred.set_as_sim_dataset(t_eval=t_eval, y0=None, args=None, nowy=pred_y.detach().numpy())
                core.plot_dataset(dataset, dataset_pred, fig=fig)

# _now = ", ".join([f"{n}:{p.detach().numpy()}" for n, p in ode_func.named_parameters()])
print(_now)
