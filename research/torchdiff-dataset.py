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
from torchdiffeq import odeint_event
from torchvision import models
# import torchsummary
from core_lib import DynamicShowPlot, MyDataset, plot_dataset
from core_lib import N15TracingModel
from core_lib import N15Loss


db_csv_path = "dataset/data.csv"
idata_save_path = "odes-exp04-idata-4-number-1core-c0number-halfnormks-from-core.py-success.dt"

dataset_ori = MyDataset(db_csv_path)
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
plot_dataset(dataset, dataset)


#
#
# resnet18 = models.resnet18(pretrained=True)
# num_ftrs = resnet18.fc.in_features
# resnet18.fc = nn.Linear(num_ftrs, 11)
# reset_input = torch.as_tensor(ccts, dtype=torch.float32).unsqueeze(0).repeat(3,1,1).unsqueeze(0)


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

ode_func = N15TracingModel(k_kinetics, t_eval=t_eval)
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
                plot_dataset(dataset, dataset_pred, fig=fig)

# _now = ", ".join([f"{n}:{p.detach().numpy()}" for n, p in ode_func.named_parameters()])
print(_now)
