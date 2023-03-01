#!/usr/bin/env python
# coding: utf-8


import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import pymc as pm
from pymc.ode import DifferentialEquation

from itertools import product
import os.path as osp
from scipy.optimize import leastsq
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
import math
import time

az.style.use("arviz-darkgrid")

from scipy.integrate import odeint

from datetime import datetime


class MyDataset(object):

    def __init__(self, dataset_path):

        self.dataset_path = dataset_path

        df = pd.read_csv(dataset_path)

        self.df = df
        self._setup()

    def _setup(self):

        df = self.df
        # 数据初步处理
        # 计算反应速率rate，初始速率固定设置为0
        for c_i, (col_name, col_sr) in enumerate(df.items()):
            if "error" in col_name or "time" in col_name or "rate" in col_name:
                continue
            rate_col_name = f"{col_name}_rate"
            rates = []
            pre_t = None
            pre_v = None
            for th, (index, value) in zip(df['time'], col_sr.items()):
                if int(index) == 0:
                    rates.append(0.0)
                    pre_t = th
                    pre_v = value
                    continue

                delta_t = th - pre_t
                delta_value = value - pre_v
                # print(col_name, index, pre_t, th, pre_v ,value)
                rates.append(delta_value / delta_t)
                pre_t = th
                pre_v = value
            df[rate_col_name] = rates

        # 准备输出值 Y
        self.cct_names = []
        for x in self.df.columns:
            if "time" in x or "error" in x or "rate" in x:
                continue
            self.cct_names.append(x)
        self.rates_names = [f"{x}_rate" for x in self.cct_names]
        self.error_names = [f"{x}-error" for x in self.cct_names]

        self.cct = self.df[self.cct_names].values
        self.rates = self.df[self.rates_names].values
        self.errors = self.df[self.error_names].values

    def set_as_sim_dataset(self, dcdt_fuc, t_eval, y0, args):

        y = odeint(dcdt_fuc, y0=y0, t=t_eval, args=args)

        # y.shape (size, 10)
        df_new = pd.DataFrame(columns=['time'] + self.cct_names)
        df_new['time'] = t_eval

        for c_name, col_val in zip(self.cct_names, np.transpose(y, [1, 0])):
            df_new[c_name] = col_val
            df_new[f"{c_name}-error"] = 0.001

        self.df = df_new
        self._setup()

    def get_rates(self):
        return self.rates

    def get_df(self):
        return self.df

    def get_errors(self):
        return self.errors

    def get_cct(self):
        return self.cct

    def get_var_col_names(self):
        return self.cct_names, self.rates_names, self.error_names

    def get_weights(self):
        max_value = self.df[self.cct_names].describe().loc['max'].values.max()

        vars_max = self.df[self.cct_names].describe().loc['max']
        weights = (max_value / vars_max).values

        return np.array(weights)

    def get_vars_max(self):
        vars_max = self.df[self.cct_names].describe().loc['max'].values
        return vars_max


def get_format_time(f_s=None):
    haomiao = str(time.time()).split('.')[-1]
    if f_s is None:
        f_s = "%Y%m%d%H%M%S"
        return datetime.now().strftime(f_s) + haomiao
    return datetime.now().strftime(f_s)


def dcdt_func_for_odeint(c, t, ks, k_kinetics):
    # print(c, t, ks, k_kinetics)
    # print()
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

    return np.array(dcdts)


# simulator function
def competition_model(rng, t_eval, y0, ks, k_kinetics):
    y = odeint(dcdt_func_for_odeint, y0=y0, t=t_eval, args=(ks, k_kinetics))
    return y

def plot_dataset(dataset):
    df = dataset.get_df()

    cols = 5
    rows = math.ceil(len(cct_names) / cols)

    fig, fig_axes = plt.subplots(ncols=cols, nrows=rows, figsize=(4.2 * cols, 4 * rows), dpi=100)
    if isinstance(fig_axes, np.ndarray):
        fig_axes = fig_axes.reshape(-1)
    else:
        fig_axes = [fig_axes]

    for i, axes in enumerate(fig_axes):
        if i >= len(cct_names):
            axes.axis('off')
            continue

        y_name = cct_names[i]
        Y = df[y_name].values
        axes.plot(df['time'].values, Y, label=f"ob")
        axes.set_ylabel(f'cct_{y_name}')
        axes.set_xlabel(f'time(h)')

        # axes.plot(df['time'].values, df[rates_names[i]].values, '+', label=f"rate")

        # axes.plot(t_eval, cs[i, :], 'r', label=f"c(t)")
        # axes.plot(t_eval, dcdt_df[y_name].values,'g', label=f"c'(t)")

        axes.legend()
        # axes.set_title(f"{y_name}", fontsize=14)

    plt.tight_layout()
    plt.show()


def r2_loss(pred, y):
    r2_loss = 1 - np.square(pred - y).sum() / np.square(y - np.mean(y)).sum()
    return r2_loss


def get_model(dataset, t_eval, k_kinetics, k_sigma_priors=0.01, kf_type=0):
    df = dataset.get_df()
    times = df['time'].values
    cct_names, rates_names, error_names = dataset.get_var_col_names()
    ccts = dataset.get_cct()

    mcmc_model = pm.Model()
    params_n = 11

    parames = []
    c0 = []

    c0 = dataset.get_cct()[0]
    with mcmc_model:
        for ki in range(1, params_n + 1):
            if kf_type == 0:
                p_dense = pm.HalfNormal(f"k{ki}", sigma=k_sigma_priors)
            else:
                p_dense = pm.Normal(f"k{ki}", mu=0, sigma=k_sigma_priors)
            parames.append(p_dense)

        # for c_name in cct_names:
        #     _maxx = df[c_name].values.max()
        #     c0.append(pm.HalfNormal(f"{c_name}_s", sigma=_maxx))


    with mcmc_model:
        sim = pm.Simulator("sim", competition_model, params=(t_eval, c0, parames, k_kinetics), epsilon=1, observed=ccts)
        # sim = pm.Simulator("sim", competition_model, params=(t_eval,), epsilon=1, observed=ccts)

    return mcmc_model
# aesara/tensor/basic.py

from aesara.tensor import basic


k_kinetics = np.repeat(1, 11).astype(np.uint8)

t_eval = np.arange(0, 150, 0.5)
ks = np.random.random(11) / 100  # 先验k

dataset = MyDataset("dataset/data.csv")
df = dataset.get_df()
cct_names, rates_names, error_names = dataset.get_var_col_names()

c0 = df[cct_names].iloc[0].values
dataset.set_as_sim_dataset(dcdt_func_for_odeint, t_eval, c0, args=(ks, k_kinetics))
df = dataset.get_df()

mcmc_model = get_model(dataset, t_eval, k_kinetics, k_sigma_priors=0.1, kf_type=0)

with mcmc_model:
    idata_lv = pm.sample_smc(cores=4)


res_df  = az.summary(idata_lv)

res_df.to_csv("sssssss.csv")


# In[ ]:

# In[ ]:


az.plot_trace(idata_lv, kind="rank_vlines");

# In[ ]:


az.plot_posterior(idata_lv);

# In[ ]:


# plot results
_, ax = plt.subplots(figsize=(14, 6))
posterior = idata_lv.posterior.stack(samples=("draw", "chain"))
ax.plot(observed[:, 0], "o", label="prey", c="C0", mec="k")
ax.plot(observed[:, 1], "o", label="predator", c="C1", mec="k")

y_posterior = (posterior["y0_1"].mean(), posterior["y0_2"].mean())
ax.plot(competition_model(None, y_posterior, posterior["a"].mean(), posterior["b"].mean()), linewidth=3)
for i in np.random.randint(0, size, 10):
    sim = competition_model(None, (posterior["y0_1"][i], posterior["y0_2"][i]), posterior["a"][i], posterior["b"][i])
    ax.plot(sim[:, 0], alpha=0.1, c="C0")
    ax.plot(sim[:, 1], alpha=0.1, c="C1")
ax.set_xlabel("time")
ax.set_ylabel("population")
ax.legend();

# In[ ]:


pm.sample_smc()
pm.smc.sample_smc()
