#! /usr/bin/python
# -*- coding: utf-8 -*-
# @author izhangxm
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

import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
from itertools import product
import os.path as osp
from scipy.optimize import leastsq
import time
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
import math
import sunode
import sunode.wrappers.as_pytensor
from copy import deepcopy
from multiprocessing import Process, Queue, cpu_count
from tqdm import  tqdm
from datetime import datetime
import pickle
az.style.use("arviz-darkgrid")
import os
from pathlib import Path

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
        
        for c_name, col_val in zip(self.cct_names, np.transpose(y, [1,0])):
            df_new[c_name] = col_val
            df_new[f"{c_name}-error"] = 0.001
        
        self.df = df_new
        self._setup()
    
    def set_dataset(self, c_pred_df):
        
        df = deepcopy(c_pred_df)
        
        for c_name in self.cct_names:
            c_error_name = f"{c_name}-error"
            n_len = len(c_pred_df[c_name].values)
            _error = np.repeat(0.0001, n_len)
            c_pred_df[c_error_name] = _error
        self.df = df
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
        f_s = "%Y-%-m-%d %H:%M:%S."
        return datetime.now().strftime(f_s) + haomiao
    return datetime.now().strftime(f_s)


def plot_dataset(dataset, dataset_pred=None):
    
    df = dataset.get_df()
    cct_names, rates_names, error_names = dataset.get_var_col_names()
    
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
        axes.plot(df['time'].values, Y, '*', label=f"ob")
        axes.set_ylabel(f'cct_{y_name}')
        axes.set_xlabel(f'time(h)')

        # axes.plot(df['time'].values, df[rates_names[i]].values, '+', label=f"rate")
        
        if dataset_pred:
            _df_pred = dataset_pred.get_df()
            t_eval = _df_pred['time'].values
            axes.plot(t_eval, _df_pred[y_name].values, 'r', label=f"c(t)")
        
        
        # axes.plot(t_eval, dcdt_df[y_name].values,'g', label=f"c'(t)")

        axes.legend()
        # axes.set_title(f"{y_name}", fontsize=14)

    plt.tight_layout()
    plt.show()


def get_dcdt_func(k_kinetics):
    k_kinetics = np.array(k_kinetics).astype(np.uint8)
    def _dcdt_func(t, c, p):
        r1 = p.k1 * c.xN2 if k_kinetics[0] == 1 else p.k1
        r2 = p.k2 * c.xNH3 if k_kinetics[1] == 1 else p.k2
        r3 = p.k3 * c.xNO2 if k_kinetics[2] == 1 else p.k3
        r4 = p.k4 * c.xNO3 if k_kinetics[3] == 1 else p.k4
        r5 = p.k5 * c.xNO2 if k_kinetics[4] == 1 else p.k5
        r6 = p.k6 * c.xNO2 * c.xNO3 if k_kinetics[5] == 1 else p.k6
        r7 = p.k7 * c.xNO3 if k_kinetics[6] == 1 else p.k7
        r8 = p.k8 * c.xNO3 if k_kinetics[7] == 1 else p.k8
        r9 = p.k9 * c.xNH3 if k_kinetics[8] == 1 else p.k9
        r10 = p.k10 * c.xNOrg if k_kinetics[9] == 1 else p.k10
        r11 = p.k11 * c.xNOrg if k_kinetics[10] == 1 else p.k11

        
        dc_xNH3 = 2 * r1 + r7 + r10 - r2 - r6 - r9
        dc_xNO3 = r3 - r7 - r4 - r8 + r11
        dc_xNO2 = r2 + r4 - r3 - r6 - 2 * r5
        dc_xNOrg = r8 + r9 - r10 - r11
        dc_xN2 = r5 + r6 - r1

        dc_ANH3 = (2 * r1 * (c.AN2 - c.ANH3) + (c.ANO3 - c.ANH3) * r7 + (c.ANOrg - c.ANH3) * r10) / c.xNH3
        dc_ANO3 = ((c.ANO2 - c.ANO3) * r2 + (c.ANOrg - c.ANO3) * r11) / c.xNO3
        dc_ANO2 = ((c.ANH3 - c.ANO2) * r2 + (c.ANO3 - c.ANO2) * r4) / c.xNO2
        dc_ANOrg = ((c.ANO3 - c.ANOrg) * r8 + (c.ANH3 - c.ANOrg) * r9) / c.xNOrg
        dc_AN2 = ((c.ANO2 - c.AN2) * r5 + (c.ANO2 * c.ANH3 - c.AN2) * r6) / c.xN2

        # dcdts = [dc_xNH3, dc_xNO3, dc_xNO2, dc_xNOrg, dc_xN2, dc_ANH3, dc_ANO3, dc_ANO2, dc_ANOrg, dc_AN2]
        
        dcdts =  {
            'xNH3': dc_xNH3,
            'xNO3': dc_xNO3,
            'xNO2': dc_xNO2,
            'xNOrg': dc_xNOrg,
            'xN2': dc_xN2,
            'ANH3': dc_ANH3,
            'ANO3': dc_ANO3,
            'ANO2': dc_ANO2,
            'ANOrg': dc_ANOrg,
            'AN2': dc_AN2,
        }
        return dcdts
    
    return _dcdt_func


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
def competition_model(rng, t_eval, y0,  ks, k_kinetics, size=None):
    # print(y0)
    y = odeint(dcdt_func_for_odeint, y0=y0, t=t_eval, args=(ks, k_kinetics))
    return y



def r2_loss(pred, y):
    r2_loss = 1 - np.square(pred - y).sum() / np.square(y - np.mean(y)).sum()
    return r2_loss


def get_model(dataset, k_kinetics, k_sigma_priors=0.01, kf_type=0):

    df = dataset.get_df()
    times = df['time'].values
    
    errors = dataset.get_errors()
    rates = dataset.get_rates()
    cct_names, rates_names, error_names = dataset.get_var_col_names()
        
    # 定义参数优化模型
    mcmc_model = pm.Model()
    ## 参数个数
    params_n = 11
    parames ={}
    
    with mcmc_model:
        for ki in range(1, params_n + 1):
            if kf_type == 0:
                p_dense = pm.HalfNormal(f"k{ki}", sigma=k_sigma_priors)
            else:
                p_dense = pm.Normal(f"k{ki}",mu=0, sigma=k_sigma_priors)
            parames[f"k{ki}"] = (p_dense, ())
    
    parames['extra']=  np.zeros(1)
    
    c0 = {}
    with mcmc_model:
        for c_name in cct_names:
            _maxx = df[c_name].values.max()
            c0[f"{c_name}"] = (pm.HalfNormal(f"{c_name}_s", sigma=_maxx), ())
        

        y_hat, _, problem, solver, _, _ = sunode.wrappers.as_pytensor.solve_ivp(
            y0=c0,
            params=parames,
            rhs=get_dcdt_func(k_kinetics),
            tvals=times,
            t0=times[0],
        )
        
        sd = pm.HalfNormal('sd')
        for c_name in cct_names:
            pm.Normal(f'{c_name}', mu=y_hat[f"{c_name}"], sigma=sd, observed=df[f"{c_name}"].values)
            pm.Deterministic(f'{c_name}_mu', y_hat[f"{c_name}"])

    return mcmc_model


def get_predict_ks(idata):
    parames_summary = az.summary(idata, round_to=10)
    ks_names = [f"k{x+1}" for x in range(11)]

    predict_ks = []
    for k_name in ks_names:
        k_v = parames_summary["mean"][k_name]
        predict_ks.append(k_v)
    return np.array(predict_ks)


class NTraceModel(Process):
    def __init__(self, p_name, dataset, k_queue:Queue):
        super(NTraceModel, self).__init__()
        self.p_name = p_name
        self.dataset = dataset
        self.k_queue = k_queue

    def run(self):
        while not self.k_queue.empty():
            try:
                k_kinetics = np.array(self.k_queue.get(1, timeout=100))
                print(f"{self.p_name} {k_kinetics}")
                k_str = "".join([f"{x}" for x in k_kinetics])
                k_order = int(k_str,2)
                save_file_path = f"save_idata/{k_order}-idata.dt"
                if os.path.exists(save_file_path):
                    continue
                Path(save_file_path).touch()
                dataset = self.dataset
                mcmc_model = get_model(dataset, k_kinetics, k_sigma_priors=0.01, kf_type=0)
                idata = pm.sample(draws=1000, model=mcmc_model, chains=4, cores=4, progressbar=False)
                pickle.dump(idata,open(save_file_path, 'wb'))
            except Exception as e:
                pass

def mutil_run():
    kk_list_all = list(product([0,1], repeat=11))
    q = Queue(10000)
    for k_k in kk_list_all:
        q.put(k_k)
    dataset = MyDataset("dataset/data.csv")
    p_list = []
    
    
    n_cpu = int(cpu_count()/4)
    print(cpu_count(), n_cpu)

    
    for i in range(n_cpu):
        p = NTraceModel(f"Process-{i}", dataset, q)
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()
    print("all_finished")


if __name__ == '__main__':
    mutil_run()

