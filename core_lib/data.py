#! /usr/bin/python
# -*- coding: utf-8 -*-
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import os.path as osp
import math
import time
from datetime import datetime
from scipy.integrate import odeint, solve_ivp

import sys
from contextlib import redirect_stdout
from .dff_odes import xj_diff_solve_ivp


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


    def set_as_sim_dataset(self, t_eval, y0, args, t0=None, nowy=None):
        # 默认情况下 t0= t_eval[0]
        if t_eval[-1] < t_eval[0]:
            raise Exception("不支持反向模式")
        if nowy is None:
            if t0 is None:
                # y = odeint(dcdt_fuc, y0=y0, t=t_eval, args=args)
                # y_s = solve_ivp(get_dcdts(c_first=False), t_span=(np.min(t_eval), np.max(t_eval)), y0=y0, t_eval=t_eval, args=args)
                # y = y_s.y.transpose(1,0)
                y = xj_diff_solve_ivp(y0, t_eval, args)

            else:
                _i = -1
                insert = False
                for i, x in enumerate(t_eval):
                    if x == t0:
                        _i = i
                        break
                if _i == -1:
                    # 没有找到就插入
                    insert = True
                    t_eval = list(t_eval) + [t0]
                    t_eval = sorted(t_eval)
                    _i = -1
                    for i, x in enumerate(t_eval):
                        if x == t0:
                            _i = i
                            break
                    if _i == -1:
                        raise Exception("没有找到t0的位置")

                # [7,6,5,4,3,2,1,0]
                # [0,1,2,3,4,5,6,7,8]
                # 假设t0 = 2, _i = 2

                # 反向的
                t_eval1 = np.array(t_eval[:_i+1])[::-1]

                # y1_s = solve_ivp(get_dcdts(c_first=False), t_span=(np.min(t_eval1), np.max(t_eval1)), y0=y0, t_eval=t_eval1, args=args)
                # if len(t_eval1) != len(y1_s.t):
                #     y1 = odeint(get_dcdts(c_first=True), y0=y0, t=t_eval1, args=args)
                # else:
                #     y1 = y1_s.y.transpose(1,0)
                y1 = xj_diff_solve_ivp(y0, t_eval1, args)


                # 正向的
                t_eval2 = np.array(t_eval[_i:])
                # y2_s = solve_ivp(get_dcdts(c_first=False), t_span=(np.min(t_eval2), np.max(t_eval2)), y0=y0, t_eval=t_eval2, args=args)
                # if len(t_eval2) != len(y2_s.t):
                #     y2 = odeint(get_dcdts(c_first=True), y0=y0, t=t_eval2, args=args)
                # else:
                #     y2 = y2_s.y.transpose(1,0)
                y2 = xj_diff_solve_ivp(y0, t_eval2, args)

                if not insert:
                    t = np.concatenate([t_eval1[::-1][:-1],t_eval2])
                    y = np.concatenate([y1[::-1][:-1],y2])
                else:
                    t = np.concatenate([t_eval1[::-1][:-1],t_eval2[1:]])
                    y = np.concatenate([y1[::-1][:-1],y2[1:]])

                t_eval = t
        else:
            y = nowy

        # y.shape (size, 10)
        df_new = pd.DataFrame(columns=['time'] + self.cct_names)
        df_new['time'] = t_eval

        for c_name, col_val in zip(self.cct_names, np.transpose(y, [1,0])):
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


def plot_dataset(dataset, dataset_pred=None, fig=None):

    df = dataset.get_df()
    cct_names, rates_names, error_names = dataset.get_var_col_names()

    cols = 5
    rows = math.ceil(len(cct_names) / cols)

    if fig:
        # fig.clear()
        fig = plt.figure(fig.get_label())
        fig.set_dpi(100)
        fig.set_size_inches(4.2 * cols, 4 * rows)
        if len(fig.axes) != cols * rows:
            fig.clear()
            fig_axes = fig.subplots(ncols=cols, nrows=rows)
        else:
            fig_axes = np.array(fig.axes)
    else:
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

        axes.clear()
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
    plt.draw()
    plt.pause(0.1)
