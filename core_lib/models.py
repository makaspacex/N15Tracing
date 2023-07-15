#! /usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import pymc as pm
from .dff_odes import competition_model
from .dff_odes import get_dcdt_func_for_sunode
import sunode
import torch
import torch.nn as nn

# 这个函数是一个错误的解决方案
def get_target(ks, df, k_kinetics):
    print("WARNING: 错误的解决方案函数，不应该使用整体速率作为参考结果")
    
    target = [] # 'xNH3', 'xNO3', 'xNO2', 'xNOrg', 'xN2', 'ANH3', 'ANO3', 'ANO2', 'ANOrg', 'AN2'
    for i in range(0,len(df)):
        sr = df.iloc[i]
        
        r1 = ks[0] * sr['xN2'] if k_kinetics[0] == 1 else ks[0]
        r2 = ks[1] * sr['xNH3'] if k_kinetics[1] == 1 else ks[1]
        r3 = ks[2] * sr['xNO2'] if k_kinetics[2] == 1 else ks[2]
        r4 = ks[3] * sr['xNO3'] if k_kinetics[3] == 1 else ks[3]
        r5 = ks[4] * sr['xNO2'] if k_kinetics[4] == 1 else ks[4]
        r6 = ks[5] * sr['xNO2'] * sr['xNO3'] if k_kinetics[5] == 1 else ks[5]
        r7 = ks[6] * sr['xNO3'] if k_kinetics[6] == 1 else ks[6]
        r8 = ks[7] * sr['xNO3'] if k_kinetics[7] == 1 else ks[7]
        r9 = ks[8] * sr['xNH3'] if k_kinetics[8] == 1 else ks[8]
        r10 = ks[9] * sr['xNOrg'] if k_kinetics[9] == 1 else ks[9]
        r11 = ks[10] * sr['xNOrg'] if k_kinetics[10] == 1 else ks[10]

        xNH3_rate =  2*r1 + r7 + r10 - r2 - r6 - r9
        xNO3_rate = r3 - r7 - r4 - r8 + r11
        xNO2_rate = r2 + r4 - r3 - r6 - 2*r5
        xNOrg_rate = r8 + r9 - r10 -r11
        xN2_rate = r5 + r6 - r1
        ANH3_rate = (2*r1*(sr['AN2'] - sr['ANH3']) + (sr['ANO3']-sr['ANH3'])*r7 + (sr['ANOrg']-sr['ANH3'])*r10 )/sr['xNH3']
        ANO3_rate = ( (sr['ANO2'] - sr['ANO3'])*r2 + (sr['ANOrg'] - sr['ANO3'])*r11 ) / sr['xNO3']
        ANO2_rate = ( (sr['ANH3']-sr['ANO2'] )*r2 + (sr['ANO3']-sr['ANO2'])*r4 ) / sr['xNO2']
        ANOrg_rate = ( (sr['ANO3']-sr['ANOrg'] )*r8 + (sr['ANH3']-sr['ANOrg'])*r9 ) / sr['xNOrg']
        AN2_rate = ( (sr['ANO2']-sr['AN2'] )*r5 + (sr['ANO2']*sr['ANH3'] - sr['AN2'])*r6 ) / sr['xN2']

        line_rate = [xNH3_rate, xNO3_rate,xNO2_rate, xNOrg_rate, xN2_rate, ANH3_rate,ANO3_rate, ANO2_rate,ANOrg_rate,AN2_rate]
        target.append(line_rate)
    
    target = np.array(target)
    
    return target

def get_model(dataset, t_eval, k_kinetics, k_sigma_priors=0.01, kf_type=0, c0_type=1, distance="gaussian", epsilon=1):
    df = dataset.get_df()
    cct_names, _, _ = dataset.get_var_col_names()
    ccts = dataset.get_cct()
    
    mcmc_model = pm.Model()
    params_n = 11

    parames =[]
    c0 = []
    
    with mcmc_model:
        for ki in range(1, params_n + 1):
            if kf_type == 0:
                p_dense = pm.HalfNormal(f"k{ki}", sigma=k_sigma_priors)
                # _sigma = ks[ki-1] * np.pi **0.5 / 2 ** 0.5
                # p_dense = pm.HalfNormal(f"k{ki}", sigma=_sigma)
            else:
                p_dense = pm.Normal(f"k{ki}",mu=0, sigma=k_sigma_priors)
            parames.append(p_dense)
        
        if c0_type == 1:
            for c_name in cct_names:
                _maxx = df[c_name].values.max()
                c0.append(pm.HalfNormal(f"{c_name}_s", sigma=_maxx))
        else:
            c0 = df[cct_names].values[0]
        print(c0)
        sim = pm.Simulator("sim", competition_model, params=(t_eval, c0, parames, k_kinetics),distance=distance, epsilon=epsilon, observed=ccts)
    return mcmc_model


def get_model2(dataset, t_eval, k_kinetics, k_sigma_priors=0.01, kf_type=0):
    import sunode
    import sunode.wrappers
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
            rhs=get_dcdt_func_for_sunode(k_kinetics),
            tvals=times,
            t0=times[0],
        )
        
        sd = pm.HalfNormal('sd')
        for c_name in cct_names:
            pm.Normal(f'{c_name}', mu=y_hat[f"{c_name}"], sigma=sd, observed=df[f"{c_name}"].values)
            pm.Deterministic(f'{c_name}_mu', y_hat[f"{c_name}"])
    return mcmc_model

def distance_func(epsilon, obs_data, sim_data):
    # dis = -0.5 * ((obs_data - sim_data) / epsilon / 10) ** 2
    dis = -0.5 * ((obs_data - sim_data) * epsilon) ** 2
    return dis
MY_EPSILON =   [1,   1,  100,   0.1, 10,   10, 10, 10, 1000, 10]


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

    def __init__(self, t_eval):
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
        setattr(self.ks, 'constrain', [1e-6, 1e-3])

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

    def __init__(self, ccts, t_eval, device="cpu"):
        super(N15TracingModel, self).__init__()

        self.ks_net = nn.Sequential(
            nn.Linear(40,256),
            nn.Linear(256,11),
            nn.Softmax()
        ).to(device)

        self.kk_net = nn.Sequential(
            nn.Linear(40, 256),
            nn.Linear(256, 11),
            nn.Softmax()
        ).to(device)
        
        self.ccts = torch.as_tensor(ccts, dtype=torch.float32).to(device)
        
        self.t_eval = t_eval
        self.eval_times = 0
        self.device = device

    def pre_opti(self):
        self.ks = self.ks_net(self.ccts.reshape(-1))
        self.kk = self.kk_net(self.ccts.reshape(-1))


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
        self.pre_opti()
        
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

    def __init__(self, method="r2l1", delta=0.1, device="cpu"):
        super(N15Loss, self).__init__()
        self.method = method
        self.delta = delta
        self.device = device

    def forward(self, pred, target):

        _l = torch.tensor(0.).to(self.device)

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

