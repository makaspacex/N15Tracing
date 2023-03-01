#! /usr/bin/python
# -*- coding: utf-8 -*-
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
from itertools import product
import os.path as osp
from scipy.optimize import leastsq
import time

az.style.use("arviz-darkgrid")

class MyDataset(object):
    
    def __init__(self, dataset_path):
        
        self.dataset_path = dataset_path
        
        df = pd.read_csv(dataset_path)
        # 数据初步处理
        # 计算反应速率rate，初始速率固定设置为0

        for c_i, (col_name, col_sr) in enumerate(df.items()):
            if "error" in col_name or "time" in col_name or "rate" in col_name:
                continue
            rate_col_name = f"{col_name}_rate"
            rates = []
            pre_t = None
            pre_v = None
            for th, (index, value) in zip(df['time'],col_sr.items()):
                if int(index) == 0:
                    rates.append(0.0)
                    pre_t = th
                    pre_v = value
                    continue

                delta_t = th-pre_t
                delta_value = value - pre_v
                # print(col_name, index, pre_t, th, pre_v ,value)
                rates.append(delta_value/delta_t)
                pre_t = th
                pre_v = value
            df[rate_col_name] = rates
        
        self.df = df
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
    
def get_target(ks, df, k_kinetics):
    
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


def r2_loss(pred, y):
    r2_loss = 1 - np.square(pred - y).sum() / np.square(y - np.mean(y)).sum()
    return r2_loss

def get_model(dataset, k_kinetics, k_sigma_priors = 0.1, kf_type=0):
    # 定义参数优化模型
    mcmc_model = pm.Model()
    ## 参数个数
    params_n = 11

    ks = []
    with mcmc_model:
        for ki in range(1, params_n+1):
            if kf_type == 0:
                p_dense = pm.HalfNormal(f"k{ki}", sigma=k_sigma_priors)
            else:
                p_dense = pm.Normal(f"k{ki}",mu=0, sigma=k_sigma_priors)
            ks.append(p_dense)
    
    df = dataset.get_df()
    errors = dataset.get_errors()
    rates = dataset.get_rates()
    
    
    target= get_target(ks, df, k_kinetics)
    target = np.array(target)[1:].reshape(-1).tolist()
    sigma_Y = errors[1:].reshape(-1).tolist()
    rata_Y = rates[1:].reshape(-1).tolist()
    
    with mcmc_model:
        sigma = pm.HalfCauchy('sigma', beta=10, initval=0.1)
        y_obs = pm.Normal(f"rates", mu=target, sigma=sigma, observed=rata_Y, shape=len(rata_Y))
    
    return mcmc_model
def get_predict_ks(idata):
    parames_summary = az.summary(idata, round_to=10)
    ks_names = [f"k{x+1}" for x in range(11)]
    
    predict_ks = []
    for k_name in ks_names:
        k_v = parames_summary["mean"][k_name]
        predict_ks.append(k_v)
    return np.array(predict_ks)

def opt_model(dataset, k_kinetics, k_sigma_priors=0.01,  kf_type=0, draws=10000, tune=2000, chains=4, cores=4):
    mcmc_model = get_model(dataset, k_kinetics, k_sigma_priors=k_sigma_priors,kf_type=kf_type)
    idata = pm.sample(draws=draws,model=mcmc_model, chains=chains, cores=cores, tune=tune)
    return idata

def eval_model(idata, dataset):
    predict_ks = get_predict_ks(idata)
    predict = get_target(predict_ks, df, k_kinetics)
    rates_y  = dataset.get_rates()
    r2 = r2_loss(predict[1:],rates_y[1:])
    return r2

def ltq_fit(dataset,k_kinetics):
    def _error_loss(ks, dataset):
        df = dataset.get_df()
        rates_y = dataset.get_rates()
        predict= get_target(ks, df, k_kinetics)

        r2 = r2_loss(predict[1:],rates_y[1:])
        res =  (rates_y[1:] - predict[1:]).reshape(-1)

        is_nagative = False
        for x in ks:
            if x<=0:
                is_nagative = True
                break
        # if is_nagative:
        #     res = res + 1000
        # print('step', ks,r2, res.shape)
        return res

    ks_o = np.repeat(1,11).tolist()
    ks_res =leastsq(_error_loss, ks_o, args=(dataset,))[0]
    return ks_res


if __name__ == '__main__':
    pass