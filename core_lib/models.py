#! /usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import pymc as pm
from .dff_odes import competition_model
from .dff_odes import get_dcdt_func_for_sunode
import sunode

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
        # for c_name in cct_names:
        #     _maxx = df[c_name].values.max()
        #     _c0 = df[c_name].values[0]
        #     _sigma_c0 = _c0 * np.pi **0.5 / 2 ** 0.5

        #     # half_c0 = pm.HalfNormal(f"{c_name}_s", sigma=_maxx)
        #     half_c0 = pm.HalfNormal(f"{c_name}_s", sigma=_sigma_c0)
        #     # dira_c0 = pm.DiracDelta(f"{c_name}_s",c=_c0)
        #     c0.append(half_c0)
        print(c0)
        sim = pm.Simulator("sim", competition_model, params=(t_eval, c0, parames, k_kinetics),distance=distance, epsilon=epsilon, observed=ccts)
    return mcmc_model


def get_model2(dataset, t_eval, k_kinetics, k_sigma_priors=0.01, kf_type=0):

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

# epsilon =   [1,   1,  100,   0.1, 10,   10, 10, 10, 1000, 10]

