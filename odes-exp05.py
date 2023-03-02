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
from IPython.display import display as print
from datetime import datetime
from scipy.integrate import odeint

# import sunode
import pickle

import core



db_csv_path = "dataset/data.csv"
idata_save_path = "odes-exp04-idata-4-number-single-core-from-core.py-success.dt"

dataset_ori = core.MyDataset(db_csv_path)
df_ori = dataset_ori.get_df()
cct_names, rates_names, error_names = dataset_ori.get_var_col_names()
c0 = df_ori[cct_names].iloc[0].values


# 假设都是一级动力学
k_kinetics = np.repeat(1, 11).astype(np.uint8) 
# k_kinetics = np.array([0,0,0,0,1,1,0,0,1,1,0]).astype(np.uint8) 
ks = np.array([0.00071942, 0.00269696, 0.00498945, 0.00444931, 0.00571299, 0.00801272, 0.00131931, 0.00319959, 0.00415571, 0.00228432, 0.00177611])
#  =======================================================

# t_eval = np.linspace(0.5, 150, 8)
t_eval = np.array([0.5, 48, 96, 144])


dataset = core.MyDataset(db_csv_path)
df = dataset.get_df()
cct_names, rates_names, error_names = dataset.get_var_col_names()
c0 = df[cct_names].iloc[0].values
dataset.set_as_sim_dataset(core.dcdt_func_for_odeint, t_eval, c0, args=(ks, k_kinetics))
df = dataset.get_df()

# core.plot_dataset(dataset, dataset)


mcmc_model = core.get_model(dataset, t_eval, k_kinetics, distance=core.distance_func, epsilon=core.epsilon, k_sigma_priors=0.01, kf_type=0, c0_type=1)

print(idata_save_path)

idata_lv = pm.sample_smc(draws=2000, chains=1, model=mcmc_model, progressbar=True)
pickle.dump(idata_lv,open(idata_save_path, 'wb'))
