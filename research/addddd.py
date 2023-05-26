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
from datetime import datetime
from scipy.integrate import odeint

# import sunode
import pickle
import os
import sys
sys.path.append(os.getcwd())


import core



db_csv_path = "dataset/data.csv"
idata_save_path = "odes-exp03-idata-4-number-success.dt"

dataset = core.MyDataset(db_csv_path)
df = dataset.get_df()
cct_names, rates_names, error_names = dataset.get_var_col_names()
c0 = df[cct_names].iloc[0].values
kk_list_all = list(product([0,1], repeat=11))
# kk_list = np.vstack([np.repeat(0,11).astype(np.uint8), np.eye(11,11, dtype=np.uint8), np.repeat(1,11).astype(np.uint8)]).tolist()
t_eval = np.array([0.5, 48, 96, 144])
ks = np.array([0.00071942, 0.00269696, 0.00498945, 0.00444931, 0.00571299, 0.00801272, 0.00131931, 0.00319959, 0.00415571, 0.00228432, 0.00177611])
c0 = df[cct_names].iloc[0].values

all_dis = {}
min_dis = np.inf
kkkk = None

f_res = open("test/aa.txt", 'w+')

epsilon =   [1,   1,  100,   0.1, 10,   10, 10, 10, 1000, 10]
for k_kinetics in kk_list_all:
    try:
        k_kinetics = np.array(k_kinetics).astype(np.uint8) 
        dataset_new = core.MyDataset(db_csv_path)
        dataset_new.set_as_sim_dataset(core.dcdt_func_for_odeint, t_eval, c0, args=(ks, k_kinetics))
        df_new = dataset_new.get_df()
        # core.plot_dataset(dataset, dataset_new)
        aa  = (np.abs(df_new[cct_names].values - df[cct_names].values) * np.array(epsilon)).sum()
        if aa < min_dis:
            min_dis = aa
            kkk = k_kinetics
            f_res.write(f"{aa} {k_kinetics}\n")
            f_res.flush()
    except Exception as e:
        pass
f_res.flush()

