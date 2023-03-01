#! /usr/bin/python
# -*- coding: utf-8 -*-
# @author izhangxm
# @date 2021/10/12
# @fileName train.py
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
from scipy.integrate import solve_ivp, odeint
from scipy.integrate import  RK45
from scipy.integrate import ode
from scipy.optimize import leastsq

# df = pd.read_csv("dataset/data.csv")


def odefunc(y, t, p):
    #Logistic differential equation
    return p[0] * y[0] * (1 - y[0])

times = np.arange(0.5, 5, 0.5)

ode_model = pm.ode.DifferentialEquation(func=odefunc, times=times, n_states=1, n_theta=1, t0=0)



print(ode_model)


a =pd.DataFrame()
a.to_excel("aaa.xlsx")
