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


def get_dcdt_func_for_sunode(k_kinetics):
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


def get_dcdts_for_scipy(c_first=False):

    def dcdt_func(c, t, *args):
        # print(c, t, ks, k_kinetics)
        # print()
        if not c_first:
            _x = c
            c = t
            t = _x

        ks, k_kinetics = args
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

    return dcdt_func


def dcdt_func_for_diffrax(t, c, args):
    
    ks, k_kinetics = args
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

def get_dcdts_for_solve_ivp():
    return get_dcdts_for_scipy(c_first=False)

def get_dcdts_for_scipy_odeint():
    return get_dcdts_for_scipy(c_first=True)


# simulator function
def competition_model(rng, t_eval, y0,  ks, k_kinetics, size=None):
    args = (ks, k_kinetics)
    y = xj_diff_solve_ivp(y0, t_eval, args)
    return y

def xj_diff_solve_ivp(y0, t_eval, args, ivp_first=False):
    if ivp_first:
        y_s = solve_ivp(get_dcdts_for_solve_ivp, t_span=(t_eval[0], t_eval[-1]), y0=y0, t_eval=t_eval, args=args)
    if ivp_first and len(y_s.t) == len(t_eval):
        y = y_s.y.transpose(1,0)
    else:
        y = odeint(get_dcdts_for_scipy_odeint(), y0=y0, t=t_eval, args=args)
    return y
