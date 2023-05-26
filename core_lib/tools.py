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


def get_format_time(f_s=None):
    haomiao = str(time.time()).split('.')[-1]
    if f_s is None:
        f_s = "%Y-%-m-%d %H:%M:%S"
        # return datetime.now().strftime(f_s) + haomiao
    return datetime.now().strftime(f_s)

def get_predict_ks(idata):
    parames_summary = az.summary(idata, round_to=10)
    ks_names = [f"k{x+1}" for x in range(11)]

    predict_ks = []
    for k_name in ks_names:
        k_v = parames_summary["mean"][k_name]
        predict_ks.append(k_v)
    return np.array(predict_ks)


def get_predict_starts(cct_names, idata):
    parames_summary = az.summary(idata, round_to=10)
    s_names = [f"{x}_s" for x in cct_names]
    predict_s = []
    for s_name in s_names:
        k_v = parames_summary["mean"][s_name]
        predict_s.append(k_v)
    return np.array(predict_s)


def r2_loss(pred, y):
    r2_loss = 1 - np.square(pred - y).sum() / np.square(y - np.mean(y)).sum()
    return r2_loss

class WriteProcessor:
    def __init__(self):
        self.buf = ""
        self.real_stdout = sys.stdout
    def write(self, buf):
        # emit on each newline
        while buf:
            try:
                newline_index = buf.index("\n")
            except ValueError:
                # no newline, buffer for next call
                self.buf += buf
                break
            # get data to next newline and combine with any buffered data
            data = self.buf + buf[:newline_index + 1]
            self.buf = ""
            buf = buf[newline_index + 1:]
            # perform complex calculations... or just print with a note.
            if "lsoda--  warning" in data:
                continue
            self.real_stdout.write("fiddled with " + data)
    
    def flush(self):
        self.real_stdout.flush()

# class XJOutFilter(object):
    
#     def __init__(self) -> None:
#         pass

#     def __enter__(self):
#         self.ff = redirect_stdout(WriteProcessor())
#         self.ff.__enter__()
#         return self.ff

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         self.ff.__exit__(exc_type, exc_val, exc_tb)
#         return True

from io import StringIO
import contextlib
import contextlib, sys

@contextlib.contextmanager
def log_print(file):
    # capture all outputs to a log file while still printing it
    class Logger:
        def __init__(self, file):
            self.terminal = sys.stdout
            self.log = file

        def write(self, message):
            self.terminal.write("www" + message)
            self.log.write("ddddd" + message)

        def __getattr__(self, attr):
            return getattr(self.terminal, attr)

    logger = Logger(file)

    _stdout = sys.stdout
    _stderr = sys.stderr
    sys.stdout = logger
    sys.stderr = logger
    try:
        yield logger.log
    finally:
        sys.stdout = _stdout
        sys.stderr = _stderr
