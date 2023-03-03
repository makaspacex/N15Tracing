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
import numpy as np
import pymc as pm
from itertools import product
from multiprocessing import Process, Queue, cpu_count
import pickle
import os
from pathlib import Path
import core
import multiprocessing
import time
import subprocess
import argparse
# multiprocessing.set_start_method('spawn')  # default on WinOS or MacOS
# multiprocessing.set_start_method('fork')   # default on Linux (UnixOS)

# 请注意：我说 分支fork 在初始化创建多进程的时候比 分产spawn 快，而不是说高性能计算会比较快。
# 通常高性能计算需要让程序运行很久，因此为了节省内存以及进程安全，我建议选择 spawn。
# https://zhuanlan.zhihu.com/p/340657122


def run_model(opt):
    def __init__(self, p_name, dataset, k_queue:Queue, t_eval, cores=1, draws=2000):
        self.p_name = p_name
        self.dataset = dataset
        self.k_queue = k_queue
        self.t_eval = t_eval
        self.cores = cores
        self.draws = draws

    def run(self):
        while not self.k_queue.empty():
            save_file_path = None
            try:
                k_kinetics = np.array(self.k_queue.get(1, timeout=100))
                print(f"{self.p_name} {k_kinetics}")
                k_str = "".join([f"{x}" for x in k_kinetics])
                k_order = int(k_str,2)
                save_file_path = f"saved_idata-v2/{k_order}-idata.dt"
                
                Path(save_file_path).parent.mkdir(parents=True, exist_ok=True)

                if  Path(save_file_path).exists() or Path(save_file_path + '.fail').exists():
                    continue
                
                Path(save_file_path+'.running').touch(exist_ok=True)
                
                s_time = time.time()
                str_f_s_time = core.get_format_time()

                dataset = self.dataset
                mcmc_model = core.get_model(dataset, self.t_eval, k_kinetics, k_sigma_priors=0.01, kf_type=0, distance=core.distance_func, epsilon=core.epsilon, c0_type=0)
                idata = pm.sample_smc(draws=self.draws, model=mcmc_model,  chains=self.cores, cores=self.cores, progressbar=False)
                pickle.dump(idata,open(save_file_path, 'wb'))
                cost_time = int(time.time() - s_time)
                str_f_e_time = core.get_format_time()
                subprocess.call(f'echo \'"{save_file_path}", "s:{str_f_s_time}","e:{str_f_e_time}","{cost_time}"\' >> model-cost_time.csv', shell=True)

            except Exception as e:
                if save_file_path is not None:
                    Path(save_file_path + '.fail').touch()
            finally:
                if Path(save_file_path + '.running').exists():
                    os.remove(save_file_path + '.running')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path',type=str, default='dataset/data.csv')
    parser.add_argument('--kk', type=str, required=True)
    parser.add_argument('--cores', type=int, default=1)

    opts = parser.parse_args()


    print(opts)

    exit()
    multiprocessing.set_start_method('spawn')
    kk_list_all = list(product([0,1], repeat=11))[:1024]
    q = Queue(10000)
    for k_k in kk_list_all:
        q.put(k_k)
    dataset = core.MyDataset("dataset/data.csv")
    df = dataset.get_df()
    t_eval = df['time'].values

    cores = 1

    p_list = []
    n_cpu = int(cpu_count()/cores) -2

    print(cpu_count(), n_cpu)
    
    for i in range(n_cpu):
        p = NTraceModel(f"Process-{i}", dataset, q, t_eval=t_eval, cores=cores)
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()
    print("all_finished")

