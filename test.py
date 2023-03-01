#! /usr/bin/python
# -*- coding: utf-8 -*-
# @author izhangxm
# @date 2021/12/12
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
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from dataset import MyDataset, load_des
from model import NeuralNetwork, RMSE
from tqdm import tqdm, trange
import argparse
from argparse import Namespace
import model as loss_func
from torchsummary import summary

def main(opt):
    
    data_loc = opt.data_loc
    label_loc =  opt.label_loc
    
    dev = opt.dev
    dataset_csv = opt.dataset_csv
    model_path = opt.model_pth
    out_path = opt.out_path
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    dataset= MyDataset( dataset_csv,time_loc=opt.time_loc, data_loc=data_loc, label_loc=label_loc, mode='predict')
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    
    loss_fn = RMSE()
    
    model = NeuralNetwork(data_loc=data_loc, label_loc=label_loc, fc_nums=opt.fc_nums)
    model.to(dev)
    
    summary(model, input_size=(len(opt.data_loc),))
    
    
    if os.path.exists(model_path):
        print('loading model....')
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt)
    
    out_f = open(out_path, 'w+', encoding='utf8')
    out_f.write(f"date-UTC,Cos-Julian day,T,SAL,PH,domgl,WL,co2 uatm, predict\n")
    
    with torch.no_grad():
        pred_all, ori_Y_all = None, None
        for _time, ori_X, X, ori_Y, Y in dataloader:
            X = X.to(dev)
            Y = Y.to(dev)
            ori_Y = ori_Y.to(dev)
            pred = model(X)
    
            pred_all = pred if pred_all is None else torch.cat([pred_all, pred], dim=0)
            ori_Y_all = ori_Y if ori_Y_all is None else torch.cat([ori_Y_all, ori_Y], dim=0)
            
            valid_bool_list = (torch.sum(~torch.isnan(pred_all), dim=1)>0) & (torch.sum(~torch.isnan(ori_Y_all), dim=1)>0)
            pred_all = pred_all[valid_bool_list]
            ori_Y_all = ori_Y_all[valid_bool_list]
            
            loss_dict ={k:v.item() for k,v in loss_fn(pred_all, ori_Y_all).items()}
            print(loss_dict)
            
            for _t, _x, p, y in zip(list(_time[0]), ori_X, pred, ori_Y):
                _d = ','.join([f"{x.cpu():.4f}" for x in _x])
                out_f.write(f"{_t},{_d},{y[0].cpu():.4f},{p[0].cpu():.4f}\n")

        valid_bool_list = (torch.sum(~torch.isnan(pred_all), dim=1)>0) & (torch.sum(~torch.isnan(ori_Y_all), dim=1)>0)
        pred_all = pred_all[valid_bool_list]
        ori_Y_all = ori_Y_all[valid_bool_list]
        loss_dict ={k:v.item() for k,v in loss_fn(pred_all, ori_Y_all).items()}
        
        print(loss_dict)

    out_f.close()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_csv', type=str, default="dataset/data_2015_7-11_full/test copy.csv")
    parser.add_argument('--dataset_des', type=str, default="dataset/data_2015_7-11_full/des.json")
    parser.add_argument('--model_pth', type=str, default="dataset/data_2015_7-11_full/run/best.pth")
    parser.add_argument('--fc_nums', type=tuple, default=[6,128, 256])
    parser.add_argument('--time_loc', type=tuple, default=(0,))
    parser.add_argument('--data_loc', type=tuple, default=(1, 2, 3, 4,5,6))
    parser.add_argument('--label_loc', type=tuple, default=(7, ))
    parser.add_argument('--out_path', type=str, default="dataset/data_2015_7-11_full/predict_test copy.csv")
    parser.add_argument('--dev', type=str, default="cuda:0")
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = get_args()
    # opt = Namespace(dataset_csv="dataset/data_v1/2014.7-10-5min-without-par.csv"
    #                 ,dataset_des="dataset/data_v1/des.json"
    #                 ,model_pth="dataset/data_v1/run5_new/best.pth"
    #                 ,time_loc = (0,)
    #                 ,data_loc = (1, 2, 3, 4,5,6)
    #                 ,label_loc = (7, )
    #                 ,out_path="dataset/data_v1/predict_2014.7-10-5min-without-par.csv"
    #                 ,dev = "cuda:0"
    #                 )
    main(opt)
    print('*' * 80 + '\n\n')



