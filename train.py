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
from torchsummary import summary


def train_loop(dataloader, model, loss_fn, optimizer, dev):
    num_batches = len(dataloader)
    rmse_loss, mse_loss, r2_loss = 0, 0, 0
    for _, _, X, ori_Y, Y in dataloader:
        X = X.to(dev)
        Y = Y.to(dev)
        ori_Y = ori_Y.to(dev)
        pred = model(X)
        loss_dict = loss_fn(pred, ori_Y)

        rmse_loss += loss_dict['rmse'].item()
        mse_loss += loss_dict['mse'].item()
        r2_loss += loss_dict['r2'].item()

        loss = loss_dict['mse']

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    rmse_loss /= num_batches
    mse_loss /= num_batches
    r2_loss /= num_batches

    loss_dict = {"rmse": rmse_loss, "mse": mse_loss, "r2": r2_loss}

    return loss_dict


def test_loop(dataloader, model, loss_fn, dev):
    pred_all, ori_Y_all = None, None
    with torch.no_grad():
        for _, _, X, ori_Y, Y in dataloader:
            X = X.to(dev)
            Y = Y.to(dev)
            ori_Y = ori_Y.to(dev)
            pred = model(X)

            pred_all = pred if pred_all is None else torch.cat([pred_all, pred], dim=0)
            ori_Y_all = ori_Y if ori_Y_all is None else torch.cat([ori_Y_all, ori_Y], dim=0)

    loss_dict = loss_fn(pred, ori_Y)
    rmse_loss = loss_dict['rmse'].item()
    mse_loss = loss_dict['mse'].item()
    r2_loss = loss_dict['r2'].item()

    loss_dict = {"rmse": rmse_loss, "mse": mse_loss, "r2": r2_loss}
    return loss_dict


def main(opt):
    data_loc = opt.data_loc
    label_loc = opt.label_loc
    dev = opt.dev
    dataset_dir = opt.dataset_dir

    save_model_path = os.path.join(dataset_dir, opt.run_name, "best.pth")
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)

    if opt.model_pth is None:
        restore_model_pth = os.path.join(dataset_dir, opt.run_name, "best.pth")
    else:
        restore_model_pth = opt.model_pth

    train_data = MyDataset(os.path.join(dataset_dir, "train.csv"), time_loc=opt.time_loc, data_loc=data_loc,
                           label_loc=label_loc)
    test_data = MyDataset(os.path.join(dataset_dir, "test.csv"), time_loc=opt.time_loc, data_loc=data_loc,
                          label_loc=label_loc)
    train_dataloader = DataLoader(train_data, batch_size=256, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=2048, shuffle=True)

    loss_rmse_fn = RMSE()
    model = NeuralNetwork(data_loc=data_loc, label_loc=label_loc, fc_nums=opt.fc_nums)
    model.to(dev)

    summary(model, input_size=(len(opt.data_loc),))

    # 随机数据 不能恢复模型 模型已经见过测试数据了
    if os.path.exists(restore_model_pth):
        print('try to loading model....')
        try:
            ckpt = torch.load(restore_model_pth)
            model.load_state_dict(ckpt)
            print('loaded model')
        except Exception as e:
            print(e)

    learning_rate = 0.01
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate,weight_decay=1e-4)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    epochs = opt.epoch
    test_margin = 2

    best_te_loss = {}
    min_loss = np.Inf

    print(f"training on {len(train_data)} examples, testing on {len(test_data)} examples")
    # pbar = tqdm(total=epochs, ascii=True, ncols=200)
    for t in range(epochs):
        tr_loss = train_loop(train_dataloader, model, loss_rmse_fn, optimizer, dev)
        if t == epochs - 1 or t % test_margin == 0:
            te_loss = test_loop(test_dataloader, model, loss_rmse_fn, dev)
            te_mse = te_loss['mse']
            if te_mse < min_loss:
                print('updating best model ......')
                torch.save(model.state_dict(), save_model_path)
                min_loss = te_mse
                best_te_loss = te_loss

        train_str = " ".join([f"{k}:{v:.4f}" for k, v in tr_loss.items()])
        test_str = " ".join([f"{k}:{v:.4f}" for k, v in te_loss.items()])
        best_str = " ".join([f"{k}:{v:.4f}" for k, v in best_te_loss.items()])

        p_str = f"[E:{t + 1}/{epochs} Train[{train_str}] TEST[{test_str}] Best[{best_str}]"
        print(p_str)
        # pbar.set_description(p_str)
        # pbar.update(1)
    # pbar.close()

    print(f"training has done! Best loss on testing dataset is {min_loss} and model params saved.")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default="dataset/data_2014_7-10_full")
    parser.add_argument('--run_name', type=str, default="run_without_ph_wl")
    parser.add_argument('--model_pth', type=str, default="dataset/data_2014_7-10_full/run_without_ph_wl/best.pth")
    parser.add_argument('--fc_nums', type=int, nargs='+', default=[4, 128, 256])
    parser.add_argument('--time_loc', type=int, nargs='+', default=[0])
    parser.add_argument('--data_loc', type=int, nargs='+', default=[1, 2, 3, 5])
    parser.add_argument('--label_loc', type=int, nargs='+', default=[7])
    parser.add_argument('--epoch', type=int, default=1000 * 4)
    parser.add_argument('--dev', type=str, default="cuda:0")
    opt = parser.parse_args()

    print(opt)
    # exit(0)
    return opt


if __name__ == '__main__':
    opt = get_args()
    main(opt)
    print('*' * 80 + '\n\n')


# python train.py --dataset_dir="dataset/data_2015_7-11_full" \
# --run_name="run_without_sal_wl" \
# --fc_nums 4 128 256 \
# --data_loc 1 2 4 5


# python train.py --dataset_dir "dataset/data_2014_7-10_full" \
# --run_name "run_without_sal_wl" \
# --fc_nums 4 128 256 \
# --data_loc 1 2 4 5


# python train.py --dataset_dir "dataset/data_2014_7-10_full" \
# --run_name "run_without_ph_wl" \
# --fc_nums 4 128 256 \
# --data_loc 1 2 3 5

# python train.py --dataset_dir "dataset/data_2015_7-11_full" \
# --run_name "run_without_ph_wl" \
# --fc_nums 4 128 256 \
# --data_loc 1 2 3 5
