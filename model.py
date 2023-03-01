#! /usr/bin/python
# -*- coding: utf-8 -*-
# @author izhangxm
# @date 2021/12/12
# @fileName model.py
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
from torch import nn
import torch

criterion = nn.SmoothL1Loss()


def mse(pred, y):
    eps = 1e-8
    return criterion(pred, y)
    # return torch.sqrt(torch.square(pred - y).mean().sum() + eps)
    # return torch.square(pred - y).mean().sum() + eps


def rmse(pred, y):
    rmse_loss = torch.sqrt(torch.square(pred - y).mean())
    return rmse_loss


def r2(pred, y):
    # mean = 2479.03125
    r2_loss = 1 - torch.square(pred - y).sum() / torch.square(y - torch.mean(y)).sum()
    # r2_loss = 1 - torch.square(pred-y).sum()/torch.square(y-mean).sum()
    return r2_loss


class RMSE(nn.Module):

    def __init__(self):
        super(RMSE, self).__init__()

    def forward(self, pred, y):
        rmse_loss = rmse(pred, y)
        mse_loss = mse(pred, y)
        r2_loss = r2(pred, y)

        loss_dict = {"rmse": rmse_loss, 'r2': r2_loss, 'mse': mse_loss}
        return loss_dict


class NeuralNetwork(nn.Module):
    def __init__(self, data_loc, label_loc, fc_nums=[6, 128, 256]):
        super(NeuralNetwork, self).__init__()

        self.data_loc = data_loc
        self.label_loc = label_loc
        # c1_out, c2_out, c3_out = 128, 64, 32
        # self.conv_stack = nn.Sequential(
        #     nn.Conv1d(1, c1_out, kernel_size=3,padding=1),
        #     # nn.BatchNorm1d(c1_out),
        #     nn.Sigmoid(),
        #     nn.Conv1d(c1_out, c2_out, kernel_size=3, padding=1),
        #     # nn.BatchNorm1d(c2_out),
        #     nn.ReLU(),
        #     nn.Conv1d(c2_out, c3_out, kernel_size=3,padding=1),
        #     nn.BatchNorm1d(c3_out),
        #     nn.ReLU(),
        # )

        # self.input_dim = len(data_loc)
        # self.output_dim = len(label_loc)
        # self.data_loc, self.label_loc = list(data_loc), list(label_loc)

        # self.flatten = nn.Flatten()
        # multi_n = 32
        # self.linear_stack = nn.Sequential(
        #     nn.Linear(c3_out * self.input_dim, multi_n * self.input_dim),
        #     nn.Linear(multi_n * self.input_dim, self.output_dim),
        # )

        # fc_nums = [len(data_loc), 32,32,32,32, 32, 128]
        # fc_nums = [len(data_loc),  128,256]

        self.fc_layers = nn.Sequential()
        for i, fc_n in enumerate(fc_nums[1:]):
            self.fc_layers.add_module(f"fc{i}", nn.Sequential(nn.Linear(fc_nums[i], fc_n),
                                                              nn.LayerNorm([fc_n], elementwise_affine=True), nn.Tanh()))
        self.fc_layers.add_module(f"fc{100}", nn.Sequential(nn.Linear(fc_nums[-1], len(label_loc))))

    def forward(self, x):
        # x = x.unsqueeze(dim=1)
        # x = self.conv_stack(x)
        # x = self.flatten(x)
        # logits = self.linear_stack(x)

        logits = self.fc_layers(x)

        return logits
