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
import time
import subprocess
import core
for x in range(2):
    s_time = time.time()
    str_f_s_time = core.get_format_time()
    time.sleep(1)
    cost_time = int(time.time() - s_time)
    str_f_e_time = core.get_format_time()
    save_file_path = "addd"
    subprocess.call(f'echo \'"{save_file_path}","s:{str_f_s_time}","e:{str_f_e_time}","{cost_time}"\' >> cost_time.csv', shell=True)
    # subprocess.call(f'echo \'`{save_file_path}`,`{str_f_s_time}`,`{str_f_e_time}`,`{cost_time}`\' >> cost_time.csv', shell=True)
    