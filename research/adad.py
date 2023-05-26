import time

import numpy as np
import matplotlib.pyplot as plt
import random
from core import DynamicShowPlot


with DynamicShowPlot(block=False):
    # plt.rcParams["figure.figsize"] = [7.50, 3.50]
    # plt.rcParams["figure.autolayout"] = True

    m = 2
    n = 4

    fix, axes = plt.subplots(nrows=m, ncols=n)
    hexadecimal_alphabets = '0123456789ABCDEF'

    color = ["#" + ''.join([random.choice(hexadecimal_alphabets)
                            for j in range(6)]) for i in range(m * n)]

    for i in range(m):
        for j in range(n):
            axes[i][j].clear()
            axes[i][j].plot(np.random.rand(10), np.random.rand(10),
                            color=color[100 % np.random.randint(1, len(color))])
            plt.pause(0.1)


with DynamicShowPlot(block=True):
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True

    m = 2
    n = 4

    fix, axes = plt.subplots(nrows=m, ncols=n)
    hexadecimal_alphabets = '0123456789ABCDEF'

    color = ["#" + ''.join([random.choice(hexadecimal_alphabets)
                            for j in range(6)]) for i in range(m * n)]

    for i in range(m):
        for j in range(n):
            axes[i][j].clear()
            axes[i][j].plot(np.random.rand(10), np.random.rand(10),
                            color=color[100 % np.random.randint(1, len(color))])
            plt.draw()
            plt.pause(0.1)



print("+++++++++++++++++")

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

m = 2
n = 4

fix, axes = plt.subplots(nrows=m, ncols=n)
hexadecimal_alphabets = '0123456789ABCDEF'

color = ["#" + ''.join([random.choice(hexadecimal_alphabets)
                        for j in range(6)]) for i in range(m * n)]

for i in range(m):
    for j in range(n):
        axes[i][j].clear()
        axes[i][j].plot(np.random.rand(10), np.random.rand(10),
                        color=color[100 % np.random.randint(1, len(color))])
        plt.draw()
        plt.pause(0.1)
