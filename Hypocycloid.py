#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np


def hypocycloid(theta, k=4):
    x = ((k-1)*np.cos(theta) + np.cos((k-1)*theta)) / k
    y = ((k-1)*np.sin(theta) - np.sin((k-1)*theta)) / k
    return x, y

if __name__ == '__main__':
    theta = np.linspace(0, 2*np.pi, 256)
    fig, ax = plt.subplots()
    print(dir(fig))
    # fig.plot(x,y)
