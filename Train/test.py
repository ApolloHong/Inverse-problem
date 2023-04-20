# coding : utf-8
# professor : Helin Gong
# author : Lizhan Hong

__author__ = 'Lizhan Hong'

import matplotlib.pyplot as plt
from ProSub import *
import numpy as np
import pandas as pd
import scipy.sparse as spr
from matplotlib.ticker import LinearLocator

plt.rc('font', family='Times New Roman', size=1)
plt.style.use('seaborn')  # Set the theme of seaborn



if __name__ == '__main__':
    # data space
    selectedGrid = [8, 10, 12, 14, 24, 26, 28, 38, 40, 42, 49, 51]
    selectedSection = [1, 5, 9, 13, 17, 21, 25]
    PowerData5517 = read('../Input/powerIAEA5517.txt', 'txt')
    numGridQuarter = 52
    numGrid = 177
    numSection = 28

    # # test space

    # # select the powerdata of 5517
    # PowerData: The Power matrix we obtained. 5517 * 1456
    sensors = read('../Input/sensors.txt', 'txt')
    Y5517 = pd.DataFrame(sensors @ PowerData5517.T)
    Y5517.to_csv('Y5517.txt', sep=' ', index=False, header=False)