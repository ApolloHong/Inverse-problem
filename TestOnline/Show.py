# coding : utf-8
# professor : Helin Gong
# author : Lizhan Hong

__author__ = 'Lizhan Hong'


import numpy as np
import math
import sklearn
from InverseProblem.pod.Workspace.Train.Read import read
from InverseProblem.pod.Workspace.Train.FapyProSub import *
from InverseProblem.pod.Workspace.Train.FapyKnnFor import KnnWithoutK
from InverseProblem.pod.Workspace.Train.FapyKnnInv import KnnInvWithoutK1s, KnnInvWithoutK2s
import pandas as pd
import matplotlib.pyplot as plt



if __name__ == '__main__':

    # # POD modes
    # rModes = 50
    # ExtractTenSamples('..\Input\Inpower5517.txt',r'..\Input\alpha5517.txt',
    #                   '..\Input\Y5517.txt', '..\Input\power5517.out', [500*(_+1) for _ in range(10)])
    #
    # # generate the specimens for online test
    # #SpecimenInpower = read('..\Input\Inpower10.txt','txt')
    # SpecimenInpower = np.loadtxt('..\Input\Inpower10.txt')
    # SpecimenAlpha = np.loadtxt(r'..\Input\alpha10.txt')
    # SpecimenY = np.loadtxt('..\Input\Y10.txt')
    # q = np.loadtxt('..\Input\q5517.txt')
    #
    # # True alpha:
    # TrueAlpha = [SpecimenAlpha[_].reshape((1,rModes)) @ q for _ in range(10)]
    # # print(TrueAlpha[0].shape)
    # #print(SpecimenInpower.shape)
    #
    # # # Forward:
    # # print(KnnWithoutK(SpecimenInpower,1))
    # # print(KnnWithoutK(SpecimenInpower,0))
    # PredictAlphaFor = [KnnWithoutK(SpecimenInpower, _) @ q for _ in range(10)]
    # # print(PredictAlphaFor[0].shape)
    #
    #
    # # Inv:
    # PredictAlphaInv1s = [KnnInvWithoutK1s(SpecimenY, _) @ q for _ in range(10)]
    # PredictAlphaInv2s = [KnnInvWithoutK2s(SpecimenY, _) @ q for _ in range(10)]
    # # print(PredictAlphaInv2s[0].shape)
    #
    #
    # # plot
    # PlotKnnOnlie(PredictAlphaFor,TrueAlpha, 17)
    # PlotKnnOnlie(PredictAlphaInv1s,TrueAlpha,17)
    # PlotKnnOnlie(PredictAlphaInv2s,TrueAlpha,17)


    PlotEveryDetail(1)
    PlotInvK()
    PlotForK()
    print(read('..\Input\Power10.txt','txt'))




