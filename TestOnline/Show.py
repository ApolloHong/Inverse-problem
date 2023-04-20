# coding : utf-8
# professor : Helin Gong
# author : Lizhan Hong

__author__ = 'Lizhan Hong'


import numpy as np
import math
import sklearn
from InverseProblem.Project.Workspace.Train.Read import read
from InverseProblem.Project.Workspace.Train.ProSub import *
from InverseProblem.Project.Workspace.Train.KnnFor import KnnWithoutK
from InverseProblem.Project.Workspace.Train.KnnInv import KnnInvWithoutK1s, KnnInvWithoutK2s
import pandas as pd
import matplotlib.pyplot as plt



if __name__ == '__main__':

    # POD modes
    rModes = 50


    # ExtractTenSamples('..\Input\Inpower5517.txt',r'..\Input\alpha5517.txt',
    #                   '..\Input\Y5517.txt', '..\Input\power5517.out', [50*(_+1) for _ in range(10)])
    #
    # # # generate the specimens for online test
    # SpecimenInpower = np.loadtxt('..\Input\Inpower10.txt')
    SpecimenAlpha = np.loadtxt(r'../Input/alpha10.txt')
    # SpecimenY = np.loadtxt('..\Input\Y10.txt')
    q = np.loadtxt('../Input/q5517.txt')

    # True alpha:
    # TrueAlpha = [SpecimenAlpha[_].reshape((1,rModes)) @ q for _ in range(10)]
    # TrueAlpha = [KnnWithoutK(_)[1] @ q for _ in range(10)]
    #
    # Forward:
    # PredictAlphaFor = [KnnWithoutK(SpecimenInpower, _) @ q for _ in range(10)]
    # PredictAlphaFor = [KnnWithoutK(_)[0] @ q for _ in range(10)]

    PredictAlphaFor = []
    TrueAlpha = []
    for i in range(10):
        PredictAlphaFori, TrueAlphai = KnnWithoutK(i)
        # print(PredictAlphaFori,TrueAlphai.reshape((1,rModes)))
        PredictAlphaFor.append(np.dot(np.array(PredictAlphaFori), q))
        TrueAlpha.append(np.dot(np.array(TrueAlphai.reshape((1,rModes))), q))

    # print(TrueAlpha[0].shape,PredictAlphaFor[0].shape)


    # plot
    PlotKnnOnlie(PredictAlphaFor,TrueAlpha, 17)
    # PlotKnnOnlie(PredictAlphaInv1s,TrueAlpha,17)
    # PlotKnnOnlie(PredictAlphaInv2s,TrueAlpha,17)


    # four graph in one
    PlotEveryDetail(1)
    # Find the K
    PlotInvK()
    PlotForK()
    PlotResult2s()
    # print(read(r'..\Input\Y10.txt','txt'))


    # for _ in range(10):
    #    print(KnnWithoutK(50*_+1))




