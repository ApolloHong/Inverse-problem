# coding : utf-8
# professor : Helin Gong
# author : Lizhan Hong

__author__ = 'Lizhan Hong'

import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
import joblib
from ProSub import *
from OptimaizeSub import *

plt.rc('font', family='Times New Roman', size=15)
plt.style.use('seaborn')  # Set the theme of seaborn


def KnnForward():
    # 导入数据
    input_data = np.loadtxt('../Input/Inpower5517.txt')
    input_label = np.loadtxt(r'../Input/alpha5517.txt')

    # 划分数据集
    train_input, test_input, train_output, test_output = train_test_split(input_data, input_label, random_state=42)

    # 计算错误率,寻找合适的k
    Error = []
    k_range = range(1, 21)
    error_k = 1
    k_appropriate = 0

    for k in k_range:
        error = 0
        for i in range(test_input.shape[0]):
            test = test_input[i, :]
            error = error + np.linalg.norm(
                KNN(k, test, train_input, train_output) - test_output[i, :]) / np.linalg.norm(
                test_output[i, :])  # 计算每个的误差并求和
        error = error / test_input.shape[0]
        Error.append(error)
        if (error < error_k):
            k_appropriate = k
            error_k = error

    # 输出合适的k及对应的错误率
    print(k_appropriate, error_k)

    return Error

def KnnForwardSklearn():
    # 导入数据集
    data = np.loadtxt('../Input/Inpower5517.txt')
    label = np.loadtxt(r'../Input/alpha5517.txt')

    # 划分数据集
    train_input, test_input, train_output, test_output = train_test_split(data, label, random_state=42)

    # 计算错误率，寻找合适的k
    Error = []
    k_range = range(1, 21)
    error_k = 1
    k_appropriate = 0
    for k in k_range:
        KN_model = KNeighborsRegressor(n_neighbors=k)
        KN_model.fit(train_input, train_output)
        error = 0
        for i in range(test_input.shape[0]):
            test_line = test_input[i, :]
            test_line.shape = (1, test_line.size)
            predict = KN_model.predict(test_line)
            label = test_output[i, :]
            error = error + np.linalg.norm(label - predict) / np.linalg.norm(label)
        error = error / test_input.shape[0]
        Error.append(error)
        if (error < error_k):
            error_k = error
            k_appropriate = k

    # 输出合适的k及错误率
    print(k_appropriate, error_k)

    return Error

def KnnForwardSklearnPower():
    # 导入数据集
    data = np.loadtxt('../Input/Inpower5517.txt')
    label = np.loadtxt(r'../Input/alpha5517.txt')
    q = np.loadtxt('../Input/q5517.txt')

    # 划分数据集
    train_input, test_input, train_output, test_output = train_test_split(data, label, random_state=42)

    # 计算错误率，寻找合适的k
    Error = []
    k_range = range(1, 21)
    error_k = 1
    k_appropriate = 0
    for k in k_range:
        KN_model = KNeighborsRegressor(n_neighbors=k)
        KN_model.fit(train_input, train_output)
        error = 0
        for i in range(test_input.shape[0]):
            test_line = test_input[i, :]
            test_line.shape = (1, test_line.size)
            predict = KN_model.predict(test_line)
            label = test_output[i, :]
            testout = np.dot(predict, q)
            testoutput = np.dot(label, q)
            error = error + np.linalg.norm(testout - testoutput) / np.linalg.norm(testoutput)
        error = error / test_input.shape[0]
        Error.append(error)
        if (error < error_k):
            error_k = error
            k_appropriate = k

    # 输出合适的k及错误率
    print(k_appropriate, error_k)

    # # 生成图像
    # plt.figure()
    # plt.plot(k_range, Error)
    # plt.xlabel('Value of k for KNN')
    # plt.ylabel('the average of error rate')
    # plt.savefig('handwrite mu--alpha,q')
    # plt.show()

    return Error

def KnnForwardPower():

    # 导入数据
    input_data = np.loadtxt('../Input/Inpower5517.txt')
    input_label = np.loadtxt(r'../Input/alpha5517.txt')
    q = np.loadtxt('../Input/q5517.txt')

    # 划分数据集
    train_input, test_input, train_output, test_output = train_test_split(input_data, input_label, random_state=42)

    # 计算错误率,寻找合适的k
    Error = []
    k_range = range(1, 21)
    error_k = 1
    k_appropriate = 0

    for k in k_range:
        error = 0
        for i in range(test_input.shape[0]):
            test = test_input[i, :]
            error = error + np.linalg.norm(
                KNN(k, test, train_input, train_output) - test_output[i, :]) / np.linalg.norm(
                test_output[i, :])  # 计算每个的误差并求和
        error = error / test_input.shape[0]
        Error.append(error)
        if (error < error_k):
            k_appropriate = k
            error_k = error

    # 输出合适的k及对应的错误率
    print(k_appropriate, error_k)

    return Error


def KnnWithoutK(nSample):
    # 导入数据
    input_data_mu = np.loadtxt('../Input/Inpower5517.txt')
    input_data_mu = normalize(input_data_mu)
    input_data_alpha = np.loadtxt(r'../Input/alpha5517.txt')

    # 划分数据集
    train_input, test_input, train_output, test_output = train_test_split(input_data_mu, input_data_alpha,
                                                                          test_size=0.25, random_state=42, shuffle=True)

    # 训练模型
    k = 4
    KN_model_alpha = KNeighborsRegressor(n_neighbors=k, weights='distance', p=1, metric='minkowski')
    KN_model_alpha.fit(train_input, train_output)

    # 输入需要测试的数据
    test = test_input[nSample, :]
    label=test_output[nSample,:]

    # 预测
    test.shape = (1, test.size)
    predict_alpha = KN_model_alpha.predict(test)
    # error=np.linalg.norm(predict_alpha-label)/np.linalg.norm(label)

    # 输出
    return predict_alpha, label


def trainKnn(k = 5):
    '''

    :param k: the k nearist vectors we used. (In our case 5 performs the best)
    :return:
    '''
    # data prepro
    sigmas = [0, 1, 2, 3, 4, 5]
    sigma = 0
    sensors = pd.read_csv('../Input/sensors.txt', header=None, delimiter=' ', dtype=float)
    field = pd.read_csv('../Input/powerIAEA18480.txt', delimiter=' ', header=None, dtype=float)
    observations = np.dot(field, sensors.T)
    input_data_mu_nor = np.loadtxt('../Input/inpowerNor18480_4.txt')
    scaling_index = np.loadtxt('../Input/scalingNor.txt')
    input_data_alpha = np.loadtxt(r'../Input/powerIAEA18480coef.txt')



    # train_input, test_input, train_output, test_output = train_test_split(input_data_mu_nor, input_data_alpha,
    #                                                                       test_size=0.25, random_state=42)


    # # add noise to observation
    # observations = observations + np.random.normal(0, sigma / 100.0, observations.shape) * observations
    # split train, validate, test dataset

    # split train dataset, test dataset
    mu_train_full, mu_test, alpha_train_full, alpha_test = train_test_split(input_data_mu_nor, input_data_alpha, test_size=0.05, \
                                                                      random_state=42, stratify=input_data_alpha)
    # split validate dataset and the train dataset
    in_train, in_val, alpha_train, alpha_val = train_test_split(mu_train_full, alpha_train_full,  test_size=0.2, random_state=42)

    # Train a KNN model on the data
    KNN_model_alpha = KNeighborsRegressor(n_neighbors=k, weights='distance', p=1, metric='minkowski')
    KNN_model_alpha.fit(in_train, alpha_train)

    # # inverse_normalize
    # mu_test = mu_test @ scaling_index

    # Save the model to a file using pickle
    with open('../Input/knn4.pkl', 'wb') as file:
        joblib.dump(KNN_model_alpha, file)
    # Save the test_input to a file using pickle
    with open('../Input/knntest_input.pkl', 'wb') as file:
        joblib.dump(mu_test, file)
    # Save the test_output to a file using pickle
    with open('../Input/knntest_output.pkl', 'wb') as file:
        joblib.dump(alpha_test, file)

def FindBestModelKnn(k = 5):
    # data prepro
    sensors = pd.read_csv('../Input/sensors.txt', header=None, delimiter=' ', dtype=float)
    field = pd.read_csv('../Input/powerIAEA18480.txt', delimiter=' ', header=None, dtype=float)
    observations = pd.read_csv('../Input/Y18480.txt', delimiter=' ', header=None, dtype=float)
    input_data_mu_nor = np.loadtxt('../Input/inpowerNor18480_4.txt')
    scaling_index = np.loadtxt('../Input/scalingNor.txt')
    input_data_alpha = np.loadtxt(r'../Input/powerIAEA18480coef.txt')
    basis = pd.read_csv('../Input/powerIAEA18480basis.txt', delimiter=' ', header=None, dtype=float)  # shape:(n, M), n=150, M=52*28=1456

    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    # print(input_data_alpha[:,0].shape, input_data_mu_nor.shape)
    kf.get_n_splits(input_data_mu_nor[:,0], input_data_alpha[:,0])

    Error = []
    # create the split index data
    folds = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4']
    label = ['Train_index', 'Test_index']
    labels = [folds[_] + label[__] for _ in range(len(folds)) for __ in range(len(label))]
    # print(labels)
    split_index = {}
    for i, (train_index, test_index) in enumerate(kf.split(input_data_mu_nor[:,0], input_data_alpha[:,0])):
        print(f"Fold {i}:")
        # 13860
        print(f"  Train: index={train_index}")
        # 4620
        print(f"  Test:  index={test_index}")
        split_index[labels[i*2 + 0]] = train_index
        split_index[labels[i*2 + 1]] = test_index

        # print(train_index.shape)
        mu_train, mu_validate, alpha_train, alpha_validate = \
            input_data_mu_nor[train_index],input_data_mu_nor[test_index],input_data_alpha[train_index],input_data_alpha[test_index]
        # print(mu_train,mu_validate, alpha_train, alpha_validate)

        # Train a KNN model on the data
        KNN_model_alpha = KNeighborsRegressor(n_neighbors=k, weights='distance', p=1, metric='minkowski')
        KNN_model_alpha.fit(mu_train, alpha_train)

        error = 0
        # generate the testdata and calculate the error
        for j in test_index:
            test_data = input_data_mu_nor[j, :]
            test_data.shape = (1, test_data.size)
            alpha_predict = KNN_model_alpha.predict(test_data)
            field_predict = alpha_predict @ basis @ sensors.T
            field_true = input_data_alpha @ basis
            # calculate the L2 relative error
            error += field_error_L2(field_predict, field_true)
        error = error / len(test_index)

        Error.append(error)

    Error = np.array(Error)
    ibest = np.argmin(Error)


    keys = [str(key) for key in split_index.keys()]
    values = [list(value) for value in split_index.values()]
    split_index = pd.DataFrame(zip(keys, values), columns=['name', 'index'])
    split_index.to_csv('../Input/split_index_ibest'+ f'{ibest}' +'.txt')
    # print(split_index.iloc[0,1])
    # print(ibest)






    # with open('../Input/knn4.pkl', 'wb') as file:
    #     joblib.dump(KNN_model_alpha, file)
    # # Save the test_input to a file using pickle
    # with open('../Input/knntest_input.pkl', 'wb') as file:
    #     joblib.dump(mu_validate, file)
    # # Save the test_output to a file using pickle
    # with open('../Input/knntest_output.pkl', 'wb') as file:
    #     joblib.dump(alpha_validate, file)


def PlotResult1s():
    data = read('../Output/ForwardError.txt', 'txt')
    fig = plt.figure(figsize=(15,7))
    plt.subplots_adjust(left=0.1,bottom=0.1,right=0.9,top=0.9,wspace=0.3,hspace=0.1)


    ax1 = fig.add_subplot(121)
    line1, = ax1.plot(np.arange(1,21), data.iloc[0], 'g', alpha=0.5, lw=1.5, ls='-', marker='^')
    ax1.scatter(np.arange(1, 21), data.iloc[0], color='g', alpha=0.5, lw=1.5, ls='-', marker='^')
    ax1.legend('I')
    # create the twin subplot
    ax2 = ax1.twinx()
    line2, = ax2.plot(np.arange(1,21), data.iloc[1], 'b', alpha=0.5, lw=1.5, ls='-', marker='+')
    ax2.scatter(np.arange(1, 21), data.iloc[1], color='b', alpha=0.5, lw=1.5, ls='-', marker='+')
    ax2.legend('I')
    plt.legend((line1, line2), ['KnnForward', 'KnnForwardSklearn'], loc=0,
               title='Methods', ncol=1, markerfirst=False,
               numpoints=2, frameon=True, fancybox=True,
               facecolor='#d3d3da', edgecolor='b', shadow=True)

    # match the colors
    for tl in ax1.get_yticklabels():
        tl.set_color("g")
    for tl in ax2.get_yticklabels():
        tl.set_color("b")

    ax1.set_xlabel('Value of k for KNN')
    ax1.set_ylabel('The average of Alpha error rate')


    ax3 = fig.add_subplot(122)
    line3, = ax3.plot(np.arange(1,21), data.iloc[2], 'g', alpha=0.5, lw=1.5, ls='-', marker='^')
    ax3.scatter(np.arange(1, 21), data.iloc[2], color='g', alpha=0.5, lw=1.5, ls='-', marker='^')
    # create the twin subplot
    ax4 = ax3.twinx()
    line4, = ax4.plot(np.arange(1,21), data.iloc[3], 'b', alpha=0.5, lw=1.5, ls='-', marker='+')
    ax4.scatter(np.arange(1, 21), data.iloc[3], color='b', alpha=0.5, lw=1.5, ls='-', marker='+')
    plt.legend((line3, line4), ['KnnForwardPower', 'KnnForwardSklearnPower'], loc=0,
               title='Methods', ncol=1, markerfirst=False,
               numpoints=2, frameon=True, fancybox=True,
               facecolor='#d3d3da', edgecolor='b', shadow=True)

    # match the colors
    for tl in ax3.get_yticklabels():
        tl.set_color("g")
    for tl in ax4.get_yticklabels():
        tl.set_color("b")

    ax3.set_xlabel('Value of k for KNN')
    ax3.set_ylabel('The average of Power error rate')

    plt.savefig('..\Output\ForwardKnn Handwritten vs Sklearn.png')
    plt.show()




if __name__ =='__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # errors = [KnnAlpha(), KnnMu(), KnnAlphaSklearn(), KnnMuSklearn()]
    # df = pd.DataFrame(errors)
    # df.to_csv('InverseError.txt', sep=' ', index=False, header=False)

    # work space
    # 显示所有列
    pd.set_option('display.max_columns', None)
    # 显示所有行
    pd.set_option('display.max_rows', None)

    # trainKnn()
    # PlotResult1s()
    FindBestModelKnn()
