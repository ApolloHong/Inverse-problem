# coding : utf-8
# professor : Helin Gong
# author : Lizhan Hong

__author__ = 'Junyu Pan', 'Lizhan Hong'


import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

# from Read import read
from .ProSub import normalize, KNN, read

plt.rc('font', family='Times New Roman', size=15)
plt.style.use('seaborn')  # Set the theme of seaborn


def KnnAlpha():
    # 导入数据
    input_data = np.loadtxt('../Input/Y5517.txt')
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


def KnnMu():

    # 导入数据
    input_data = np.loadtxt('../Input/Y5517.txt')
    input_label = np.loadtxt('../Input/Inpower5517.txt')

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


def KnnMuSklearn():
    # 导入数据集
    data = np.loadtxt('../Input/Y5517.txt')
    label = np.loadtxt('../Input/Inpower5517.txt')

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

def KnnAlphaSklearn():
    # 导入数据集
    data = np.loadtxt('../Input/Y5517.txt')
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



# y-mu-alpha
def KnnInvWithK2s():
    # 导入数据
    input_data_Y = np.loadtxt('../Input/Y5517.txt')
    input_data_mu = np.loadtxt('../Input/Inpower5517.txt')
    input_data_mu = normalize(input_data_mu)
    input_data_alpha = np.loadtxt(r'../Input/alpha5517.txt')

    # 统计样本数
    n_samples = input_data_Y.shape[0]
    n_test = math.ceil(n_samples / 4)
    n_train = n_samples - n_test

    # 划分数据集
    A = range(0, n_samples)
    A_test = random.sample(A, n_test)
    A_train = list(set(A) - set(A_test))
    Y_test = np.zeros((n_test, input_data_Y.shape[1]))
    Y_train = np.zeros((n_train, input_data_Y.shape[1]))
    mu_test = np.zeros((n_test, input_data_mu.shape[1]))
    mu_train = np.zeros((n_train, input_data_mu.shape[1]))
    alpha_test = np.zeros((n_test, input_data_alpha.shape[1]))
    alpha_train = np.zeros((n_train, input_data_alpha.shape[1]))
    for i in range(n_test):
        Y_test[i, :] = input_data_Y[A_test[i], :]
        mu_test[i, :] = input_data_mu[A_test[i], :]
        alpha_test[i, :] = input_data_alpha[A_test[i], :]
    for i in range(n_train):
        Y_train[i, :] = input_data_Y[A_train[i], :]
        mu_train[i, :] = input_data_mu[A_train[i], :]
        alpha_train[i, :] = input_data_alpha[A_train[i], :]

    # 寻找最合适的k1，k2
    k1_appropriate = 0
    k2_appropriate = 0
    error_k1_k2 = 1
    Error_k1_k2 = np.zeros((20, 20))
    for k1 in range(1, 21):
        error_k1 = []
        for k2 in range(1, 21):
            # 构建预测模型
            KN_model_mu = KNeighborsRegressor(n_neighbors=k1, weights='distance', p=1, metric='minkowski')
            KN_model_mu.fit(Y_train, mu_train)
            KN_model_alpha = KNeighborsRegressor(n_neighbors=k2, weights='distance', p=1, metric='minkowski')
            KN_model_alpha.fit(mu_train, alpha_train)
            # 开始预测并计算错误率
            error = 0
            for i in range(n_test):
                test = Y_test[i, :]
                test.shape = (1, test.size)
                predict_mu = KN_model_mu.predict(test)
                predict_mu.shape = (1, predict_mu.size)
                predict_alpha = KN_model_alpha.predict(predict_mu)
                label = alpha_test[i, :]
                error = error + np.linalg.norm(predict_alpha - label) / np.linalg.norm(label)
            error = error / n_test
            error_k1.append(error)
            if (error < error_k1_k2):
                error_k1_k2 = error
                k1_appropriate = k1
                k2_appropriate = k2
        Error_k1_k2[k1-1, :] = error_k1
    print('k1_appropriate=', k1)
    print('k2_appropriate=', k2)
    print('error_k1_k2=', error_k1_k2)
    # print(Error_k1_k2)
    Error_k1_k2 = pd.DataFrame(Error_k1_k2)
    Error_k1_k2.to_csv('..\Output\Errork1k2.txt', sep = ' ', index=False, header=False)


def KnnInvWithoutK1s(y_test, nSample):
    # 导入数据
    input_data_Y = np.loadtxt('../Input/Y5517.txt')
    input_data_alpha = np.loadtxt(r'../Input/alpha5517.txt')

    # 统计样本数
    n_samples = input_data_Y.shape[0]
    n_test = math.ceil(n_samples / 4)
    n_train = n_samples - n_test

    # 划分数据集
    A = range(0, n_samples)
    A_test = random.sample(A, n_test)
    A_train = list(set(A) - set(A_test))
    Y_test = np.zeros((n_test, input_data_Y.shape[1]))
    Y_train = np.zeros((n_train, input_data_Y.shape[1]))
    alpha_test = np.zeros((n_test, input_data_alpha.shape[1]))
    alpha_train = np.zeros((n_train, input_data_alpha.shape[1]))
    for i in range(n_test):
        Y_test[i, :] = input_data_Y[A_test[i], :]
        alpha_test[i, :] = input_data_alpha[A_test[i], :]
    for i in range(n_train):
        Y_train[i, :] = input_data_Y[A_train[i], :]
        alpha_train[i, :] = input_data_alpha[A_train[i], :]

    # 训练模型
    k1 = 50
    KN_model_alpha = KNeighborsRegressor(n_neighbors=k1, weights='distance', p=1, metric='minkowski')
    KN_model_alpha.fit(Y_train, alpha_train)

    # 输入需要测试的数据
    test = y_test[nSample, :]

    # 预测
    test.shape = (1, test.size)
    predict_alpha = KN_model_alpha.predict(test)

    # 输出
    return predict_alpha


def KnnInvWithoutK2s(y_test, nSample):
    # 导入数据
    input_data_Y = np.loadtxt('../Input/Y5517.txt')
    input_data_mu = np.loadtxt('../Input/Inpower5517.txt')
    input_data_mu = normalize(input_data_mu)
    input_data_alpha = np.loadtxt(r'../Input/alpha5517.txt')

    # 统计样本数
    n_samples = input_data_Y.shape[0]
    n_test = math.ceil(n_samples / 4)
    n_train = n_samples - n_test

    # 划分数据集
    A = range(0, n_samples)
    A_test = random.sample(A, n_test)
    A_train = list(set(A) - set(A_test))
    Y_test = np.zeros((n_test, input_data_Y.shape[1]))
    Y_train = np.zeros((n_train, input_data_Y.shape[1]))
    mu_test = np.zeros((n_test, input_data_mu.shape[1]))
    mu_train = np.zeros((n_train, input_data_mu.shape[1]))
    alpha_test = np.zeros((n_test, input_data_alpha.shape[1]))
    alpha_train = np.zeros((n_train, input_data_alpha.shape[1]))
    for i in range(n_test):
        Y_test[i, :] = input_data_Y[A_test[i], :]
        mu_test[i, :] = input_data_mu[A_test[i], :]
        alpha_test[i, :] = input_data_alpha[A_test[i], :]
    for i in range(n_train):
        Y_train[i, :] = input_data_Y[A_train[i], :]
        mu_train[i, :] = input_data_mu[A_train[i], :]
        alpha_train[i, :] = input_data_alpha[A_train[i], :]

    # 训练模型
    k1 = 50
    KN_model_mu = KNeighborsRegressor(n_neighbors=k1, weights='distance', p=1, metric='minkowski')
    KN_model_mu.fit(Y_train, mu_train)
    KN_model_alpha = KNeighborsRegressor(n_neighbors=k1, weights='distance', p=1, metric='minkowski')
    KN_model_alpha.fit(mu_train, alpha_train)

    # 输入需要测试的数据
    test = y_test[nSample, :]

    # 预测
    test.shape = (1, test.size)
    predict_mu = KN_model_mu.predict(test)
    predict_mu.shape = (1, predict_mu.size)
    predict_alpha = KN_model_alpha.predict(predict_mu)

    # 输出
    return predict_alpha




if __name__ == '__main__':
    pass
    # errors = [KnnForward(), KnnForwardSklearn(), KnnForwardPower(), KnnForwardSklearnPower()]
    # df = pd.DataFrame(errors)
    # df.to_csv('ForwardError.txt', sep=' ', index=False, header=False)
    # PlotResult()
    # KnInvWithK2s()
