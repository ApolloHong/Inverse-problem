# coding : utf-8
# professor : Helin Gong
# author : Lizhan Hong

__author__ = 'Lizhan Hong'

import math
import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

from .FapyProSub import KNN, normalize, read

plt.rc('font', family='Times New Roman', size=15)
plt.style.use('seaborn')  # Set the theme of seaborn


def KnnForward():
    # 导入数据
    input_data = np.loadtxt('..\Input\Inpower5517.txt')
    input_label = np.loadtxt(r'..\Input\alpha5517.txt')

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
    data = np.loadtxt('..\Input\Inpower5517.txt')
    label = np.loadtxt(r'..\Input\alpha5517.txt')

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
    data = np.loadtxt('..\Input\Inpower5517.txt')
    label = np.loadtxt(r'..\Input\alpha5517.txt')
    q = np.loadtxt('..\Input\q5517.txt')

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
    input_data = np.loadtxt('..\Input\Inpower5517.txt')
    input_label = np.loadtxt(r'..\Input\alpha5517.txt')
    q = np.loadtxt('..\Input\q5517.txt')

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


def KnnWithoutK(Mu_test, nSample):
    # 导入数据
    input_data_mu = np.loadtxt('..\Input\Inpower5517.txt')
    input_data_mu = normalize(input_data_mu)
    input_data_alpha = np.loadtxt(r'..\Input\alpha5517.txt')

    # 统计样本数
    n_samples = input_data_mu.shape[0]
    n_test = math.ceil(n_samples / 4)
    n_train = n_samples - n_test

    # 划分数据集
    A = range(0, n_samples)
    A_test = random.sample(A, n_test)
    A_train = list(set(A) - set(A_test))
    mu_test = np.zeros((n_test, input_data_mu.shape[1]))
    mu_train = np.zeros((n_train, input_data_mu.shape[1]))
    alpha_test = np.zeros((n_test, input_data_alpha.shape[1]))
    alpha_train = np.zeros((n_train, input_data_alpha.shape[1]))

    for i in range(n_test):
        mu_test[i, :] = input_data_mu[A_test[i], :]
        alpha_test[i, :] = input_data_alpha[A_test[i], :]
    for i in range(n_train):
        mu_train[i, :] = input_data_mu[A_train[i], :]
        alpha_train[i, :] = input_data_alpha[A_train[i], :]
    # 训练模型
    k1 = 50
    KN_model_alpha = KNeighborsRegressor(n_neighbors=k1, weights='distance', p=1, metric='minkowski')
    KN_model_alpha.fit(mu_train, alpha_train)

    # 输入需要测试的数据
    test = Mu_test[nSample, :]

    # 预测
    test.shape = (1, test.size)
    predict_alpha = KN_model_alpha.predict(test)

    # 输出
    return predict_alpha


def PlotResult1s():
    data = read('..\Output\ForwardError.txt','txt')
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


    PlotResult1s()