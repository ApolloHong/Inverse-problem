# coding : utf-8
# professor : Helin Gong
# author : Lizhan Hong

__author__ = 'Lizhan Hong'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as spr

plt.rc('font', family='Times New Roman', size=1)
plt.style.use('seaborn')  # Set the theme of seaborn


def read(path:str, choice:str):
    '''
    Read txt or excel.

    :param path: the absolute path of the data.
    :param choice: txt or excel.
    :return: Inpower_raw(ndarray of the data).
    '''
    if choice == 'txt':
        data = pd.read_table(path, sep=' ', header=None)
        data.dropna(axis=1, how='any', inplace=True)
        return data
    if choice == 'excel':
        data = pd.read_excel(path, header=None)
        data.dropna(axis=1, how='any', inplace=True)
        return data


def reactorGetSectionIndex(pathControl:str):
    '''
    Generate the index of the one section data in a grid square.
    :param pathControl: The path of our one section data.
    :return:
    '''
    data = pd.read_excel(pathControl, header = None)
    lst = []
    # data.dropna(axis=1, how='any', inplace=True)
    for i in range(len(data.iloc[0])):
        for j in range(len(data.iloc[1])):
            if data.iloc[i, j] > 0:
                lst.append(i*len(data.iloc[0])+j)

    return lst


def normalize(data):  # 归一
    x = data.shape[0]
    y = data.shape[1]
    norm = np.zeros((x, y))
    for i in range(y):
        s = 0
        t = data[:, i]
        for j in range(x):
            s = s + t[j]
        norm[:, i] = data[:, i] / s
    return norm

def KNN(k, test, data1, label1):  # 预测结果
    distance = []
    for i in range(data1.shape[0]):  # 计算测试集和样本之间的距离
        distance.append(np.linalg.norm(test - data1[i, :]))
    d1 = np.argsort(distance)
    d = 0
    n = 0
    while (distance[d1[n]] == 0):
        n = n + 1
    for i in range(k):
        d = d + 1 / distance[d1[n + i]]
    predict = np.zeros((1, label1.shape[1]))
    for i in range(k):
        predict = predict + np.array(label1[d1[n + i], :]) / (distance[d1[n + i]] * d)  # 加权预测

    return predict

def ExtractTenSamples(pathInpower, pathAlpha, pathY, pathPower, LstSmecimen:list):
    '''
    Output 10 samples of mu and power data, also the whole data that we need.
    restore all of them by mu1 to mu10

    :param pathInpower:
    :param pathAlpha:
    :param pathY:
    :param LstSmecimen:
    :return:
    '''
    dfInpower = read(pathInpower,'txt')
    dfAlpha = read(pathAlpha,'txt')
    dfY = read(pathY,'txt')
    dfPower = read(pathPower,'txt')
    dfInpowerTen = dfInpower.iloc[LstSmecimen[0]]
    dfAlphaTen = dfAlpha.iloc[LstSmecimen[0]]
    dfYTen = dfY.iloc[LstSmecimen[0]]
    dfPowerTen = dfPower.iloc[LstSmecimen[0]]
    for i in range(1, len(LstSmecimen)):
        dfInpowerTen = np.vstack([dfInpowerTen, dfInpower.iloc[LstSmecimen[i]]])
        dfAlphaTen = np.vstack([dfAlphaTen, dfAlpha.iloc[LstSmecimen[i]]])
        dfYTen = np.vstack([dfYTen, dfY.iloc[LstSmecimen[i]]])
        dfPowerTen = np.vstack([dfPowerTen, dfPower.iloc[LstSmecimen[i]]])


    dfInpowerTen = pd.DataFrame(dfInpowerTen)
    dfInpowerTen.to_csv('..\Input\Inpower10.txt', sep=' ', index=False, header=False)
    dfAlphaTen = pd.DataFrame(dfAlphaTen)
    dfAlphaTen.to_csv(r'..\Input\alpha10.txt', sep=' ', index=False, header=False)
    dfYTen = pd.DataFrame(dfYTen)
    dfYTen.to_csv(r'..\Input\Y10.txt', sep=' ', index=False, header=False)
    dfPowerTen = pd.DataFrame(dfPowerTen)
    dfPowerTen.to_csv(r'..\Input\Power10.txt', sep=' ', index=False, header=False)


def PlotKnnOnlie(PredictData, TrueData, nSection):
    '''

    :param PredictData:
    :param TrueData:
    :param nSection:
    :return:
    '''
    plt.rc('font', family='Times New Roman', size=0.01)
    def GerData(data, nSection):
        '''
        Generate the nSection^th section in one state of reactor.

        :param data:
        :param nSection:
        :param nSample:
        :return:
        '''
        numLength = 15
        numWidth = 15
        numGrid = 177
        numSection = 28
        pathControl = "E:\\Apollo\\数据项目组\\数据\\index.xlsx"

        # The final data is in shape of (numGrid,1)
        # data = read(path, 'txt')
        data = [data[:,nSection + i * numSection] for i in range(numGrid)]
        data = np.array(data)
        lst = reactorGetSectionIndex(pathControl)

        # generate the kernel projection matrix onto the square manner base.
        def select():
            row = lst
            col = np.arange(numGrid)
            data = np.ones(numGrid)
            return spr.csc_matrix((data, (row, col)), shape=(numLength * numWidth, numGrid)).todense()

        maindata = select() @ data
        maindata = maindata.reshape((numLength, numWidth))

        return maindata

    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    ErrorData = [TrueData[i]-PredictData[i] for i in range(10)]

    maindata = [GerData(PredictData[_], nSection) for _ in range(10)]
    # truedata = [GerData(PredictData[_], nSection) for _ in range(6)]
    truedata = [GerData(TrueData[_], nSection) for _ in range(10)]
    error = [GerData(ErrorData[_], nSection)  for _ in range(10)]

    # plt.imshow(maindata, cmap='viridis')
    # plt.xticks(range(0, 15), letters[0:15])
    # plt.yticks(np.arange(0,15),range(1,16))
    # plt.grid(None)
    # plt.ylabel('y')
    # plt.xlabel('x')

    plt.rcParams['figure.figsize'] = [12,8]
    fig, axs = plt.subplots(5, 6)
    plt.subplots_adjust(left=0.1,bottom=0.1,right=0.9,top=0.9,wspace=0.2,hspace=0.4)


    for i in range(5):
        for j in range(2):
            axs[i, 0+j*3].imshow(maindata[i+j*5], cmap='viridis')
            axs[i, 0+j*3].set_title('PredictData')
            axs[i, 0+j*3].set_xticks(range(0, 15), letters[0:15])
            axs[i, 0+j*3].set_yticks(np.arange(0, 15), range(1, 16))
            axs[i, 0+j*3].grid(None)
            # plt.colorbar()
            # mappable = plt.collections[0]
            # plt.colorbar(mappable)

            axs[i, 1+j*3].imshow(truedata[i+j*5], cmap='viridis')
            axs[i, 1+j*3].set_title('TrueData')
            axs[i, 1+j*3].set_xticks(range(0, 15), letters[0:15])
            axs[i, 1+j*3].set_yticks(np.arange(0, 15), range(1, 16))
            axs[i, 1+j*3].grid(None)


            axs[i, 2+j*3].imshow(error[i+j*5],cmap='viridis')
            axs[i, 2+j*3].set_title('Error')
            axs[i, 2+j*3].set_xticks(range(0, 15), letters[0:15])
            axs[i, 2+j*3].set_yticks(np.arange(0, 15), range(1, 16))
            axs[i, 2+j*3].grid(None)
            # plt.colorbar()

    # fig.tight_layout()
    plt.show()


def PlotForK():
    data = read('..\Input\ForwardError.txt','txt')
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


def PlotInvK():
    data = read('..\Input\InverseError.txt','txt')

    fig = plt.figure(figsize=(15,7))
    plt.subplots_adjust(left=0.1,bottom=0.1,right=0.9,top=0.9,wspace=0.3,hspace=0.1)



    ax1 = fig.add_subplot(121)
    # ,lable='KnnMu',lable='KnnMuSklearn', lable='KnnMuSklearn',lable='KnnAlphaSklearn'
    # ax1.plot(np.arange(1,21),data.iloc[0],c='g' )
    # # ax1.scatter(data.iloc[0,0] , list(data.iloc[0,2].split())[data.iloc[0,0]] , 'r')
    # ax1.plot(np.arange(1,21),data.iloc[2],c='b' )
    # ax1.scatter(data.iloc[2,0] , data.iloc[2,2][data.iloc[2,0]] , 'r')
    line1, = ax1.plot(np.arange(1,21), data.iloc[0], 'g', alpha=0.5, lw=1.5, ls='-')
    ax1.scatter(np.arange(1, 21), data.iloc[0], color='g', alpha=0.5, lw=1.5, ls='-', marker='^')
    ax1.legend('I')
    ax2 = ax1.twinx()
    line2, = ax2.plot(np.arange(1,21), data.iloc[2], 'b', alpha=0.5, lw=1.5, ls='-', marker='+')
    ax2.scatter(np.arange(1, 21), data.iloc[2], color='b', alpha=0.5, lw=1.5, ls='-', marker='+')
    ax2.legend('I')
    plt.legend((line1, line2), ['KnnAlpha', 'KnnAlphaSklearn'], loc=0,
               title='Methods', ncol=1, markerfirst=False,
               numpoints=2, frameon=True, fancybox=True,
               facecolor='#d3d3da', edgecolor='b', shadow=True)
    ax2.set_xlabel('Value of k for KNN')

    # match the colors
    for tl in ax1.get_yticklabels():
        tl.set_color("g")
    for tl in ax2.get_yticklabels():
        tl.set_color("b")

    ax2.set_ylabel('The average of I error rate')




    ax3 = fig.add_subplot(122)
    # ax3.plot(np.arange(1,21),data.iloc[1],c='g')
    # # ax2.scatter(data.iloc[1,0] , data.iloc[1,2][data.iloc[1,0]] , 'r')
    # ax3.plot(np.arange(1,21),data.iloc[3],c='b')
    # # ax2.scatter(data.iloc[3,0] , data.iloc[3,2][data.iloc[3,0]] , 'r')
    line3, = ax3.plot(np.arange(1,21), data.iloc[1], 'g', alpha=0.5, lw=1.5, ls='-')
    ax3.scatter(np.arange(1, 21), data.iloc[1], color='g', alpha=0.5, lw=1.5, ls='-', marker='^')
    # ax3.legend('α')
    ax4 = ax3.twinx()
    line4, = ax4.plot(np.arange(1,21), data.iloc[3], 'b', alpha=0.5, lw=1.5, ls='-', marker='+')
    ax4.scatter(np.arange(1, 21), data.iloc[3], color='b', alpha=0.5, lw=1.5, ls='-', marker='+')
    # ax4.legend('α')
    plt.legend((line3, line4), ['KnnMu', 'KnnMuSklearn'], loc=0,
               title='Methods', ncol=1, markerfirst=False,
               numpoints=2, frameon=True, fancybox=True,
               facecolor='#d3d3da', edgecolor='b', shadow=True)

    # match the colors
    for tl in ax3.get_yticklabels():
        tl.set_color("g")
    for tl in ax4.get_yticklabels():
        tl.set_color("b")

    ax4.set_xlabel('Value of k for KNN')
    ax4.set_ylabel('The average of Alpha error rate')

    plt.savefig('..\Output\InverseKnn Handwritten vs Sklearn.png')
    plt.show()


def PlotEveryDetail(nSample:int):
    '''
    Plot the one column of a reactor state.

    :param nSample: The control number of sample we want (ranging from 0 to 10) over the 10 specimen.
    :return:
    '''

    # prepare the data
    pathControl = "E:\\Apollo\\数据项目组\\数据\\index.xlsx"
    path = '..\Input\Power10.txt'
    letters = ['A','B','C','D','E','F','G','H','I','J','K','L','M',
                'N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

    def reactorPlot3D(pathControl: str, path: str, nSample: int, numLength: int, numWidth: int, numGrid: int,
                      numSection: int):
        '''
        Plot the 3Dheatmap of the whole reactor.

        :param pathControl: The path of data we want to plot.
        :param path: The path of data we want to plot.
        :param nSample: The number of sample we want (ranging from 0 to 5516).
        :param numLength: The length of one section.
        :param numWidth: The width of one section.
        :param numGrid: The number of grids per section  (In our case, numGrid = 177).
        :param numSection: The number of vertical sections  (In our case, numSection = 28).
        :return:
        '''
        # The final data is in shape of (numGrid,numSection)
        data = read(path, 'txt')
        data = data.iloc[nSample, :]
        data = np.array(data).T
        lst = reactorGetSectionIndex(pathControl)

        def select():
            # first the righter loops(inner) then the lefter loops(outer)
            row = [i * numSection + k for i in lst for k in range(numSection)]
            col = np.arange(numGrid * numSection)
            data = np.ones(numGrid * numSection)
            return spr.csc_matrix((data, (row, col)),
                                  shape=(numLength * numWidth * numSection, numGrid * numSection)).todense()

        maindata = select() @ data
        maindata = maindata.T

        hist = np.zeros((numLength, numWidth, numSection))
        for j in range(numGrid):
            if lst[j] % numWidth != 0:
                [*hist[(lst[j] // numWidth), (lst[j] % numWidth), :]] = maindata[
                                                                        lst[j] * numSection:(lst[j] + 1) * numSection]
            else:
                [*hist[(lst[j] // numWidth), (lst[j] % numWidth), :]] = maindata[
                                                                        lst[j] * numSection:(lst[j] + 1) * numSection]

        return hist


    def reactorPlotCol(pathControl: str, path: str, nSample: int, nGrid: int, numLength: int, numWidth: int,
                       numGrid: int, numSection: int):
        '''
        Plot the one column of a reactor state.

        :param pathControl: The path of data we want to plot.
        :param path: The path of data we want to plot.
        :param nSample: The number of sample we want (ranging from 0 to 5516).
        :param numLength: The length of one section.
        :param numWidth: The width of one section.
        :param nGrid: The number of gird we want (ranging from 0 to 176).
        :param numGrid: The number of grids per section  (In our case, numGrid = 177).
        :param numSection: The number of vertical sections  (In our case, numSection = 28).
        :return: The graph of targeted section.
        '''

        data = read(path, 'txt')
        maindata = np.array([data.iloc[nSample, i + nGrid * numSection] for i in range(numSection)]).T
        lst = reactorGetSectionIndex(pathControl)

        hist = np.zeros((numLength, numWidth, numSection))
        if lst[nGrid] % numWidth != 0:
            [*hist[(lst[nGrid] // numWidth) + 1, (lst[nGrid] % numWidth), :]] = maindata
        else:
            [*hist[(lst[nGrid] // numWidth) + 1, (lst[nGrid] % numWidth), :]] = maindata

        return hist


    def reactorPlotSection(pathControl: str, path: str, nSample: int, nSection: int, numLength: int, numWidth: int,
                           numGrid: int, numSection: int):
        '''
        Plot the section we want.

        :param path: The path of data we want to plot.
        :param nSample: The number of sample we want (ranging from 0 to 5516).
        :param nSection: The number of section we want (ranging from 0 to 27).
        :param numLength: The length of one section.
        :param numWidth: The width of one section.
        :param numGrid: The number of grids per section  (In our case, numGrid = 177).
        :param numSection: The number of vertical sections  (In our case, numSection = 28).
        :return: The graph of targeted section.
        '''
        # The final data is in shape of (numGrid,1)
        data = read(path, 'txt')
        data = [data.iloc[nSample, nSection + i * numSection] for i in range(numGrid)]
        # print(data)
        data = np.array(data).T
        # print(data.shape)
        lst = reactorGetSectionIndex(pathControl)

        def select():
            row = lst
            col = np.arange(numGrid)
            data = np.ones(numGrid)
            return spr.csc_matrix((data, (row, col)), shape=(numLength * numWidth, numGrid)).todense()

        maindata = select() @ data
        maindata = maindata.reshape((numLength, numWidth))

        return maindata


    hist = reactorPlot3D(pathControl, path, nSample , 15, 15, 177, 28 )
    hist2 = reactorPlotCol(pathControl, path, nSample, 88, 15, 15, 177, 28 )
    maindata = reactorPlotSection(pathControl, path, nSample ,25 , 15, 15, 177, 28)

    fig = plt.figure(figsize=(15,7))
    plt.subplots_adjust(left=0.1,bottom=0.1,right=0.9,top=0.9,wspace=0.2,hspace=0.3)
    plt.tight_layout()

    # ===============
    #  First subplot
    # ===============
    # set up the axes for the first plot
    ax1 = fig.add_subplot(221, projection='3d')

    # make the color matrix
    viridis = plt.cm.get_cmap('viridis',8)
    colors1 = viridis( hist )
    ax1.voxels(hist, facecolors=colors1, alpha=0.8)
    # # Add colorbar
    # mappable = ax1.collections[0]
    # plt.colorbar(mappable)

    # Set labels
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')


    # ===============
    # Second subplot
    # ===============
    # set up the axes for the second plot
    ax2 = fig.add_subplot(222, projection='3d')
    # make the color matrix
    viridis = plt.cm.get_cmap('viridis',8)
    colors2 = viridis( hist2 )

    # and plot everything
    ax2.voxels(hist2, facecolors=colors2, alpha=0.8)

    # # Add colorbar
    # mappable = ax2.collections[0]
    # plt.colorbar(mappable)

    # Set labels
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    # ===============
    #  Third subplot
    # ===============
    # set up the axes for the third plot
    ax3 = fig.add_subplot(223)
    # # make the color matrix
    # hot_r = plt.cm.get_cmap('hot_r',8)
    # colors = hot_r( maindata/ 1.5 )

    # ax3.imshow(maindata, cmap = 'hot_r',vmin=0.3, vmax=0.8)
    ax3.imshow(maindata, cmap = 'viridis')
    plt.xticks(range(0,15),letters[0:15])
    plt.yticks(np.arange(0,15),range(1,16))
    ax3.grid(None)
    plt.ylabel('y')
    plt.xlabel('x')
    # # Add colorbar
    # mappable = ax3.collections[0]
    # plt.colorbar(mappable)

    # ===============
    #  Fourth subplot
    # ===============
    # set up the axes for the fourth plot
    ax4 = fig.add_subplot(224)
    # # make the color matrix
    # hot_r = plt.cm.get_cmap('hot_r',8)
    # colors = hot_r( maindata/ 1.5 )

    # ax4.imshow(maindata, cmap = 'hot_r',vmin=0.3, vmax=0.8)
    ax4.imshow(maindata, cmap = 'viridis')
    plt.xticks(range(0,15),letters[0:15])
    plt.yticks(np.arange(0,15),range(1,16))
    ax4.grid(None)
    plt.ylabel('y')
    plt.xlabel('x')
    # # Add colorbar
    # mappable = ax4.collections[0]
    # plt.colorbar(mappable)

    #plot everything
    plt.show()


if __name__ == '__main__':
    pass





