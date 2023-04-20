# coding : utf-8
# professor : Helin Gong
# author : Lizhan Hong

__author__ = 'Lizhan Hong'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as spr
from sklearn.utils.extmath import randomized_svd

from Read import read


def extractInpower(InpowerRaw, start:int, end:int):
    '''
    In our case, we take the 2,3,4,5th columns of the inpower matrix.

    :param InpowerRaw: The matrix we want to process.
    :param start: Start of useful column number.
    :param end: End of useful column number.
    :return: The Inpower matrix with useful columns.
    '''
    InpowerData = InpowerRaw.iloc[:, start:end].values
    return InpowerData


def contractIntoOneInFour(PowerRaw, numGrid:int, numSection:int, rangeUtile):
    '''

    :param PowerRaw: The Power matrix we obtained.
    :param numGrid: The number of grids per section  (In our case, numGrid = 177).
    :param numSection: The number of vertical sections  (In our case, numSection = 28).
    :param rangeUtile: The range of the chosen quarter section, where we can ignore the others by symetry.
    (In our case, rangeUtile = [[88,96],[103,111],[118,126],[132,139],[145,152],[157,163],[167,172],[174,177]])
    :return: The Power matrix simplified by symetry
    '''

    # create a list for storing index that indicate the useful ones. eventually length of 1428
    index = [(j * numSection + k) for k in range(numSection) for range_tuple in rangeUtile for j in
             range(range_tuple[0], range_tuple[1])]

    def generSelectMatrix4():
        '''
        Make the grid select matrix with size of (length of numGrid) * (numGrid)
        :return: sparce select matrix applied on one section.
        '''
        row = np.arange(len(index))
        col = np.array(index)
        data = np.ones(len(index))
        return spr.csc_matrix((data, (row, col)), shape=(len(index), len(PowerRaw.iloc[0]))).todense()


    return generSelectMatrix4() @ PowerRaw.T


# 困难，未完成。
def ReconstructToOrigin(numGrid:int, numSection:int, numGridQuarter:int,lstGridControl:list, rangeUtile):
    '''
    Reconstruct the quatered data back to the origin.

    :param PowerRaw: The Power matrix we obtained.
    :param numGrid: The number of grids per section  (In our case, numGrid = 177).
    :param numSection: The number of vertical sections  (In our case, numSection = 28).
    :param numGridQuarter: The number of the grid_column per row in a quarter section (In our case, numGridQuarter = 52).
    :param lstGridControl: The list for all the way we place the grid in a quarter (In our case,
    lstGridControl = [8,8,8,7,7,6,5,3] )
    :param rangeUtile: The range of the chosen quarter section, where we can ignore the others by symetry.
    (In our case, rangeUtile = [[88,96],[103,111],[118,126],[132,139],[145,152],[157,163],[167,172],[174,177]])
    :return: The Power matrix simplified by symetry
    '''
    Power2 = read('../Input/powerIAEA10000.txt', 'txt')
    Power3 = read('../Input/powerIAEA8480.txt', 'txt')

    def iter(PowerRaw):

        # find the symmetric and antisymmetric indexes.
        index1 = [j for range_tuple in rangeUtile for j in range(range_tuple[0], range_tuple[1])]
        # suitable for k2
        index2 = [j for range_tuple in rangeUtile[1:-1] for j in range(range_tuple[0]+1, range_tuple[1])]
        # indexS = [88*2 for _ in range(len(index)-1)] - index[1,:]

        # premierS = index1.index(rangeUtile[0][1]-1)
        premier = range(rangeUtile[0][0], rangeUtile[0][1])
        premierCol = [rangeUtile[_][0] for _ in range(len(rangeUtile))]
        lst2 = [lstGridControl[_]-1 for _ in range(len(lstGridControl))]

        def generSelectMat(data, row, col):
            '''
            Make the grid select matrix with size of (length of numGrid) * (numGrid)
            :return: sparce select matrix applied on one section.
            '''
            return spr.csc_matrix((data, (row, col)), shape=(numGrid , numGridQuarter)).todense()


        # symmetric
        def K1():
            row = index1
            col = [k for k in range(len(index1))]
            # index = [i for i in range(numGrid)]
            for j in range(1, len(index1)+1):
                row.append(88*2 - index1[j])
                col.append(j)
            return generSelectMat(np.ones(len(row)) ,row,col)

        # antisymmetric
        def K2():
            row = []
            col = []
            # symmetric with the axis horizontal.
            for t in range(len(index2)):
                n = index1.index(index2[t])
                bar = 0
                for k in range(1, len(lstGridControl)-1):
                    if sum(lstGridControl[0:k]) < n < sum(lstGridControl[0:k+1]):
                        bar = sum(lstGridControl[0:k])

                row.append(2*premier[n - bar] - index2[t])
                col.append(n)

            # symmetric with the axis vertical
            for t in range(len(index2)):
                n = index1.index(index2[t])
                bar = 0
                for k in range(1, len(lstGridControl)-1):
                    if sum(lstGridControl[0:k]) < n < sum(lstGridControl[0:k+1]):
                        m = k
                        # bar = sum(lstGridControl[0:k])

                row.append(2*premierCol[m] - index2[t])
                col.append(n)

            return generSelectMat(np.ones(len(row)), row,col)

        # selection matrix for one submatrix(section)
        Kl = K1() + K2()
        K = np.kron(Kl, np.eye(numSection))

        return K @ PowerRaw.T
        # print(K1())

    iter(Power2)


def POD(r:int, pathin, pathoutAlpha, pathoutQ):
    '''

    :param r: Number of singular values and vectors to extract.
    :param pathin: The path of powerdata
    :param pathoutAlpha: the alpha path we save
    :param pathoutQ: the basis path we save
    :return: The reduced order basis with r columns.
    '''
    def rsvd(PowerData, r:int):
        '''

        Reference:
        Finding structure with randomness: Stochastic algorithms for constructing approximate matrix decompositions
        (Algorithm 4.3) Halko, et al., 2009 https://arxiv.org/abs/0909.4061 .

        :param PowerData: The Power matrix simplified by symetry.
        :param r: Number of singular values and vectors to extract.
        :return: The reduced order basis with r columns.
        '''
        PowerData = PowerData.to_numpy()
        U, s, Vh = randomized_svd(PowerData, n_components=r, random_state=0)

        return Vh, np.matmul(U, np.diag(s))


    q, alpha = rsvd(read(pathin,'txt'), r)
    alpha = pd.DataFrame(alpha)
    q = pd.DataFrame(q)

    alpha.to_csv(pathoutAlpha, sep=' ', index=False, header=False)
    q.to_csv(pathoutQ, sep=' ', index=False, header=False)


def CompileData():
    dfInpower1 = read('../Input/inpower5517.out', 'txt')
    dfInpower2 = read('../Input/inpower18480.out', 'txt')

    dfPower1 = read('../Input/powerIAEA5517.txt', 'txt')
    dfPower2 = read('../Input/powerIAEA10000.txt', 'txt')
    dfPower3 = read('../Input/powerIAEA8480.txt', 'txt')

    dfInpower = pd.concat([dfInpower1, dfInpower2])
    dfPower = pd.concat([dfPower1, dfPower2])
    # print(dfPower.shape)
    dfPower = pd.concat([dfPower, dfPower3])
    dfInpower = dfInpower.iloc[:,1:5]

    # save to the csv file
    Inpower = pd.DataFrame(dfInpower)
    Power = pd.DataFrame(dfPower)
    Inpower.to_csv('..\Input\Inpower.txt', sep=' ', index=False, header=False)
    Power.to_csv('..\Input\Power.txt', sep=' ', index=False, header=False)


def GerSelectedMatrix(selectedGrid: list, selectedSection: list, numGridQuarter: int, numSection: int):
    '''
    Generate the data of sensors selector.

    :param selectedGrid: The grid that we measure. ( In our case,
    selectedGrid = [8,10,12,14,24,26,28,38,40,42,49,51]
    :param selectedSection: The sections that we choose (In our case, selectedSection = [1,5,9,13,17,21,25])
    :param numGridQuarter: The number of the grids in a quarter section (In our case, numGridQuarter = 52).
    :param numSection: The number of vertical sections  (In our case, numSection = 28).
    :return: The matrix we select.
    '''

    def grid():
        '''
        make the grid select matrix with size of (length of numGrid) * (numGrid)
        :return: sparce select matrix applied on one section.
        '''
        row = np.arange(len(selectedGrid))
        col = np.array(selectedGrid)
        data = np.ones(len(selectedGrid))
        return spr.csc_matrix((data, (row, col)), shape=(len(selectedGrid), numGridQuarter)).todense()

    def section():
        '''
        Make the section select matrix with size of (length of numGrid) * (numSection)
        :return: sparce select matrix applied on different sections.
        '''
        row = np.arange(len(selectedSection))
        col = np.array(selectedSection)
        data = np.ones(len(selectedSection))
        return spr.csc_matrix((data, (row, col)), shape=(len(selectedSection), numSection)).todense()

    def generSelectMat():
        '''
        Compose the two matrix by kronecker product.
        :return: The final select matrix.
        '''
        return np.kron(grid(), section())

    generSelectMat = pd.DataFrame(generSelectMat())
    generSelectMat.to_csv('..\Input\sensors.txt', sep=' ', index=False, header=False)


if __name__ == '__main__':
    # data space

    selectedGrid = [8, 10, 12, 14, 24, 26, 28, 38, 40, 42, 49, 51]
    selectedSection = [1, 5, 9, 13, 17, 21, 25]
    PowerData5517 = read('../Input/powerIAEA5517.txt', 'txt')
    numGridQuarter = 52
    numGrid = 177
    numSection = 28


    ## # test space

    # # test the shape
    # print(read('../Input/.txt', 'txt').shape)


    # inpower18480_4 = extractInpower(read('../Input/inpower18480.out','txt'),1,5)
    # inpower18480_4 = pd.DataFrame(inpower18480_4)
    # inpower18480_4.to_csv('..\Input\inpower18480_4.txt', sep=' ', index=False, header=False)
    # print(read('../Input/inpower18480_4.txt', 'txt').shape)


    # # print(read('..\Input\powerIAEA5517.txt','txt').shape)
    # # POD(50)
    # # compile the 18480 data
    # field1 = pd.read_csv('../Input/powerIAEA10000.txt', delimiter=' ', header=None, dtype=float)
    # field2 = pd.read_csv('../Input/powerIAEA8480.txt', delimiter=' ', header=None, dtype=float)
    # field = pd.concat([field1, field2], ignore_index=True)
    # field.to_csv('..\Input\powerIAEA18480.txt', sep=' ', index=False, header=False)
    #
    # # 5517
    # POD(r=50, pathin='..\Input\power5517.out', pathoutAlpha=r'..\Input\alpha5517.txt',
    #         pathoutQ='..\Input\q5517.txt')_
    # # 18480
    # POD(r=50, pathin='..\Input\powerIAEA18480.txt', pathoutAlpha=r'..\Input\powerIAEA18480coef.txt',
    #         pathoutQ='..\Input\powerIAEA18480basis.txt')


    # GerSelectedMatrix(selectedGrid, selectedSection, numGridQuarter, numSection)


