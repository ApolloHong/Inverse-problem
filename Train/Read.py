# coding : utf-8
# professor : Helin Gong
# author : Lizhan Hong

__author__ = 'Lizhan Hong'


import pandas as pd


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