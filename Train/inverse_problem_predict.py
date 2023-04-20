# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import datetime
# import de
from pandas.plotting import scatter_matrix
import sklearn
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
import joblib
import os.path
import sys
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html
from scipy.optimize import rosen, differential_evolution
from ProSub import *

def load_power18480(name='18480'):
    '''


    :param name:
    :return:
    '''
    field1 = pd.read_csv('../Input/powerIAEA10000.txt', delimiter=' ', header=None, dtype=float)
    field2 = pd.read_csv('../Input/powerIAEA8480.txt', delimiter=' ', header=None, dtype=float)
    field = pd.concat([field1, field2], ignore_index=True)
    parameters = pd.read_csv('../Input/inpower18480.out', header=None, delimiter=' ', dtype=float).iloc[:, [1, 2, 3, 4]]
    parameters.rename(columns={1: 'st', 2: 'bu', 3: 'pw', 4: 'tin'}, inplace=True)
    sensors = pd.read_csv('../Input/sensors.txt', header=None, delimiter=' ', dtype=float)
    observations = np.dot(field, sensors.T)
    return parameters, field, observations, sensors


def load_power18480_pod(dim=40):
    '''

    :param dim: The number of modes.
    :return:
    '''
    qbasis = pd.read_csv('../Input/powerIAEA18480basis.txt', delimiter=' ', header=None, dtype=float)  # shape:(n, M), n=150, M=52*28=1456
    basis = qbasis.iloc[:dim, :]  # DIM, M
    coefficient = pd.read_csv('../Input/powerIAEA18480coef.txt', header=None, delimiter=' ', dtype=float)
    return basis, coefficient

def trainKnn(k = 4):
    # 导入数据
    input_data_mu = np.loadtxt('../Input/inpower18480_4.txt')
    # standerd = StandardScaler()
    # input_data_mu = standerd.fit(input_data_mu)
    input_data_mu = normalize(input_data_mu)
    input_data_alpha = np.loadtxt(r'../Input/powerIAEA18480coef.txt')


    # 划分数据集
    train_input, test_input, train_output, test_output = train_test_split(input_data_mu, input_data_alpha,
                                                                          test_size=0.25, random_state=42)

    # Train a KNN model on the data
    KNN_model_alpha = KNeighborsRegressor(n_neighbors=k, weights='distance', p=1, metric='minkowski')
    KNN_model_alpha.fit(train_input, train_output)

    # Save the model to a file using pickle
    with open('../Input/knn4.pkl', 'wb') as file:
        joblib.dump(KNN_model_alpha, file)
    # Save the test_input to a file using pickle
    with open('../Input/knntest_input.pkl', 'wb') as file:
        joblib.dump(test_input, file)
    # Save the test_output to a file using pickle
    with open('../Input/knntest_output.pkl', 'wb') as file:
        joblib.dump(test_output, file)



def field_error_L2(field, field_true):
    '''
    Calculates the L2 norm of the difference between the `field` and the `field_true` arrays,
    normalized by the L2 norm of the `field_true` array.
    This is a measure of the relative error between the two arrays.
    :param field:
    :param field_true:
    :return:
    '''
    return np.linalg.norm((field - field_true), axis=-1) / np.linalg.norm(field_true)  # shape: n,1


# 目前没用
def field_error_Linf(field, field_true):
    '''
    Calculates the Linf norm of the difference between the `field` and the `field_true` arrays,
    normalized by the Linf norm of the `field_true` array.
    This is a measure of the relative error between the two arrays.

    :param field:
    :param field_true:
    :return:
    '''
    return np.abs(field - field_true).max() / field_true.max()  # shape: n,1


def reconstruct_error_pod(coefficient, qbasis, original_field):
    '''
    Reconstructs a field from a set of POD (proper orthogonal decomposition) coefficients and a basis,
    and calculates the field reconstruction error using the `field_error` function.
    :param coefficient:
    :param qbasis:
    :param original_field:
    :return:
    '''
    n1 = coefficient.shape[1]
    top_qbasis = qbasis[:n1]
    field = np.dot(coefficient, top_qbasis)  # N, M
    return field_error_L2(field, original_field)


def reconstruct_field_by_inputs(inputs, model, basis):
    '''
    Uses a machine learning `model` to predict POD coefficients from `inputs`,
    then reconstructs the corresponding field using the `basis` and returns the result.
    :param inputs:
    :param model:
    :param basis:
    :return:
    '''
    inputs_dim = len(inputs.shape)
    if inputs_dim == 1:
        inputs = inputs.reshape(1, -1)
    alphas = model.predict(inputs)  # shape: n, 40
    return np.dot(alphas, basis)  # shape: n, 1456


def reconstruct_observation_error_from_input(inputs, model, basis, observation_true, sensors):
    '''

    :param inputs:
    :param model:
    :param basis:
    :param observation_true:
    :param sensors:
    :return:
    '''
    field = reconstruct_field_by_inputs(inputs, model, basis)
    observation = np.dot(field, sensors.T)
    return field_error_L2(observation, observation_true)


def reconstruct_compress_observation_error_from_input(inputs, model, basis, observation_true, sensors, compress):
    '''
    Reconstructs a field from `inputs` using `model` and `basis`,
    compresses the field using a specified compression method `compress`,
    and calculates the error between the compressed field and the true `observation_true` array.
    :param inputs:
    :param model:
    :param basis:
    :param observation_true:
    :param sensors:
    :param compress:
    :return:
    '''
    field = reconstruct_field_by_inputs(inputs, model, basis)
    return field_error_L2(compress.transform(field), observation_true)


def reconstruct_field_error_from_input(inputs, model, basis, filed_true):
    '''
    Reconstructs a field from `inputs` using `model` and `basis`,
    and calculates the error between the resulting field and the true `filed_true` array.
    :param inputs:
    :param model:
    :param basis:
    :param filed_true:
    :return:
    '''
    field = reconstruct_field_by_inputs(inputs, model, basis)
    return field_error_L2(field, filed_true)


def in2c(data, anchors):
    '''
    discretize the data into.
    Given an array `data` and an array of `anchors`,
    computes the index of the nearest anchor for each element of `data`.
    This can be used for quantization or clustering purposes.

    :param data:
    :param anchors:
    :return: for each data (parameter), return the index of the nearest anchor.
    '''
    data = data.values.repeat(anchors.size).reshape(-1, anchors.size)
    return np.abs(data - anchors).argmin(axis=1)


class MultiPredictor (BaseEstimator, TransformerMixin):
    '''
    This is a class that extends the `BaseEstimator` and `TransformerMixin` classes from Scikit-learn.
    It takes a list of predictor models in its constructor and implements the `fit`, `transform`, `predict`,
    and `decision_function` methods. `fit` returns the class instance, `transform` returns the input data unchanged,
    `predict` calls `predict` on each predictor model and returns the results as a 2D array
    where each row is a prediction for a single input,
    and `decision_function` calls `decision_function` on each predictor model and returns the results as a 1D array.
    '''
    def __init__(self, predictors):
        self.predictors = predictors

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X

    def predict(self, X):
        y =np.array([e.predict(X) for e in self.predictors])
        return y.T.squeeze()

    def decision_function(self, X):
        y =np.array([e.decision_function(X) for e in self.predictors]).squeeze()
        return y


def get_bounds(ys,  offsets, limits):
    '''
    This function takes three arguments: `ys`, `offsets`, and `limits`. `ys` is a 1D array of values,
    `offsets` is a 1D array of offsets (one for each value in `ys`),
    and `limits` is a 2D array of upper and lower bounds (one row for each value in `ys`).

    :param ys:  ( EX: ys = to_anchors(c, [anchors['st'], anchors['bu'], anchors['pw']]) )
    :param offsets:
    :param limits: The bound of the ( EX: limits = np.array([[0, 615], [0, 2500], [20, 100], [290, 300]]) )
    :return: It returns a list of tuples,
    where each tuple contains the lower and upper bounds for a single value in `ys` based on its corresponding offset and limit.
    '''
    # cs.shape:1*3, ms.shape:1*3, anchors.shape:1*3
    return [(max(l[0], y - o),  min(l[1], y + o)) for y, l, o in zip(ys, limits[0:3], offsets)]


def generate_anchors(bin={"bu":10, "st": 20, "pw": 10, "tin": 10}):
    '''
    This function generates a dictionary of anchor values for each parameter based on the specified `bin` values.

    :param bin:  The `bin` dictionary specifies the number of bins to use for each parameter.
    :return:
    '''
    anchors = {}
    anchors['st'] = np.arange(0 + 615 / (2 * bin['st']), 600, 615 / bin['st'])
    anchors['bu'] = np.array([0, 50, 100, 150, 200, 500, 1000, 1500, 2000, 2500])
    anchors['pw'] = np.arange(20 + 40 / bin['pw'], 100, 80 / bin['pw'])
    anchors['tin'] = np.arange(295 + 2.5 / bin['tin'], 300, 5 / bin['tin'])
    return anchors


def discrete_parameters(parameters, bin={"st": 20, "pw": 10, "tin": 10}):
    '''
    This function takes a dictionary of parameters and applies binning to each parameter based on the specified `bin` values.
    It returns a tuple containing the anchor dictionary and the updated parameter dictionary.

    :param parameters:
    :param bin:
    :return:
    '''

    anchors = generate_anchors(bin)
    parameters["st_c"], parameters['bu_c'], parameters['pw_c'], parameters["tin_c"] = \
        in2c(parameters['st'], anchors['st']), in2c(parameters['bu'], anchors['bu']), in2c(parameters['pw'], anchors['pw']), in2c(parameters['tin'], anchors['tin'])
    return anchors, parameters


def to_anchors(y, anchors):
    '''
    This function takes a 2D array of parameter values and an anchor dictionary
    and returns a 2D array of anchor values based on the parameter values.
    The anchor values for each parameter are determined by the corresponding anchor dictionary.
    :param y:
    :param anchors:
    :return:
    '''
    return np.array([a[c] for a, c in zip(anchors, y.T)]).T

# svc
def predict_parameter_from_observation(argv):
    '''
    This function takes command-line arguments as input and loads data and a model.
    It then generates a grid of hyperparameters, adds noise to the data, and trains SVC for each combination of hyperparameters.
    It saves the resulting performance metrics and models to disk.

    :param argv: the input data, which control the parameter range and step.
    :return:
    '''
    # get args
    if len(argv) > 1:
        nc_start = int(argv[1])
        nc_end = int(argv[2])
        nc_step = int(argv[3])
    else:
        nc_start = 30
        # nc_end = 100
        nc_end = 60
        nc_step = 2

    print((nc_start, nc_end, nc_step))
    # load data and model
    parameters, field, observations, sensors = load_power18480()
    basis, coefficient = load_power18480_pod(40)
    anchors = discrete_parameters(parameters)

    # sigmas = [0, 1, 2, 3, 4, 5]
    sigmas = [0]
    ncs = list(range(nc_start, nc_end, nc_step))
    index_sigmas = []
    for s in sigmas:
        index_sigmas += [s] * len(ncs)

    results = pd.DataFrame(np.zeros((len(sigmas) * len(ncs), 6)),
                           columns=[ 'bu_f1', 'st_f1','pw_f1', 'st_t', 'bu_t', 'pw_t'],
                           index=[index_sigmas, ncs*len(sigmas)])
    results.index.names = ['sigma', 'nc']

    models = {}
    for sigma in sigmas:
        print("sigma: " + str(sigma))
        # add noise to observation
        observations = observations + np.random.normal(0, sigma/100.0, observations.shape) * observations
        # split train, validate, test dataset
        ob_train_full, ob_test, in_train_full, in_test = train_test_split(observations, parameters, test_size=0.05, \
                                                                          random_state=42)
        ob_train, ob_val, in_train, in_val = train_test_split(ob_train_full, in_train_full, test_size=0.2, random_state=42)
        # train models
        for nc in ncs:
            print("nc: " + str(nc))
            for param in ['st', 'bu', 'pw']:
                print(param)
                model = Pipeline([('pca', PCA(n_components=nc)),
                                  ('ss', StandardScaler()),
                                  ('svc', SVC())])
                t1 = time.time()
                model.fit(ob_train, in_train[param + "_c"])
                prd = model.predict(ob_val)
                results.loc[sigma, nc][param + '_f1'] = f1_score(in_val[param + "_c"], prd, average='macro')
                results.loc[sigma, nc][param + '_t'] = time.time() - t1
                models[param + str(sigma) + '_' + str(nc)] = model

    now = datetime.datetime.now().strftime("%m%d") # get the date when it is administered.
    print(f'nc_start = {str(nc_start)},now = {now}')

    # In summary, while pd.to_pickle() is optimized for saving and loading Pandas data structures,
    # joblib.dump() is a more general-purpose serialization method that can handle a wider
    # range of Python objects.
    # pd.to_pickle(results, 'Input/' + str(nc_start) + 'inverse_problem' + now + '.pkl')
    # joblib.dump(models, 'Input/' + str(nc_start) + 'inverse_problem_models' + now + ".pkl")
    pd.to_pickle(results, '../Input/' + 'inverse_problem' + '.pkl')
    joblib.dump(models, '../Input/' + 'inverse_problem_models' + ".pkl")


def select_best_models(s, params, fname='..\Input\inverse_problem_best_models.pkl'):
    '''
    This function selects the best performing models from those generated by `predict_parameter_from_observation()`.
    It loads the saved performance metrics and models, selects the best performing hyperparameters for each sigma and parameter,
    and returns a list of the best models for the specified parameters and sigma values.
    :param s:
    :param params:
    :param fname:
    :return:
    '''
    if not os.path.isfile(fname):
        # models = joblib.load('result/5inverse_problem_models0505.pkl')
        # results = joblib.load('result/5inverse_problem0505.pkl')
        models = joblib.load('../Input/inverse_problem_models.pkl')
        results = joblib.load('../Input/inverse_problem.pkl')
        best_models = {}
        for s in [0, 1, 2, 3, 4, 5]:
            for p in ['st', 'bu', 'pw']:
                nc = results.loc[s][p + '_f1'].idxmax()
                print((s, p, nc, results.loc[s][p + '_f1'].max()))
                best_models[p + str(s)] = models[p + str(s) + '_' + str(nc)]
        joblib.dump(best_models, fname)

    best_models = joblib.load(fname)
    return [best_models[param + str(s)] for param in params]


class DeBest:
    '''
    This is a callable class that is used as a callback function for a scipy optimization routine.
    It stores the best solutions found by the optimization routine and the corresponding function values.
    '''
    def __init__(self, shape, fobjs):
        self.bestX = np.zeros(shape)
        self.converge = np.zeros(shape[0])
        self.len = shape[0]
        self.it = -1
        self.fobjs = fobjs
        self.errors = np.ones(shape=(shape[0], len(fobjs)), dtype=float) * -1

    def __call__(self, xk, convergence):
        '''
        There is an additional function `__call__(self, xk, convergence)` inside the `DeBest` class.
        This function updates the stored best solutions and function values during the optimization routine.
        :param xk:
        :param convergence:
        :return:
        '''
        self.it = self.it+1
        if self.it < self.len:
            self.bestX[self.it] = xk
            self.converge[self.it] = convergence
            for k in range(len(self.fobjs)):
                self.errors[self.it, k] = self.fobjs[k](xk)
        return False

    def trim(self):
        '''
        The `trim()` function can be used to remove unused entries from the stored solutions and function values.
        :return:
        '''
        self.bestX  = self.bestX[:self.it+1]
        self.converge = self.converge[:self.it+1]
        self.errors = self.errors[:self.it+1]
        return self


def de_parameter_from_observation_5cases(runs, prefix, record, initial=False):
    '''
    The function performs differential evolution optimization using the differential_evolution() function from the scipy.optimize module.
     It uses three objective functions: fobj for the observation error, f_field_L2_error for the field L2 error,
     and f_field_linf_error for the field Linf error. The optimization is performed for each observation in observations,
     and each observation is optimized len(runs) times with different random seeds.
     The initial parameter values for each observation can be provided by setting initial=True.
     The optimization progress can be recorded by setting record=True.

    :param runs:a list of integers representing the number of runs to perform for each observation
    :param prefix:a string prefix to use in saving the results of the optimization
    :param record:a boolean flag indicating whether or not to record the optimization progress
    :param initial:a boolean flag indicating whether or not to use the provided initial parameter values for each
    :return:
    + de_results: a Pandas DataFrame containing the optimization results, including the observation error,
    field error, number of function evaluations, number of iterations, time taken for optimization, and parameter values.
    + de_iteration: a dictionary containing the optimization progress for each observation and run,
    including the best parameter values found and whether or not the optimization converged.
    '''
    nsg = np.random.RandomState(42)
    mus_indx = [0, 9841, 12288, 12589, 13326]
    mu_initials = np.array([[20, 100, 61.11, 291.76],
                            [210, 600, 71.51, 293.54],
                            [400, 1000, 18.00, 293.54],
                            [500, 1000, 35.00, 297.54],
                            [120, 1700, 52.15, 300.00],
                            [130, 1480, 52.15, 300.00]])

    knn = joblib.load('../Input/knn4.pkl')

    limits = np.array([[0, 615], [0, 2500], [20, 100], [290, 300]])



    parameters, field, observations, sensors = load_power18480()

    basis, _ = load_power18480_pod(40)
    basis = basis.to_numpy()

    index_mus = []
    for s in mus_indx:
        index_mus += [s] * len(runs)
    columns = ['field_error', 'observation_error','linf_error',
                                 'f_field_error', 'f_observation_error','f_linf_error',
                                 'nfev', 'nit','de_time',
                                 'st', 'bu', 'pw', 'tin',
                                 'st0', 'bu0', 'pw0', 'tin0']
    de_results = pd.DataFrame(np.zeros((len(runs) * len(mus_indx), len(columns))),
                              columns= columns,
                              index=[index_mus, runs * len(mus_indx)])
    de_results.index.names = ['id', 'run']

    de_iteration = {}

    for i in mus_indx:
        obs = observations.loc[[i],:]
        field_i = field.iloc[i].to_numpy()
        # observation error
        fobj = lambda p: reconstruct_observation_error_from_input(p, knn, basis, obs, sensors)
        # filed L2 error
        f_field_L2_error = lambda p: reconstruct_field_error_from_input(p, knn, basis, field_i)
        f_field_linf_error = lambda p: np.abs(reconstruct_field_by_inputs(p, knn, basis) - field_i).max() / field_i.max()

        # intial region # 0, 9841, 12288, 12589, 13326
        if(initial):
            if (i == 0):
                mu_initial = mu_initials[0, :]
            elif (i == 9841):
                mu_initial = mu_initials[1, :]
            elif ( i == 12288):
                mu_initial = mu_initials[2, :]
            elif (i == 12589):
                mu_initial = mu_initials[3, :]
            else:
                mu_initial = mu_initials[4, :]
            limits = np.array([[mu_initial[0] - 20, mu_initial[0] + 20],
                               [mu_initial[1] - 100, mu_initial[1] + 100],
                               [mu_initial[2] - 30, mu_initial[2] + 30],
                               [mu_initial[3] - 10, mu_initial[3] + 10]])

        for s in runs:
            t1 = time.time()
            if record:
                bestX = DeBest((2000, 4),[fobj, f_field_L2_error, f_field_linf_error ] )
                result = differential_evolution(fobj, limits, seed=s, polish=False, maxiter=15, callback=bestX)
                de_results.loc[i, s]['de_time'] = time.time() - t1
                bestX.trim()
                de_iteration[(i, s, 'x')] = bestX.bestX
                de_iteration[(i, s, 'c')] = bestX.converge
                de_iteration[(i, s, 'e')] = bestX.errors
            else:
                result = differential_evolution(fobj, limits, seed=42, polish=False)
                de_results.loc[i, s]['de_time'] = time.time() - t1

            de_results.loc[i, s]['observation_error'] = result.fun
            de_results.loc[i, s][['pw', 'st', 'bu', 'tin']] = np.array(result.x)
            true_parameter = parameters.loc[i].to_numpy()
            de_results.loc[i, s][['pw0', 'st0', 'bu0', 'tin0']] = true_parameter
            de_results.loc[i, s]['f_field_error'] = f_field_L2_error(true_parameter)
            de_results.loc[i, s]['f_observation_error'] = fobj(true_parameter)
            de_results.loc[i, s]['f_linf_error'] = f_field_linf_error(true_parameter)
            de_results.loc[i, s][['nfev', 'nit']] = [result.nfev, result.nit]
            de_results.loc[i, s]['field_error'] = f_field_L2_error(result.x)
            de_results.loc[i, s]['linf_error'] = f_field_linf_error(result.x)
            print(i, s)

    exp_results = {}
    exp_results['sigmas'] = runs
    exp_results['num'] = num
    exp_results['record'] = record
    exp_results['de_results'] = de_results
    exp_results['de_iteration'] = de_iteration
    exp_results['cases'] = mus_indx
    now = datetime.datetime.now().strftime("%m%d")
    joblib.dump(exp_results, '../Input/'+prefix+'_'+now+'.pkl')
    return de_results

def de_parameter_from_observation(sigmas, num, prefix, noise, record, mus_indx=None):
    '''
    The `de_parameter_from_observation` function seems to implement a Differential Evolution algorithm for inverse problem solving.

    The function starts by loading some data from files and defining some utility functions.
    It then initializes a Differential Evolution optimizer and runs it for each combination of the input parameters.
    The optimization results are stored in a Pandas DataFrame, which is saved to a file at the end of the function.
    :param sigmas: a list of float numbers that represent the standard deviation of the noise to be added to the observations.
    :param num: an integer that represents the number of samples to be considered.
    :param prefix: a string that will be used as the prefix for the filename where the results will be saved.
    :param noise:a string that specifies the type of noise to be added to the observations. It can be 'normal',
                  'uniform', or any other string to denote no noise.
    :param record: a boolean that indicates whether to record the intermediate results or not.
    :param mus_indx:
    :return:
    '''
    if not mus_indx:
        nsg = np.random.RandomState(42)
        mus_indx = list(nsg.choice(list(range(0,10000)), size=num))
    np.random.seed(0)

    knn = joblib.load('../Input/knn4.pkl')
    limits = np.array([[0, 615], [0, 2500], [20, 100], [290, 300]])

    parameters, field, observations, sensors = load_power18480()

    basis, _ = load_power18480_pod(40)
    basis = basis.to_numpy()

    index_sigmas = []
    for s in sigmas:
        index_sigmas += [s] * len(mus_indx)
    columns = ['field_error', 'observation_error','linf_error',
                                 'f_field_error', 'f_observation_error','f_linf_error',
                                 'nfev', 'nit','de_time',
                                 'st', 'bu', 'pw', 'tin',
                                 'st0', 'bu0', 'pw0', 'tin0']
    de_results = pd.DataFrame(np.zeros((len(sigmas) * len(mus_indx), len(columns))),
                        columns= columns,
                        index=[index_sigmas, mus_indx*len(sigmas)])
    de_results.index.names = ['sigma', 'id']

    de_iteration = {}

    for s in sigmas:
        if noise == 'normal':
            fileld_noise = field + nsg.normal(0, s / 100.0, field.shape) * field
        elif noise == 'uniform':
            fileld_noise = field + nsg.uniform(0, s / 100.0, field.shape) * field
        else:
            fileld_noise = field

        observations = np.dot(fileld_noise, sensors.T)

        for i in mus_indx:
            obs = observations[[i]]
            field_i = field.iloc[i].to_numpy()
            # observation error
            fobj = lambda p: reconstruct_observation_error_from_input(p, knn, basis, obs, sensors)
            # filed L2 error
            f_field_L2_error = lambda p: reconstruct_field_error_from_input(p, knn, basis, field_i)
            f_field_linf_error = lambda p: np.abs(reconstruct_field_by_inputs(p, knn, basis) - field_i).max() / field_i.max()
            # f_field_linf_error = field_error_Linf(reconstruct_field_by_inputs(p, knn, basis), field_i)

            t1 = time.time()
            if record:
                bestX = DeBest((2000, 4),[fobj, f_field_L2_error, f_field_linf_error ] )
                result = differential_evolution(fobj, limits, polish=False, maxiter=16, callback=bestX)
                de_results.loc[s, i]['de_time'] = time.time() - t1
                bestX.trim()
                de_iteration[(s, i, 'x')] = bestX.bestX
                de_iteration[(s, i, 'c')] = bestX.converge
                de_iteration[(s, i, 'e')] = bestX.errors

            else:
                result = differential_evolution(fobj, limits, polish=False, maxiter=16)
                de_results.loc[s, i]['de_time'] = time.time() - t1

            de_results.loc[s, i]['observation_error'] = result.fun
            de_results.loc[s, i][['pw', 'st', 'bu', 'tin']] = np.array(result.x)
            true_parameter = parameters.loc[i].to_numpy()
            de_results.loc[s, i][['pw0', 'st0', 'bu0', 'tin0']] = true_parameter
            de_results.loc[s, i]['f_field_error'] = f_field_L2_error(true_parameter)
            de_results.loc[s, i]['f_observation_error'] = fobj(true_parameter)
            de_results.loc[s, i]['f_linf_error'] = f_field_linf_error(true_parameter)
            de_results.loc[s,i][['nfev', 'nit']] = [result.nfev, result.nit]
            de_results.loc[s, i]['field_error'] = f_field_L2_error(result.x)
            #y = restruct_filed_by_inputs(result.x, knn, basis)
            #de_results.loc[s, i]['linf_error'] = np.abs(y - field.iloc[i].to_numpy()).max() / np.abs(field.iloc[i].to_numpy()).max()
            de_results.loc[s, i]['linf_error'] = f_field_linf_error(result.x)
            print(s, i)

    exp_results = {}
    exp_results['sigmas'] = sigmas
    exp_results['num'] = num
    exp_results['noise'] = noise
    exp_results['record'] = record
    exp_results['de_results'] = de_results
    exp_results['de_iteration'] = de_iteration
    exp_results['cases'] = mus_indx
    now = datetime.datetime.now().strftime("%m%d")
    joblib.dump(exp_results, '../Input/'+prefix+'_'+now+'.pkl')
    return de_results


def initialize_pop(popsize, bounds, x = None):
    '''
    The `initilize_pop` function implement a function to initialize the population of the optimizer. It takes in three parameters:

    :param popsize: `popsize`: an integer that represents the size of the population.
    :param bounds: `bounds`: a list of pairs of float numbers that represent the bounds for each parameter of the optimization problem.
    :param x: `x`: a numpy array that represents the initial guess for the optimizer. If provided, it will be included in the population.
    :return:
    '''
    dimensions = len(bounds) # in our case, dimensions = 4
    min_b, max_b = np.asarray(bounds).T # shape:(dimensions,)
    diff = max_b - min_b
    pop = min_b + diff * np.random.rand(popsize, dimensions)

    # if we give a x, we fix the temperature.
    if x is None:
        pop[-1] = x
    return pop


def de_predict_parameter_from_observation(sigmas, num, prefix, noise, record):
    '''
    The `de_predict_parameter_from_observation` function seems to be a wrapper function for the `de_parameter_from_observation` function, with some additional functionalities.
    It loads the results from the files generated by the `de_parameter_from_observation` function, and it performs some analysis and visualization of the results.

    The function loads some preprocessed data and models from other Python modules,
    and initializes a Pandas DataFrame to store the results of the parameter estimation algorithm.
    The function loops through each value in `sigmas` and for each sigma value,
    it loops through `num` field points to perform parameter estimation using differential evolution algorithm.

    For each field point, the function generates noisy observation data by adding noise to the corresponding field data.
     It then constructs a lambda function for the objective function to be minimized by the differential evolution algorithm. The objective function evaluates the reconstruction error between the predicted observation data and the actual observation data.

    The function then constructs other lambda functions to evaluate the reconstruction error of the predicted field data,
    as well as the L-infinity error between the predicted and actual field data.

    The function then generates an initial population of candidate solutions,
    and runs the differential evolution algorithm with the objective function
    and the initial population. If `record` is `True`, the function records the convergence history of the algorithm,
    which includes the best candidate solution and the corresponding objective function value at each iteration.

    The function then stores the resulting parameter estimates and reconstruction errors in the Pandas DataFrame.
    Finally, the function returns the DataFrame,
    which contains the results of the parameter estimation algorithm for all sigma values and field points.
    :param sigmas: is a list of values representing the standard deviation of the noise to be added to the field data.
    :param num: is an integer representing the number of field points to be used in the analysis.
    :param prefix: is a string that can be used to name the output files.
    :param noise: is a string that specifies the type of noise to be added to the field data.
    :param record: is a boolean value indicating whether to record the convergence history of the differential evolution algorithm.
    :return:
    '''
    nsg = np.random.RandomState(42)
    mus_indx = list(nsg.choice(list(range(0,10000)), size=num))
    # mus_indx = list(nsg.choice(10000, size=num))

    knn = joblib.load('../Input/knn4.pkl')
    limits = np.array([[0, 615], [0, 2500], [20, 100], [290, 300]])

    parameters, field, observations, sensors = load_power18480()

    basis, _ = load_power18480_pod(40)
    basis = basis.to_numpy()

    # sigma is the std of the noise.
    index_sigmas = []
    for s in sigmas:
        index_sigmas += [s] * len(mus_indx)
    columns = ['field_error', 'observation_error','linf_error',
                               'f_field_error', 'f_observation_error','f_linf_error',
                               'p_field_error', 'p_observation_error', 'p_linf_error',
                               'nfev', 'nit','de_time',
                                'st', 'bu', 'pw','tin',
                              'x0st', 'x0bu', 'x0pw',  'x0tin',
                              'st0', 'bu0', 'pw0',  'tin0']
    de_results = pd.DataFrame(np.zeros((len(sigmas) * len(mus_indx), len(columns))),
                        columns= columns,
                        index=[index_sigmas, mus_indx*len(sigmas)])
    de_results.index.names = ['sigma', 'id']

    de_iteration = {}

    # the reason why we use the noise is that, we want to simulate the noise get in the idustry.
    for s in sigmas:
        if noise == 'normal':
            fileld_noise = field + nsg.normal(0, s / 100.0, field.shape) * field
        else:
            fileld_noise = field + nsg.uniform(0, s / 100.0, field.shape) * field

        observations = np.dot(fileld_noise, sensors.T)

        models = select_best_models(s, ['st', 'bu', 'pw'])
        obs2in = MultiPredictor(models)
        anchors = generate_anchors()

        for i in mus_indx:
            obs = observations[[i]]
            field_i = field.iloc[i].to_numpy()
            # observation error
            fobj = lambda p: reconstruct_observation_error_from_input(p, knn, basis, obs, sensors)
            # filed L2 error
            f_field_L2_error = lambda p: reconstruct_field_error_from_input(p, knn, basis, field_i)
            f_field_linf_error = lambda p: np.abs(reconstruct_field_by_inputs(p, knn, basis) - field_i).max() / field_i.max()

            t1 = time.time()

            c = obs2in.predict(obs)
            y = to_anchors(c, [anchors['st'], anchors['bu'], anchors['pw']])
            x0 = np.concatenate((y, [295])) # the last one is the temperature fixed
            bounds = get_bounds(y, offsets=[20, 100, 15], limits=limits) + [(290, 300)]

            # generate initial population
            pop0 = initialize_pop(60, bounds, x=x0)

            if record:
                bestX = DeBest((2000, 4),[fobj, f_field_L2_error, f_field_linf_error ] )
                result = differential_evolution(fobj, limits, seed=42, polish=False, maxiter=10, callback=bestX, x0=x0, init=pop0)
                de_results.loc[s, i]['de_time'] = time.time() - t1
                bestX.trim()
                de_iteration[(s, i, 'x')] = bestX.bestX
                de_iteration[(s, i, 'c')] = bestX.converge
                de_iteration[(s, i, 'e')] = bestX.errors
            else:
                result = differential_evolution(fobj, limits, seed=42, polish=False)
                de_results.loc[s, i]['de_time'] = time.time() - t1

            de_results.loc[s, i]['observation_error'] = result.fun
            de_results.loc[s, i][['st', 'bu', 'pw', 'tin']] = np.array(result.x)
            true_parameter = parameters.loc[i].to_numpy()
            de_results.loc[s, i][['st0', 'bu0','pw0',  'tin0']] = true_parameter
            de_results.loc[s, i][['x0st', 'x0bu', 'x0pw', 'x0tin']] = x0
            de_results.loc[s, i]['f_field_error'] = f_field_L2_error(true_parameter)
            de_results.loc[s, i]['f_observation_error'] = fobj(true_parameter)
            de_results.loc[s, i]['f_linf_error'] = f_field_linf_error(true_parameter)
            de_results.loc[s, i]['p_field_error'] = f_field_L2_error(x0)
            de_results.loc[s, i]['p_observation_error'] = fobj(x0)
            de_results.loc[s, i]['p_linf_error'] = f_field_linf_error(x0)
            de_results.loc[s,i][['nfev', 'nit']] = [result.nfev, result.nit]
            de_results.loc[s, i]['field_error'] = f_field_L2_error(result.x)
            #y = restruct_filed_by_inputs(result.x, knn, basis)
            #de_results.loc[s, i]['linf_error'] = np.abs(y - field.iloc[i].to_numpy()).max() / np.abs(field.iloc[i].to_numpy()).max()
            de_results.loc[s, i]['linf_error'] = f_field_linf_error(result.x)
            print(s, i)

    exp_results = {}
    exp_results['sigmas'] = sigmas
    exp_results['num'] = num
    exp_results['noise'] = noise
    exp_results['record'] = record
    exp_results['de_results'] = de_results
    exp_results['de_iteration'] = de_iteration
    exp_results['cases'] = mus_indx
    now = datetime.datetime.now().strftime("%m%d")
    joblib.dump(exp_results, '../Input/'+prefix+'_'+now+'.pkl')
    return de_results

if __name__=="__main__":

    # # data space
    if len(sys.argv) > 1:
        ss = int(sys.argv[1])
        se = int(sys.argv[2])
        num = int(sys.argv[3])
        rand = sys.argv[4]
        record = bool(int(sys.argv[5]))
        prefix = sys.argv[6]
    else:
        ss = 0
        se = 1
        num = 2
        prefix = 'test'
        record = True
        rand = 'uniform'


    # # test space
    # print(sklearn.__version__)
    # parameters, field, observations, sensors = load_power18480()
    # print(parameters.shape, field.shape, observations.shape, sensors.shape)


    # knn = joblib.load('../Input/knn4.pkl')
    # print(knn.predict(np.array([1,2,3,4]).reshape((-1,1))))


    # # work space

    # trainKnn()
    # predict_parameter_from_observation([0])
    # data1 = joblib.load('../Input/inverse_problem.pkl')
    # data2 = joblib.load('../Input/inverse_problem_models.pkl')
    # print('data1',data1,'\n', 'data2', data2)



    result = de_predict_parameter_from_observation(list(range(ss, se)), num, prefix, rand, record)
    de_parameter_from_observation(list(range(ss, se)), num, prefix, rand, record)
    de_parameter_from_observation_5cases(list(range(1, 50)), 'clean5caseDE', 1)
    de_parameter_from_observation(list(range(0, 50)), -1, 'clean5caseDE1', 'clean', record, mus_indx=[0, 9841, 12288, 12589, 13326])
    de_parameter_from_observation_5cases(list(range(1, 50)), 'clean5caseDEInitial', 1, True)



