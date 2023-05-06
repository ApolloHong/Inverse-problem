import random
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ProSub import *
import seaborn as sns
import math
import datetime
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from scipy.optimize import differential_evolution

def load_power18480(name='18480'):
    '''


    :rtype: object
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
    observations = pd.DataFrame(observations)
    observations.to_csv('../Input/Y18480.txt',sep=' ', header=False, index=False)
    return parameters, field, observations, sensors

def load_power18480_pod():
    '''

    :return:
    '''
    basis = pd.read_csv('../Input/powerIAEA18480basis.txt', delimiter=' ', header=None, dtype=float)  # shape:(n, M), n=150, M=52*28=1456
    # basis = qbasis.iloc[:dim, :]  # DIM, M
    coefficient = pd.read_csv('../Input/powerIAEA18480coef.txt', header=None, delimiter=' ', dtype=float)
    return basis, coefficient

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

def relativeL2Error(data, data_true):
    return np.linalg.norm((data - data_true), axis=-1) / np.linalg.norm(data_true)  # shape: n,1

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
    inputs = np.array(inputs)
    if inputs_dim == 1:
        inputs = inputs.reshape(1, -1)
    alphas = model.predict(inputs)  # shape: 18480, r
    return np.dot(alphas, basis)  # shape: n, 1456

def get_bounds_from_anchors(ys, offsets, limits):
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
    observation = np.dot(field, sensors.T) # (18480, 84)
    return field_error_L2(observation, observation_true)

def initialize_pop_uni(popsize, bounds, x0 = None):
    '''
    The `initilize_pop_uni` function implement a function to initialize the population of the optimizer.
    With the uniform distribution.

    :param popsize: `popsize`: an integer that represents the size of the population.
    :param bounds: `bounds`: a list of pairs of float numbers that represent the bounds for each parameter of the optimization problem.
    :param x0: a numpy array that represents the initial guess for the optimizer. If provided, it will be included in the population.
    :return:
    '''
    dim = len(bounds) # in our case, dim = 4
    min_b, max_b = np.asarray(bounds).T # shape:(dimensions,)
    pop = min_b + (max_b - min_b) * np.random.rand(popsize, dim)

    # # if we give an x, we fix the temperature.
    # if x is None:
    #     pop[-1] = x
    return pop

def initialize_pop_obl(popsize, bounds, x0 = None):
    '''
    The `initilize_pop_obl` function implement a function to initialize the population of the optimizer
    with method of Opposition-Based Learning  (OBL).

    :param popsize: `popsize`: an integer that represents the size of the population.
    :param bounds: `bounds`: a list of pairs of float numbers that represent the bounds for each parameter of the optimization problem.
    :param x0: a numpy array that represents the initial guess for the optimizer. If provided, it will be included in the population.
    :return:
    '''
    dim = len(bounds) # in our case, dim = 4
    min_b, max_b = np.asarray(bounds).T # shape:(dimensions,)

    popsize1 = int(np.where(popsize%2==0, popsize/2, (popsize+1)/2))
    popsize2 = int(np.where(popsize%2==0, popsize/2, (popsize-1)/2))
    pop1 = min_b + (x0 - min_b) * np.random.rand(popsize1, dim)
    pop2 = x0 + (max_b - x0) * np.random.rand(popsize2, dim)

    pop = np.concatenate((pop1,pop2), axis=0)

    # # if we give an x, we fix the temperature.
    # if x is None:
    #     pop[-1] = x
    return pop

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

def generate_anchors(bin=None):
    '''
    This function generates a dictionary of anchor values for each parameter based on the specified `bin` values.
    attention: only the 'bu' parameter is not linear.

    :param bin:  The `bin` dictionary specifies the number of bins to use for each parameter.
    :return:
    '''
    if bin is None:
        bin = {"bu": 10, "st": 20, "pw": 10, "tin": 10}
    anchors = {}
    anchors['st'] = np.arange(0 + 615 / (2 * bin['st']), 600, 615 / bin['st'])
    anchors['bu'] = np.array([0, 50, 100, 150, 200, 500, 1000, 1500, 2000, 2500])
    anchors['pw'] = np.arange(20 + 40 / bin['pw'], 100, 80 / bin['pw'])
    anchors['tin'] = np.arange(295 + 2.5 / bin['tin'], 300, 5 / bin['tin'])
    return anchors

def add_noise(data, sigma, noise:str=False):
    '''

    :param data:
    :param sigma:
    :param noise:
    :return:
    '''
    nsg = np.random.RandomState(42)

    if noise == 'normal':
        data_noise = data + nsg.normal(0, sigma / 100.0, data.shape) * data
    elif noise == 'uniform':
        data_noise = data + nsg.uniform(0, sigma / 100.0, data.shape) * data
    else:
        data_noise = data

    return data_noise

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

def discrete_parameters(parameters, param_name:str, bin=None):
    '''
    This function takes a dictionary of parameters and applies binning to each parameter based on the specified `bin` values.
    It returns a tuple containing the anchor dictionary and the updated parameter dictionary.

    :param parameters:
    :param bin:
    :return:
    '''

    if bin is None:
        bin = {"st": 20, "pw": 10, "tin": 10}
    anchors = generate_anchors(bin)
    parameters = pd.DataFrame(parameters)
    parameter_index = in2c(parameters, anchors[param_name])
    # parameters['st_c'], parameters['bu_c'], parameters['pw_c'], parameters["tin_c"] = \
    #     in2c(parameters['st'], anchors['st']), in2c(parameters['bu'], anchors['bu']), in2c(parameters['pw'], anchors['pw']), in2c(parameters['tin'], anchors['tin'])
    return parameter_index

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

def initial_guess_svc(knntrain_param, knntest_param, param_names, obs_train, obs_test,
                      confusion_plot:bool=False, n_components_svc:int=2):
    # create the results
    columns = ['st', 'pw', 'bu']
    index = ['f1_score']
    results = pd.DataFrame(np.zeros((len(index), len(columns))),
                           columns=columns,
                           index=index)
    models = {}
    # train the classifying model
    # the model we use is the best pipline in the best-para finding process(like
    # 'Pipeline(steps=[('pca', PCA(n_components=2)), ('ss', StandardScaler()), ('svc', SVC())])')
    model = Pipeline([('pca', PCA(n_components=n_components_svc)),
                      ('ss', StandardScaler()),
                      ('svc', SVC())])

    models = {}
    for _ in range(len(param_names[0:3])):
        mu_train_discrete = discrete_parameters(knntrain_param[:, _], param_names[_])
        mu_test_discrete = discrete_parameters(knntest_param[:, _], param_names[_])
        # reshape the mu_train_discrete with single feature
        mu_train_discrete1feature = mu_train_discrete.reshape(-1, 1)
        mu_test_discrete1feature = mu_test_discrete.reshape(-1, 1)
        # we use np.ravel to flatten the matrix
        model.fit(np.array(obs_train) , np.ravel(mu_train_discrete1feature))
        prd = model.predict( obs_test )
        results.iloc[0,_] = f1_score(mu_test_discrete1feature, prd, average='macro')
        models[param_names[_]] = model

        if confusion_plot:
            # plot the confusion matrix
            Mat_confusion = confusion_matrix(mu_test_discrete1feature, prd)
            plt.rc('font', family='Times New Roman', size=8)
            sns.heatmap(Mat_confusion, annot=True, fmt='.0f')
            plt.savefig('../Output/Confusion_Matrix_with' + str(n_components_svc) + 'eigval' + str(param_names[_] + '.png'))
            plt.show()

    joblib.dump(models,'../Input/SVC_models'+ str(n_components_svc) +'.pkl')
    results.to_csv('../Output/SVC_F1_Score_results'+ str(n_components_svc) +'.txt')


# def plot_cv_indices(cv, X, y, group, ax, n_splits, lw=10):
#     """Create a sample plot for indices of a cross-validation object."""
#
#     # Generate the training/testing visualizations for each CV split
#     for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=group)):
#         # Fill in indices with the training/test groups
#         indices = np.array([np.nan] * len(X))
#         indices[tt] = 1
#         indices[tr] = 0
#
#         # Visualize the results
#         ax.scatter(
#             range(len(indices)),
#             [ii + 0.5] * len(indices),
#             c=indices,
#             marker="_",
#             lw=lw,
#             cmap=cmap_cv,
#             vmin=-0.2,
#             vmax=1.2,
#         )
#
#     # Plot the data classes and groups at the end
#     ax.scatter(
#         range(len(X)), [ii + 1.5] * len(X), c=y, marker="_", lw=lw, cmap=cmap_data
#     )
#
#     ax.scatter(
#         range(len(X)), [ii + 2.5] * len(X), c=group, marker="_", lw=lw, cmap=cmap_data
#     )
#
#     # Formatting
#     yticklabels = list(range(n_splits)) + ["class", "group"]
#     ax.set(
#         yticks=np.arange(n_splits + 2) + 0.5,
#         yticklabels=yticklabels,
#         xlabel="Sample index",
#         ylabel="CV iteration",
#         ylim=[n_splits + 2.2, -0.2],
#         xlim=[0, 100],
#     )
#     ax.set_title("{}".format(type(cv).__name__), fontsize=15)
#     return ax

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
        '''
        now no use
        :param X:
        :return:
        '''
        y =np.array([e.decision_function(X) for e in self.predictors]).squeeze()
        return y


if __name__ == '__main__':
    pass

    # workspace
    # load_power18480()
    # PlotResultMyPSO()





