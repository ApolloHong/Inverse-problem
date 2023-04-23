import random
import joblib
import numpy as np
import pandas as pd
from ProSub import *
import math
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from scipy.optimize import differential_evolution


# Define the fitness function to be optimized
def fitness_sphere(x):
    return np.sum(x ** 2)

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

    # if we give an x, we fix the temperature.
    if x is None:
        pop[-1] = x
    return pop


# Define the de_rand_1_bin operator to combine two solutions and create a new one
def de_rand_1_bin_operator(x1, x2, x3, F):
    '''

    :param x1: One target vector (randomly chosen ) in one population
    :param x2: One target vector (randomly chosen ) in one population
    :param x3: One target vector (randomly chosen ) in one population
    :param F: The scale factor F ∈ [0,2] and F ∈ [0.4,1.0] on the sphere fitness function
    :return:
    '''
    return x1 + F * (x2 - x3)

# Define the de_rand_2_bin operator to combine two solutions and create a new one
def de_rand_2_bin_operator(x1, x2, x3, x4, x5, F):
    '''

    :param x1: One target vector (randomly chosen ) in one population
    :param x2: One target vector (randomly chosen ) in one population
    :param x3: One target vector (randomly chosen ) in one population
    :param x4: One target vector (randomly chosen ) in one population
    :param x5: One target vector (randomly chosen ) in one population
    :param F: The scale factor F ∈ [0,2] and F ∈ [0.4,1.0] on the sphere fitness function
    :return:
    '''
    return x1 + F * (x2 + x4 - x3 - x5)

# Define the de_best_1_bin operator to combine two solutions and create a new one
def de_best_1_bin_operator(x1, x2, xbest, F):
    '''

    :param x1: One target vector (randomly chosen ) in one population
    :param x2: One target vector (randomly chosen ) in one population
    :param xbest: The best target vector (randomly chosen ) in one population
    :param F: The scale factor F ∈ [0,2] and F ∈ [0.4,1.0] on the sphere fitness function
    :return:
    '''
    return xbest + F * (x1 - x2)

# Define the de_best_2_bin operator to combine two solutions and create a new one
def de_best_2_bin_operator(x1, x2, x3, x4, xbest, F):
    '''

    :param x1: One target vector (randomly chosen ) in one population
    :param x2: One target vector (randomly chosen ) in one population
    :param x3: One target vector (randomly chosen ) in one population
    :param x4: One target vector (randomly chosen ) in one population
    :param xbest: The best target vector (randomly chosen ) in one population
    :param F: The scale factor F ∈ [0,2] and F ∈ [0.4,1.0] on the sphere fitness function
    :return:
    '''
    return xbest + F * (x1 + x2 - x3 - x4)

# Define the de_target_to_best_bin operator to combine two solutions and create a new one
def de_target_to_best_bin_operator(xi, x1, x2, xbest, F):
    '''

    :param xi One target vector (randomly chosen ) in one population
    :param x1: One target vector (randomly chosen ) in one population
    :param x2: One target vector (randomly chosen ) in one population
    :param xbest: The best target vector (randomly chosen ) in one population
    :param F: The scale factor F ∈ [0,2] and F ∈ [0.4,1.0] on the sphere fitness function
    :return:
    '''
    return xi + F * (x1 + xbest - xi - x2)

class Config:
    __PopulationSize = 50 # Population Size
    __MaxDomain = 500 # variable upper limit
    __MinDomain = -500 # variable lower limit
    __Lambda = 1.5 # parameter for Levy flight
    __Pa = 0.25
    __Step_Size = 0.01
    __Dimension = 10 # The number of dimension
    __Trial = 31
    __Iteration = 3000

    @classmethod
    def get_population_size(cls):
        return cls.__PopulationSize

    @classmethod
    def get_Pa(cls):
        return cls.__Pa

    @classmethod
    def get_iteration(cls):
        return cls.__Iteration

    @classmethod
    def get_trial(cls):
        return cls.__Trial

    @classmethod
    def get_dimension(cls):
        return cls.__Dimension

    @classmethod
    def get_max_domain(cls):
        return cls.__MaxDomain

    @classmethod
    def set_max_domain(cls, _max_domain):
        cls.__MaxDomain = _max_domain

    @classmethod
    def get_min_domain(cls):
        return cls.__MinDomain

    @classmethod
    def set_min_domain(cls, _min_domain):
        cls.__MinDomain = _min_domain

    @classmethod
    def get_lambda(cls):
        return cls.__Lambda

    @classmethod
    def set_lambda(cls, _lambda):
        cls.__Lambda = _lambda

    @classmethod
    def get_stepsize(cls):
        return cls.__Step_Size

def plot_cv_indices(cv, X, y, group, ax, n_splits, lw=10):
    """Create a sample plot for indices of a cross-validation object."""

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=group)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(
            range(len(indices)),
            [ii + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            cmap=cmap_cv,
            vmin=-0.2,
            vmax=1.2,
        )

    # Plot the data classes and groups at the end
    ax.scatter(
        range(len(X)), [ii + 1.5] * len(X), c=y, marker="_", lw=lw, cmap=cmap_data
    )

    ax.scatter(
        range(len(X)), [ii + 2.5] * len(X), c=group, marker="_", lw=lw, cmap=cmap_data
    )

    # Formatting
    yticklabels = list(range(n_splits)) + ["class", "group"]
    ax.set(
        yticks=np.arange(n_splits + 2) + 0.5,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylabel="CV iteration",
        ylim=[n_splits + 2.2, -0.2],
        xlim=[0, 100],
    )
    ax.set_title("{}".format(type(cv).__name__), fontsize=15)
    return ax


def levy_flight(Lambda):
    cf = Config()
    #generate step from levy distribution
    sigma1 = np.power((math.gamma(1 + Lambda) * np.sin((np.pi * Lambda) / 2)) \
                      / math.gamma((1 + Lambda) / 2) * np.power(2, (Lambda - 1) / 2), 1 / Lambda)
    sigma2 = 1
    u = np.random.normal(0, sigma1, size=cf.get_dimension())
    v = np.random.normal(0, sigma2, size=cf.get_dimension())
    step = u / np.power(np.fabs(v), 1 / Lambda)

    return step    # return np.array (ex. [ 1.37861233 -1.49481199  1.38124823])


# Define the CS operator to generate a new solution using Lévy flight
def cs_operator(x, alpha, lambda_, lower_bound, upper_bound):
    levy = lambda_ * np.random.standard_cauchy(len(x))
    new_x = x + alpha * levy / (abs(levy) ** (1 / 2))
    return np.clip(new_x, lower_bound, upper_bound)


class particle:
    def __init__(self):
        self.pos = 0  # 粒子当前位置
        self.speed = 0
        self.pbest = 0  # 粒子历史最好位置


class ApolloidPSO:
    '''
    our pso
    '''
    def __init__(self):
        self.w = 0.5  # 惯性因子
        self.c1 = 1  # 自我认知学习因子
        self.c2 = 1  # 社会认知学习因子
        self.gbest = 0  # 种群当前最好位置
        self.N = 20  # 种群中粒子数量
        self.POP = []  # 种群
        self.iter_N = 100  # 迭代次数

    # 适应度值计算函数
    def fitness(self, x):
        return x + 16 * np.sin(5 * x) + 10 * np.cos(4 * x)

    # 找到全局最优解
    def g_best(self, pop):
        for bird in pop:
            if bird.fitness > self.fitness(self.gbest):
                self.gbest = bird.pos

    # 初始化种群
    def initPopulation(self, pop, N):
        for i in range(N):
            bird = particle()#初始化鸟
            bird.pos = np.random.uniform(-10, 10)#均匀分布
            bird.fitness = self.fitness(bird.pos)
            bird.pbest = bird.fitness
            pop.append(bird)

        # 找到种群中的最优位置
        self.g_best(pop)

    # 更新速度和位置
    def update(self, pop):
        for bird in pop:
            # 速度更新
            speed = self.w * bird.speed + self.c1 * np.random.random() * (
                bird.pbest - bird.pos) + self.c2 * np.random.random() * (
                self.gbest - bird.pos)

            # 位置更新
            pos = bird.pos + speed

            if -10 < pos < 10: # 必须在搜索空间内
                bird.pos = pos
                bird.speed = speed
                # 更新适应度
                bird.fitness = self.fitness(bird.pos)

                # 是否需要更新本粒子历史最好位置
                if bird.fitness > self.fitness(bird.pbest):
                    bird.pbest = bird.pos

    # 最终执行
    def implement(self):
        # 初始化种群
        self.initPopulation(self.POP, self.N)

        # 迭代
        for i in range(self.iter_N):
            # 更新速度和位置
            self.update(self.POP)
            # 更新种群中最好位置
            self.g_best(self.POP)


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

def PlotResultMyPSO():
    pso = ApolloidPSO()
    pso.implement()

    best_x = 0
    best_y = 0
    for ind in pso.POP:
        # print("x=", ind.pos, "f(x)=", ind.fitness)
        if ind.fitness > best_y:
            best_y = ind.fitness
            best_x = ind.pos
    print(best_y)
    print(best_x)

    x = np.linspace(-10, 10, 100000)


    def fun(x):
        return x + 16 * np.sin(5 * x) + 10 * np.cos(4 * x)


    y = fun(x)
    plt.plot(x, y)

    plt.scatter(best_x, best_y, c='r', label='best point')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    pass

    # workspace
    # load_power18480()
    # PlotResultMyPSO()





