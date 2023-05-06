import numpy as np


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