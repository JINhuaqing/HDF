# this file contains fns to generate beta 
# based on the settings from sinica paper
# "Hypothesis Testing in Large-scale Functional Linear Regression"


import numpy as np
from numbers import Number

def fourier_basis_fn(x):
    """The function to regresion fourier basis at x. 
       The setting is based on paper 
           "Hypothesis Testing in Large-scale Functional Linear Regression"
       args:
           x: The locs to evaluate the basis
       return:
           fourier_basis: matrix of len(x) x 50
    """

    if isinstance(x, Number):
        xs = np.array([x])
    else:
        xs = np.array(x)
    
    fourier_basis = []
    for l in range(1, 51):
        if l == 1:
            fourier_basis.append(np.ones(len(xs)))
        elif l % 2 == 0:
            fourier_basis.append(np.sqrt(2)*np.cos(l*np.pi*(2*xs-1)))
        elif l % 2 == 1:
            fourier_basis.append(np.sqrt(2)*np.sin((l-1)*np.pi*(2*xs-1)))
    fourier_basis = np.array(fourier_basis).T
    return fourier_basis


def coef_fn(k=0.2):
    """The function to return the coefficients from the paper when k=0.2
    """
    part1 = 1.2 - k * np.array([1, 2, 3, 4])
    part2 = 0.4 * (1/(np.arange(5, 51)-3))**(4)
    coefs = np.concatenate([part1, part2])
    return coefs