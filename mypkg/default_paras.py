# this file contains some default values for the project
from easydict import EasyDict as edict
import numpy as np

def_paras = edict()

def_paras.npts = 40 # num of pts to evaluate X(s)


def_paras.a = 3.7 # parameter of SCAD from Fan's paper of SCAD
def_paras.alpha = 0.9 # parameter of optimization procedure, from the paper 
def_paras.beta = 1 # parameter of optimization procedure, from the pap
def_paras.NR_eps = 1e-5 # the stop criteria for Newton-Ralpson method, only for logistic model
def_paras.NR_maxit = 100 # the max num of iterations for Newton-Ralpson method, only for logistic mo
def_paras.stop_cv = 5e-4 # stop cv for convergence
def_paras.max_iter = 2000 # maximal iteration number



# the parameters for B-spline
def_paras.ord = 4
def_paras.N = 10 # num of basis for bsp
def_paras.x = np.linspace(0, 1, def_paras.npts)
