# some fns
import numpy as np
import torch

def logit_fn(x):
    rv = np.exp(x)/(1+np.exp(x))
    return rv