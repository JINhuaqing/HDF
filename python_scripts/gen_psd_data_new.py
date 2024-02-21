#!/usr/bin/env python
# coding: utf-8

# This file contains python code to compare with sinica method
# 
# It is under the linear setting
# 
# Now, I use the same beta from the paper but the PSD as X

# In[5]:


import sys
sys.path.append("../../mypkg")


# In[6]:


import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from easydict import EasyDict as edict
from tqdm import trange, tqdm
from pprint import pprint
import itertools
from scipy.stats import chi2


from constants import DATA_ROOT, RES_ROOT, FIG_ROOT, MIDRES_ROOT
from hdf_utils.data_gen import gen_simu_psd_dataset
from hdf_utils.SIS import SIS_GLIM
from utils.matrix import col_vec_fn, col_vec2mat_fn, conju_grad, svd_inverse, cholesky_inv
from utils.functions import logit_fn
from utils.misc import save_pkl, load_pkl
from splines import obt_bsp_obasis_Rfn, obt_bsp_basis_Rfn_wrapper
from projection import euclidean_proj_l1ball
from optimization.opt import HDFOpt
from scenarios.simu_linear_psd import settings

from joblib import Parallel, delayed



np.random.seed(0)
c = 0.0

setting = settings.cmpn1b
data_gen_params = setting.data_gen_params
data_gen_params.cs = data_gen_params.cs_fn(c)
data_gen_params.gt_beta = data_gen_params.beta_fn(data_gen_params.cs)


data = gen_simu_psd_dataset(n=data_gen_params.n, 
                            d=data_gen_params.d, 
                            q=data_gen_params.q, 
                            types_=data_gen_params.types_, 
                            gt_alp=data_gen_params.gt_alp, 
                            gt_beta=data_gen_params.gt_beta, 
                            freqs=data_gen_params.freqs, 
                            data_type=data_gen_params.data_type, 
                            data_params=data_gen_params.data_params, 
                            seed=0, 
                            is_std=data_gen_params.is_std, 
                            verbose=1, 
                            is_gen=False);

