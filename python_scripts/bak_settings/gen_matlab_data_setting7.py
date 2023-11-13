#!/usr/bin/env python
# coding: utf-8

# This file contains python code to generate data for comparing with sinica paper
# 
# Note the method from sinica paper does not consider the covariate and intercept, 
# 
# moreover, it is centered across the sample size
# 
# Note here I save the orginal data (X), not the theta. 

# In[1]:


import sys
sys.path.append("../mypkg")


# In[2]:


import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from numbers import Number
import multiprocessing as mp

from easydict import EasyDict as edict
from tqdm import trange, tqdm
from scipy.io import loadmat
from pprint import pprint
import itertools
from scipy.stats import chi2


# In[3]:



# In[4]:


from constants import DATA_ROOT, RES_ROOT, FIG_ROOT, MIDRES_ROOT
from default_paras import def_paras

from hdf_utils.data_gen import gen_covs, gen_simu_psd, gen_simu_ts
from hdf_utils.fns import fn1, fn2, fn3, fn4, fn5, zero_fn
from hdf_utils.fns_sinica import coef_fn, fourier_basis_fn, gen_sini_Xthetas
from hdf_utils.likelihood import obt_lin_tm
from hdf_utils.SIS import SIS_linear
from utils.matrix import col_vec_fn, col_vec2mat_fn, conju_grad, svd_inverse
from utils.functions import logit_fn
from utils.misc import save_pkl, load_pkl
from splines import obt_bsp_basis_Rfn, obt_bsp_basis_Rfn_wrapper

from joblib import Parallel, delayed


import argparse
parser = argparse.ArgumentParser(description='run')
parser.add_argument('-c', '--cs', type=float, help='cs value') 
args = parser.parse_args()
# In[5]:


plt.style.use(FIG_ROOT/"base.mplstyle")


# In[6]:


torch.set_default_tensor_type(torch.DoubleTensor)


# In[ ]:





# # Param and fns

# ## Params

# In[7]:


np.random.seed(0)
paras = edict(def_paras.copy())

paras.num_rep = 200
paras.n = 100 # num of data obs to be genareted
paras.npts = 100 # num of pts to evaluate X(s)
paras.d = 200 # num of ROIs
paras.q = 1 # num of other covariates
paras.sigma2 = 1 # variance of the error
paras.types_ = ["int"]
paras.srho = 0.3 # corr from sinica

# b-spline
paras.x = np.linspace(0, 1, paras.npts)

paras.alp_GT = np.array([0])
# fourier basis
cs = [args.cs, 0.0, 0.0] # for sinica paper
postfix = "_setting7"
paras.fourier_basis = fourier_basis_fn(paras.x)[:, :]
paras.fourier_basis_coefs = ([cs[0]*coef_fn(0.2), cs[1]*coef_fn(0.2), cs[2]*coef_fn(0.2)] + 
                             [np.zeros(50)] * (paras.d-3-1) +
                             [coef_fn(0.2)*1]
                             )
paras.fourier_basis_coefs = np.array(paras.fourier_basis_coefs).T 
paras.beta_GT = paras.fourier_basis @ paras.fourier_basis_coefs 


# In[ ]:





# ## Fns

# In[8]:


def _gen_simu_data_sinica(seed, paras):
    """
    Generate simulated data for all parameters.

    Args:
        seed (int): Seed for random number generator.
        paras (dict): Dictionary containing the following parameters:
            - srho: corr from sinica
            - fourier_basis: The fourier basis for generating X, npts x nbasis
            - n (int): Number of samples.
            - d (int): Number of dimensions.
            - q (int): Number of covariates.
            - types_ (list): List of types for generating covariates.
            - alp_GT (list): List of ground truth alpha values.
            - beta_GT (list): List of ground truth beta values.
            - sigma2 (float): Variance of the noise.

    Returns:
        all_data (dict): Dictionary containing the following simulated data:
            - X (torch.Tensor): Tensor of shape (n, d, npts) containing the simulated PSD.
            - Y (torch.Tensor): Tensor of shape (n,) containing the response variable.
            - Z (torch.Tensor): Tensor of shape (n, q) containing the covariates.
    """
    torch.set_default_tensor_type(torch.DoubleTensor)
    np.random.seed(seed)
    _paras = edict(paras.copy())
    # simulated PSD
    assert len(_paras.types_) == _paras.q
    assert len(_paras.alp_GT) == _paras.q
   
    thetas = gen_sini_Xthetas(_paras.srho, _paras.n, _paras.d);
    simu_curvs = np.random.randn(_paras.n, _paras.d, _paras.npts)*5
    simu_covs = gen_covs(_paras.n, _paras.types_)
    
    # linear term and Y
    int_part = np.sum(_paras.beta_GT.T* simu_curvs[:, :, :], axis=1).mean(axis=1)
    cov_part = simu_covs @ _paras.alp_GT 
    
    # linear term
    lin_term = cov_part + int_part
    
    # Y 
    Y = lin_term + np.random.randn(_paras.n)*np.sqrt(_paras.sigma2)

    # for Sinica paper, center X and Y
    Y_centered = Y - Y.mean(axis=0, keepdims=True)
    X_centered = simu_curvs - simu_curvs.mean(axis=0, keepdims=True)
    sinica_data = {'Y_centered':Y_centered, 
                   'X_centered':X_centered}
    
    # To torch
    X = torch.Tensor(X_centered) # n x d x npts
    Z = torch.Tensor(simu_covs) # n x q
    Y = torch.Tensor(Y_centered)
    
    all_data = edict()
    all_data.X = X
    all_data.Y = Y
    all_data.Z = Z
    return all_data, edict(sinica_data)


# In[9]:





# # Generate data
# 

# In[10]:


from scipy.io import savemat
def _get_filename_mat(params, seed):
    params = edict(params.copy())
    keys = ["d", "n"]
    folder_name = 'SinicaX_'+'_'.join(f"{k}-{params[k]}" for k in keys) + postfix
    folder_name = MIDRES_ROOT/f"matlab_simu_data/{folder_name}"
    if not folder_name.exists():
        folder_name.mkdir()
    return folder_name /f'c1_{cs[0]*1000:.0f}_seed_{seed}.mat'

def _run_fn(seed):
    _, sinica_data = _gen_simu_data_sinica(seed, paras)
    savemat(_get_filename_mat(paras, seed), sinica_data)
    return None
with Parallel(n_jobs=35) as parallel:
    ress = parallel(delayed(_run_fn)(seed) for seed in tqdm(range(paras.num_rep), total=paras.num_rep))



