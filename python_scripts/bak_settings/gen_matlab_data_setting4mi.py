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
paras.freqs = np.linspace(2, 45, paras.npts) # freqs
paras.is_std = False # whether to std PSD across freq or not

# b-spline
paras.x = np.linspace(0, 1, paras.npts)

paras.alp_GT = np.array([0])
# fourier basis
cs = [args.cs, 0.0, 0.0] # for sinica paper
postfix = "_setting4mi"
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

def _is_exists(tmp_paras):
    """
    Check if a file with the given parameters exists.

    Args:
    tmp_paras:
        d (int): The value of d in the file name.
        n (int): The value of n in the file name.
        npts:
        is_std
        seed (int): The seed value in the file name.

    Returns:
    bool or Path: Returns the file path if the file exists, otherwise returns False.
    """
    _get_n = lambda fil: int(fil.stem.split("_")[2].split("-")[-1])
    fils = MIDRES_ROOT.glob(f"PSD_d-{tmp_paras.d}_n-*npts-{tmp_paras.npts}_is_std-{tmp_paras.is_std}")
    # We do not need fil with n as we know the data with corresponding seed does not exist
    fils = [fil for fil in fils if _get_n(fil) !=tmp_paras.n]
    if len(fils) == 0:
        return False
    else:
        fils = sorted(fils, key=_get_n)
        ns = np.array([_get_n(fil) for fil in fils])
        idxs = np.where(tmp_paras.n <= ns)[0]
        if len(idxs) == 0:
            return False
        else:
            fil =fils[idxs[0]]
            path = MIDRES_ROOT/fil/f"seed_{tmp_paras.seed}.pkl"
            return path if path.exists() else False
def _get_filename(params):
    keys = ["d", "n", "npts", "is_std"]
    folder_name = 'PSD_'+'_'.join(f"{k}-{params[k]}" for k in keys)
    return folder_name + f'/seed_{params.seed}.pkl'
def _gen_simu_data_sinica(seed, paras, verbose=False, is_gen=False):
    """
    Generate simulated data for all parameters.

    Args:
        seed (int): Seed for random number generator.
        paras (dict): Dictionary containing the following parameters:
            - n (int): Number of samples.
            - d (int): Number of dimensions.
            - q (int): Number of covariates.
            - types_ (list): List of types for generating covariates.
            - alp_GT (list): List of ground truth alpha values.
            - beta_GT (list): List of ground truth beta values.
            - freqs (list): List of frequencies for generating simulated PSD.
            - sigma2 (float): Variance of the noise.
        verbose(bool): Verbose or not
        is_gen(bool): Only for generating or not. If True, only checking or generating X, not return anything.

    Returns:
        all_data (dict): Dictionary containing the following simulated data:
            - X (torch.Tensor): Tensor of shape (n, d, npts) containing the simulated PSD.
            - Y (torch.Tensor): Tensor of shape (n,) containing the response variable.
            - Z (torch.Tensor): Tensor of shape (n, q) containing the covariates.
    """
    np.random.seed(seed)
    _paras = edict(paras.copy())
    # simulated PSD
    assert len(_paras.types_) == _paras.q
    assert len(_paras.alp_GT) == _paras.q
    tmp_paras = edict()
    tmp_paras.seed = seed 
    tmp_paras.n = _paras.n
    tmp_paras.d = _paras.d
    tmp_paras.npts = _paras.npts
    tmp_paras.is_std = _paras.is_std
    
    file_path = MIDRES_ROOT/_get_filename(tmp_paras)
    if file_path.exists():
        if is_gen:
            return None
        simu_curvs = load_pkl(file_path, verbose=verbose)
    else:
        ofil =  _is_exists(tmp_paras)
        if ofil:
            if is_gen:
                return None
            simu_curvs = load_pkl(ofil, verbose=verbose)
        else:
            if _paras.is_std:
                simu_curvs = gen_simu_psd(_paras.n, _paras.d, _paras.freqs, prior_sd=10, n_jobs=28, is_prog=False, is_std=_paras.is_std)
            else:
                simu_curvs = gen_simu_psd(_paras.n, _paras.d, _paras.freqs, prior_sd=10, n_jobs=28, is_prog=False, is_std=_paras.is_std)
                simu_curvs = simu_curvs - simu_curvs.mean(axis=-1, keepdims=True); # not std, but center it
            save_pkl(file_path, simu_curvs, verbose=verbose)
    if is_gen:
        return None
    simu_curvs = simu_curvs[:_paras.n]
    simu_curvs = (simu_curvs + np.random.randn(*simu_curvs.shape)*3)*1 # larger
    #simu_curvs = np.random.randn(_paras.n, _paras.d, _paras.npts)* 10
    simu_covs = gen_covs(_paras.n, _paras.types_)
    
    # linear term and Y
    int_part = np.sum(_paras.beta_GT.T* simu_curvs[:, :, :], axis=1).mean(axis=1)
    cov_part = simu_covs @ _paras.alp_GT 
    
    # linear term
    lin_term = cov_part + int_part
    
    # Y 
    Y = lin_term + np.random.randn(_paras.n)*np.sqrt(_paras.sigma2)
    
    # center
    X_centered = simu_curvs - simu_curvs.mean(axis=0, keepdims=True)
    Y_centered = Y - Y.mean(axis=0, keepdims=True)
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



