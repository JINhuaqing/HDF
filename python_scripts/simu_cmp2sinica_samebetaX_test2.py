#!/usr/bin/env python
# coding: utf-8

# This file contains python code to compare with sinica paper
# 
# It is under the linear setting
# 
# Now, I use the same X and beta from the paper (on Sep 4, 2023)

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


# In[4]:


from constants import DATA_ROOT, RES_ROOT, FIG_ROOT, MIDRES_ROOT
from default_paras import def_paras

from hdf_utils.data_gen import gen_covs, gen_simu_psd, gen_simu_ts
from hdf_utils.fns import fn1, fn2, fn3, fn4, fn5, zero_fn
from hdf_utils.fns_sinica import coef_fn, fourier_basis_fn, gen_sini_Xthetas
from hdf_utils.likelihood import obt_lin_tm
from hdf_utils.SIS import SIS_linear
from hdf_utils.utils import gen_lam_seq
from hdf_utils.hypo_test import obt_test_stat
from utils.matrix import col_vec_fn, col_vec2mat_fn, conju_grad, svd_inverse, cholesky_inv
from utils.functions import logit_fn
from utils.misc import save_pkl, load_pkl
from splines import obt_bsp_basis_Rfn, obt_bsp_basis_Rfn_wrapper
from projection import euclidean_proj_l1ball
from optimization.one_step_opt import OneStepOpt
from optimization.cross_validation import CV_err_linear_fn
from optimization.opt import optimization
from penalties.scad_pen import SCAD
from models.linear_model import LinearModel

from joblib import Parallel, delayed


# In[5]:


import logging

logger = logging.getLogger("tmp")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

ch = logging.StreamHandler() # for console. 
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)

logger.addHandler(ch)

import argparse
parser = argparse.ArgumentParser(description='run')
parser.add_argument('-c', '--cs', type=float, help='cs value') 
args = parser.parse_args()


# In[6]:


plt.style.use(FIG_ROOT/"base.mplstyle")
torch.set_default_tensor_type(torch.DoubleTensor)


# In[ ]:





# In[ ]:





# # Param and fns

# In[ ]:





# ## Params

# In[7]:


np.random.seed(0)
paras = edict(def_paras.copy())

# Others
paras.num_rep = 200 
paras.init_noise_sd = 0.5 # the sd of the noise added to the true value for initial values
paras.SIS_ratio = 1 # the ratio to keep with SIS procedure
paras.SIS_ratio = 0.20 # the ratio to keep with SIS procedure
paras.svdinv_eps_Q = 0 # now 0 means inverse, small value like 0.01 means remove small eig vals.
paras.svdinv_eps_Psi = 0

# candidate sets of tuning parameters, only two 
# lambda: penalty term
# N: num of basis
paras.can_lams = [0.01, 0.1, 0.2, 0.3, 0.4, 0.6, 1, 2, 8]
paras.can_Ns = [4, 6, 8, 10, 12]

# generating dataset
paras.n = 100 # num of data obs to be genareted
paras.npts = 100 # num of pts to evaluate X(s)
paras.d = 200 # num of ROIs
paras.q = 1 # num of other covariates
paras.sigma2 = 1 # variance of the error
paras.types_ = ["int"]
paras.srho = 0.3 # corr from sinica

# b-spline
paras.x = np.linspace(0, 1, paras.npts)
paras.basis_mats = []
for N in paras.can_Ns:
    paras.basis_mats.append(
        torch.tensor(obt_bsp_basis_Rfn_wrapper(paras.x, N, paras.ord)).to(torch.get_default_dtype())
    )

# True parameters
paras.alp_GT = np.array([0])
# fourier basis
cs = [args.cs, args.cs, 0] # for sinica paper
paras.fourier_basis = fourier_basis_fn(paras.x)[:, :]
paras.fourier_basis_coefs = ([cs[0]*coef_fn(0.2), cs[1]*coef_fn(0.2), cs[2]*coef_fn(0.2)] + 
                             [np.zeros(50)] * (paras.d-3-1) +
                             [coef_fn(0.2)*1]
                             )
paras.fourier_basis_coefs = np.array(paras.fourier_basis_coefs).T 
paras.beta_GT = paras.fourier_basis @ paras.fourier_basis_coefs 
print(np.linalg.norm(paras.beta_GT, axis=0))

paras.Gam_GT_ests = [(np.linalg.inv(basis_mat.numpy().T 
                                  @ basis_mat.numpy()) 
                                  @ basis_mat.numpy().T 
                                  @ paras.beta_GT) 
                     for basis_mat in paras.basis_mats]

# optimization
Rmins = [(2*(np.linalg.norm(paras.Gam_GT_ests[ix]
                            /np.sqrt(paras.can_Ns[ix]), axis=0).sum() 
           + np.abs(paras.alp_GT).sum())) 
        for ix in range(len(paras.can_Ns))]
paras.Rmin = np.max(Rmins)
#without loss of generality, we assume the idxs in M is the first m betas
paras.sel_idx = np.arange(2, paras.d) # M^c set, 
paras.num_cv_fold = 5
paras.Rfct = 2
#paras.stop_cv = 5e-5
paras.stop_cv = 5e-4
#paras.max_iter = 10000
paras.max_iter = 10000

# misc
paras.linear_theta_update="cholesky_inv"
paras.is_center = True

# hypothesis test
paras.M_idxs = np.delete(np.arange(paras.d), paras.sel_idx) # the M set


# In[8]:


paras.save_dir = RES_ROOT/"simu_linear_sinica_samebetaX_test2"
if not paras.save_dir.exists():
    paras.save_dir.mkdir()


# In[ ]:





# ## Fns

# In[ ]:





# In[9]:


def _gen_simu_data_all(seed, paras, verbose=False):
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
            - freqs (list): List of frequencies for generating simulated PSD.
            - sigma2 (float): Variance of the noise.
        verbose(bool): Verbose or not

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
   
    thetas = gen_sini_Xthetas(_paras.srho, _paras.n, _paras.d);
    simu_curvs = thetas @ _paras.fourier_basis.T;
    simu_covs = gen_covs(_paras.n, _paras.types_)
    
    # linear term and Y
    int_part = np.sum(_paras.beta_GT.T* simu_curvs[:, :, :], axis=1).mean(axis=1)
    cov_part = simu_covs @ _paras.alp_GT 
    
    # linear term
    lin_term = cov_part + int_part
    
    # Y 
    Y = lin_term + np.random.randn(_paras.n)*np.sqrt(_paras.sigma2)
    
    Y_centered = Y - Y.mean(axis=0, keepdims=True)
    X_centered = simu_curvs - simu_curvs.mean(axis=0, keepdims=True)
    
    # To torch
    X = torch.Tensor(X_centered) # n x d x npts
    Z = torch.Tensor(simu_covs) # n x q
    Y = torch.Tensor(Y_centered)
    
    all_data = edict()
    all_data.X = X
    all_data.Y = Y
    all_data.Z = Z
    all_data.lin_term = lin_term
    return all_data


# In[ ]:





# # Simu

# In[ ]:








# ## Simulation

# In[10]:


def _run_fn(seed, lam, N, paras, is_save=False, is_prg=False, is_cv=False, is_verbose=False):
    """Now (on Aug 25, 2023), if we keep seed the same, the cur_data is the same. 
       If you want to make any changes, make sure this. 
    """
    torch.set_default_tensor_type(torch.DoubleTensor)
    np.random.seed(seed)
    torch.manual_seed(seed)
        
    _paras = edict(paras.copy())
    _paras.Rv = _paras.Rfct * _paras.Rmin
    _paras.seed = seed
    _paras.lam = lam
    _paras.N = N
    _paras.basis_mat = _paras.basis_mats[_paras.can_Ns.index(N)]
    _paras.Gam_GT_est = paras.Gam_GT_ests[_paras.can_Ns.index(N)]
    
    f1_name = f"seed_{seed:.0f}-lam_{lam*1000:.0f}-N_{N:.0f}-c1_{cs[0]*1000:.0f}_est.pkl"
    f2_name = f"seed_{seed:.0f}-lam_{lam*1000:.0f}-N_{N:.0f}-c1_{cs[0]*1000:.0f}_cv.pkl"
    
    
    if not (_paras.save_dir/f1_name).exists():
        cur_data = _gen_simu_data_all(_paras.seed, _paras)
        # do sure independent screening for dim reduction
        if _paras.SIS_ratio < 1:
            keep_idxs, _  = SIS_linear(cur_data.Y, cur_data.X, cur_data.Z, _paras.basis_mats[_paras.can_Ns.index(8)],
                                       _paras.SIS_ratio, _paras, ridge_pen=1)
        else:
            keep_idxs = _paras.sel_idx
        M_idxs = np.delete(np.arange(_paras.d), _paras.sel_idx)
        _paras.keep_idxs = np.sort(np.concatenate([M_idxs, keep_idxs]))
            
        _paras.sel_idx_SIS = np.where(np.array([keep_idx in _paras.sel_idx for keep_idx in _paras.keep_idxs]))[0]
        _paras.d_SIS = len(_paras.keep_idxs)
        
        cur_data_SIS = edict(cur_data.copy())
        cur_data_SIS.X = cur_data.X[:, _paras.keep_idxs, :]
        
        
        alp_init = (torch.Tensor(_paras.alp_GT) + torch.randn(_paras.q)*_paras.init_noise_sd)*0
        Gam_init = (torch.Tensor(_paras.Gam_GT_est[:, _paras.keep_idxs]) 
                    + torch.randn(_paras.N, _paras.d_SIS)*_paras.init_noise_sd)*0
        theta_init = torch.cat([alp_init, col_vec_fn(Gam_init)/np.sqrt(_paras.N)]) *0
        rhok_init = torch.randn(_paras.d_SIS*_paras.N)*0
            
        model = LinearModel(Y=cur_data_SIS.Y, 
                            X=cur_data_SIS.X, 
                            Z=cur_data_SIS.Z, 
                            basis_mat=_paras.basis_mat, 
                            sigma2=_paras.sigma2)
        # 3e0
        pen = SCAD(lams=_paras.lam, a=_paras.a,  sel_idx=_paras.sel_idx_SIS)
            
        
        main_res = optimization(model=model, 
                                penalty=pen, 
                                inits=[alp_init, Gam_init, theta_init, rhok_init],
                                is_prg=is_prg,
                                save_paras=False,    
                                input_paras=_paras)
        res1 = edict()
        res1._paras = _paras
        res1.main_res = main_res
        res1.model = model
        if is_save:
            save_pkl(_paras.save_dir/f1_name, res1, verbose=is_verbose)
    
    if is_cv:
        res1 = load_pkl(_paras.save_dir/f1_name, verbose=is_verbose)
        _paras = res1._paras
        cur_data_SIS = edict()
        cur_data_SIS.X = res1.model.X
        cur_data_SIS.Y = res1.model.Y
        cur_data_SIS.Z = res1.model.Z
        
        
        pen = SCAD(lams=_paras.lam, a=_paras.a,  sel_idx=_paras.sel_idx_SIS)
        
        # use a diff initial to reduce the overfitting
        alp_init1 = (torch.Tensor(_paras.alp_GT) + torch.randn(_paras.q)*_paras.init_noise_sd)*0
        Gam_init1 = (torch.Tensor(_paras.Gam_GT_est[:, _paras.keep_idxs]) 
                    + torch.randn(_paras.N, _paras.d_SIS)*_paras.init_noise_sd) *0
        theta_init1 = torch.cat([alp_init1, col_vec_fn(Gam_init1)/np.sqrt(_paras.N)])*0
        rhok_init1 = torch.randn(_paras.d_SIS*_paras.N)*0
        cv_errs = CV_err_linear_fn(data=cur_data_SIS, 
                                   penalty=pen, 
                                   num_cv_fold=_paras.num_cv_fold,
                                   # do not use estimated value for initial, severe overfitting !!! (on Aug 25, 2023)
                                   inits=[alp_init1, Gam_init1, theta_init1, rhok_init1], 
                                   is_prg=is_prg, 
                                   save_paras=False,    
                                   input_paras=_paras)
            
        res2 = edict()
        res2.cv_errs = cv_errs
        if is_save:
            save_pkl(_paras.save_dir/f2_name, res2, verbose=is_verbose)
    return None



all_coms = itertools.product(range(0, paras.num_rep), paras.can_lams, paras.can_Ns);
with Parallel(n_jobs=25) as parallel:
    ress = parallel(delayed(_run_fn)(seed, lam=lam, N=N, paras=paras, is_save=True, is_cv=True) for seed, lam, N 
                    in tqdm(all_coms, total=len(paras.can_Ns)*len(paras.can_lams)*paras.num_rep))

