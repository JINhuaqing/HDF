#!/usr/bin/env python
# coding: utf-8

# This file contains python code to compare with sinica paper
# 
# It is under the linear setting
# 
# Now, I use the same beta from the paper but the PSD as X

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

from easydict import EasyDict as edict
from tqdm import trange, tqdm
from scipy.io import loadmat
from pprint import pprint
import itertools
from scipy.stats import chi2



# In[6]:


from constants import DATA_ROOT, RES_ROOT, FIG_ROOT, MIDRES_ROOT
from default_paras import def_paras

from hdf_utils.data_gen import gen_covs, gen_simu_psd
from hdf_utils.fns_sinica import coef_fn, fourier_basis_fn, gen_sini_Xthetas
from hdf_utils.likelihood import obt_lin_tm
from hdf_utils.SIS import SIS_linear
from hdf_utils.utils import gen_lam_seq
from hdf_utils.hypo_test import  MS2idxs, obt_test_stat_simple2
from utils.matrix import col_vec_fn, col_vec2mat_fn, conju_grad, svd_inverse, cholesky_inv
from utils.functions import logit_fn
from utils.misc import save_pkl, load_pkl
from splines import obt_bsp_obasis_Rfn, obt_bsp_basis_Rfn_wrapper
from projection import euclidean_proj_l1ball
from optimization.one_step_opt import OneStepOpt
from optimization.cross_validation import CV_err_linear_fn
from optimization.opt import optimization
from optimization.variable_selection import GIC_fn, GCV_fn
from penalties.scad_pen import SCAD
from models.linear_model import LinearModel

from joblib import Parallel, delayed

import argparse
parser = argparse.ArgumentParser(description='run')
parser.add_argument('-c', '--cs', type=float, help='cs value') 
args = parser.parse_args()

# In[18]:


plt.style.use(FIG_ROOT/"base.mplstyle")
torch.set_default_tensor_type(torch.DoubleTensor)


# In[ ]:





# In[ ]:





# # Param and fns

# In[ ]:





# ## Params

# In[75]:


cs = [args.cs, 0.0, 0.0] # for sinica paper
obt_bsp = obt_bsp_obasis_Rfn
np.random.seed(0)
paras = edict(def_paras.copy())



# Others
paras.num_rep = 200 
paras.num_rep_CV = 200 
paras.init_noise_sd = -1 # the sd of the noise added to the true value for initial values, if -1, make init 0
#paras.SIS_ratio = 1 # the ratio to keep with SIS procedure
paras.SIS_ratio = 0.2 # the ratio to keep with SIS procedure
paras.linear_theta_update="cholesky_inv"

# candidate sets of tuning parameters, only two 
# lambda: penalty term
# N: num of basis
paras.can_lams = [0.001, 0.1, 0.3, 0.35, 0.40, 0.45, 0.50, 0.55, 0.6, 0.65, 0.7, 0.75, 0.80, 1.0, 2.0, 4.0, 16.0]
paras.can_Ns = [4, 6, 8]


# generating dataset
paras.n = 100 # num of data obs to be genareted
paras.npts = 100 # num of pts to evaluate X(s)
paras.freqs = np.linspace(2, 45, paras.npts) # freqs
paras.d = 200 # num of ROIs
paras.q = 1 # num of other covariates
paras.sigma2 = 1 # variance of the error
# variance used for estimation, note that the value does not affect any results 
# as long as I tune the parameter properly
paras.norminal_sigma2 = 1 # variance of the error
paras.types_ = ["int"]
paras.srho = 0.3 # corr from sinica

# b-spline
paras.x = np.linspace(0, 1, paras.npts)
paras.basis_mats = []
for N in paras.can_Ns:
    paras.basis_mats.append(
        torch.tensor(obt_bsp(paras.x, N, paras.ord)).to(torch.get_default_dtype())
    )

# True parameters
paras.alp_GT = np.array([0])
# fourier basis
paras.fourier_basis = fourier_basis_fn(paras.x)[:, :]
paras.fourier_basis_coefs = ([cs[0]*coef_fn(0.2), cs[1]*coef_fn(0.2), cs[2]*coef_fn(0.2)] + 
                             [np.zeros(50)] * (paras.d-3-1) +
                             [coef_fn(0.2)]
                             )
paras.fourier_basis_coefs = np.array(paras.fourier_basis_coefs).T 
paras.beta_GT = paras.fourier_basis @ paras.fourier_basis_coefs
beta_GT_norm = np.linalg.norm(paras.beta_GT, axis=0)
print(beta_GT_norm)

paras.Gam_GT_ests = [(np.linalg.inv(basis_mat.numpy().T 
                                  @ basis_mat.numpy()) 
                                  @ basis_mat.numpy().T 
                                  @ paras.beta_GT) 
                     for basis_mat in paras.basis_mats]

# optimization
# not used, to use it, you have to know GT
#Rmins = [(2*(np.linalg.norm(paras.Gam_GT_ests[ix]
#                            /np.sqrt(paras.can_Ns[ix]), axis=0).sum() 
#           + np.abs(paras.alp_GT).sum())) 
#        for ix in range(len(paras.can_Ns))]
#paras.Rmin = np.max(Rmins)
paras.Rmin = 100000
paras.Rfct = 2
paras.stop_cv = 5e-4
paras.max_iter = 10000
paras.num_cv_fold = 5
# it is the parametes for SCAD
paras.a = 3.7 

# CV
paras.cv_is_center = True
paras.cv_SIS_ratio = paras.SIS_ratio
paras.cv_SIS_pen = 1
paras.cv_SIS_basis_mat = torch.tensor(obt_bsp(paras.x, 6, paras.ord)).to(torch.get_default_dtype())
paras.num_cv_fold = 5
paras.cv_init_noise_sd = -1

# hypothesis test
#without loss of generality, we assume the idxs in M is the first m betas
paras.sel_idx = np.arange(1, paras.d) # M^c set, 
paras.M_idxs = np.delete(np.arange(paras.d), paras.sel_idx) # the M set
paras.Cmats = [
    np.eye(len(paras.M_idxs)), # m x m I matrix, [beta1, beta2] = [0, 0]
]
paras.svdinv_eps_Q = 0 # now 0 means inverse, small value like 0.01 means remove small eig vals.
paras.svdinv_eps_Psi = 0


# saving path
paras.save_dir = RES_ROOT/f"simu_setting5_{cs[0]*1000:.0f}"
if not paras.save_dir.exists():
    paras.save_dir.mkdir()


# In[ ]:





# ## Fns

# In[ ]:





# In[12]:
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
    torch.set_default_tensor_type(torch.DoubleTensor)
    np.random.seed(seed)
    _paras = edict(paras.copy())
    # simulated PSD
    assert len(_paras.types_) == _paras.q
    assert len(_paras.alp_GT) == _paras.q
   
    thetas = gen_sini_Xthetas(_paras.srho, _paras.n, _paras.d);
    simu_curvs = thetas @ _paras.fourier_basis.T; # n x d x npts
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
    #all_data.lin_term = lin_term
    return all_data



# ## Simulation

# In[19]:


def _run_fn(seed, lam, N, paras, is_save=False, is_cv=False, verbose=False):
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
    cur_data = _gen_simu_data_all(_paras.seed, _paras)
    
    f_name = f"seed_{seed:.0f}-lam_{lam*1000:.0f}-N_{N:.0f}-c1_{cs[0]*1000:.0f}_est.pkl"
    
    
    res = edict()
    if not (_paras.save_dir/f_name).exists():
        # do sure independent screening for dim reduction
        if _paras.SIS_ratio < 1:
            keep_idxs, _  = SIS_linear(cur_data.Y, cur_data.X, cur_data.Z, _paras.basis_mats[_paras.can_Ns.index(6)],
                                       _paras.SIS_ratio, _paras, ridge_pen=1)
        else:
            keep_idxs = _paras.sel_idx
        M_idxs = np.delete(np.arange(_paras.d), _paras.sel_idx)
        _paras.keep_idxs = np.sort(np.concatenate([M_idxs, keep_idxs]))
            
        _paras.sel_idx_SIS = np.where(np.array([keep_idx in _paras.sel_idx for keep_idx in _paras.keep_idxs]))[0]
        _paras.d_SIS = len(_paras.keep_idxs)
        
        cur_data_SIS = edict(cur_data.copy())
        cur_data_SIS.X = cur_data.X[:, _paras.keep_idxs, :]
        
        
        if _paras.init_noise_sd < 0:
            alp_init = torch.zeros(_paras.q)
            Gam_init = torch.zeros(_paras.N, _paras.d_SIS)
            theta_init = torch.cat([alp_init, col_vec_fn(Gam_init)/np.sqrt(_paras.N)])
            rhok_init = torch.zeros(_paras.d_SIS*_paras.N)
        else:
            alp_init = torch.Tensor(_paras.alp_GT) + torch.randn(_paras.q)*_paras.init_noise_sd
            Gam_init = torch.Tensor(_paras.Gam_GT_est[:, _paras.keep_idxs]) + torch.randn(_paras.N, _paras.d_SIS)*_paras.init_noise_sd
            theta_init = torch.cat([alp_init, col_vec_fn(Gam_init)/np.sqrt(_paras.N)])
            rhok_init = torch.randn(_paras.d_SIS*_paras.N)
            
        model = LinearModel(Y=cur_data_SIS.Y, 
                            X=cur_data_SIS.X, 
                            Z=cur_data_SIS.Z, 
                            basis_mat=_paras.basis_mat, 
                            sigma2=paras.norminal_sigma2)
        # 3e0
        pen = SCAD(lams=_paras.lam, a=_paras.a,  sel_idx=_paras.sel_idx_SIS)
            
        
        main_res = optimization(model=model, 
                                penalty=pen, 
                                inits=[alp_init, Gam_init, theta_init, rhok_init],
                                is_prg=verbose,
                                save_paras=False,    
                                input_paras=_paras)
        opt = main_res[0]
        est_Gam = opt.Gamk
        est_alp = opt.alpk
        Q_mat = -model.log_lik_der2(est_alp, est_Gam)
        model.log_lik_der1(est_alp, est_Gam);
        Sig_mat = (model.log_lik_der1_vs.unsqueeze(-1) * model.log_lik_der1_vs.unsqueeze(1)).mean(axis=0) 
        est_theta = torch.cat([est_alp, col_vec_fn(est_Gam)/np.sqrt(_paras.N)])
        nonzero_idxs = torch.nonzero(torch.norm(est_Gam, dim=0)).reshape(-1).numpy()
        MS_unions = np.sort(np.union1d(_paras.M_idxs, nonzero_idxs))
        keep_idxs_test = MS2idxs(_paras.q, _paras.N, MS_unions)
        Q_mat_part = Q_mat[keep_idxs_test][:, keep_idxs_test]
        Sig_mat_part = Sig_mat[keep_idxs_test][:, keep_idxs_test]
        
        res = edict()
        _paras.Gam_GT_ests = None
        _paras.basis_mats = None
        _paras.fourier_basis_coefs = None
        _paras.fourier_basis = None
        res._paras = _paras
        res.Sig_mat_part = Sig_mat_part
        res.Q_mat_part = Q_mat_part
        res.est_Gam = est_Gam
        res.est_alp = est_alp
        res.conv_num = main_res[1]
        res.est_sigma2 = torch.mean((model.Y - model._obt_lin_tm(est_alp, est_Gam))**2)
        res.AIC = GIC_fn(res, "AIC")
        res.BIC = GIC_fn(res, "BIC")
        res.GCV = GCV_fn(res)
    
        if is_cv:
            if _paras.init_noise_sd < 0:
                alp_init1 = torch.zeros(_paras.q)
                Gam_init1 = torch.zeros(_paras.N, _paras.d_SIS)
                theta_init1 = torch.cat([alp_init, col_vec_fn(Gam_init)/np.sqrt(_paras.N)])
                rhok_init1 = torch.zeros(_paras.d_SIS*_paras.N)
            else:
                # use a diff initial to reduce the overfitting
                alp_init1 = torch.Tensor(_paras.alp_GT) + torch.randn(_paras.q)*_paras.init_noise_sd
                Gam_init1 = torch.Tensor(_paras.Gam_GT_est[:, _paras.keep_idxs]) + torch.randn(_paras.N, _paras.d_SIS)*_paras.init_noise_sd
                theta_init1 = torch.cat([alp_init, col_vec_fn(Gam_init)/np.sqrt(_paras.N)])
                rhok_init1 = torch.randn(_paras.d_SIS*_paras.N)
            cv_errs = CV_err_linear_fn(data=cur_data, 
                                       num_cv_fold=_paras.num_cv_fold,
                                       is_prg=verbose, 
                                       save_paras=False,    
                                       input_paras=_paras)
            
            res.cv_errs = cv_errs
        else:
            res.cv_errs = None
        if is_save:
            save_pkl(_paras.save_dir/f_name, res, verbose=verbose)
    return res


# In[20]:


all_coms = itertools.product(range(0, paras.num_rep_CV), paras.can_lams, paras.can_Ns)
with Parallel(n_jobs=35) as parallel:
    ress = parallel(delayed(_run_fn)(seed, lam=lam, N=N, paras=paras, is_save=True, is_cv=True, verbose=False) 
                    for seed, lam, N 
                    in tqdm(all_coms, total=len(paras.can_Ns)*len(paras.can_lams)*paras.num_rep_CV))
