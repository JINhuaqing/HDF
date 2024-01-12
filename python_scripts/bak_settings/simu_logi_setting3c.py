#!/usr/bin/env python
# coding: utf-8

# This file contains python code to compare with sinica paper
# 
# It is under the logistic setting
# 
# Now, I use the PSD.

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


# In[4]:


from constants import DATA_ROOT, RES_ROOT, FIG_ROOT, MIDRES_ROOT
from default_paras import def_paras

from hdf_utils.data_gen import gen_covs, gen_simu_psd, gen_simu_ts
from hdf_utils.fns import fn1, fn2, fn3, fn4, fn5, zero_fn
from hdf_utils.fns_sinica import coef_fn, fourier_basis_fn, gen_sini_Xthetas
from hdf_utils.likelihood import obt_lin_tm
from hdf_utils.SIS import SIS_GLIM
from hdf_utils.utils import gen_lam_seq
from hdf_utils.hypo_test import  MS2idxs, obt_test_stat_simple2
from utils.matrix import col_vec_fn, col_vec2mat_fn, conju_grad, svd_inverse, cholesky_inv
from utils.functions import logit_fn
from utils.misc import save_pkl, load_pkl, bcross_entropy_loss
from splines import obt_bsp_obasis_Rfn, obt_bsp_basis_Rfn_wrapper
from projection import euclidean_proj_l1ball
from optimization.one_step_opt import OneStepOpt
from optimization.cross_validation import CV_err_logi_fn
from optimization.opt import optimization
from penalties.scad_pen import SCAD
from models.logistic_model import LogisticModel
from optimization.variable_selection import GIC_fn, GCV_fn

from joblib import Parallel, delayed

import argparse
parser = argparse.ArgumentParser(description='run')
parser.add_argument('-c', '--cs', type=float, help='cs value') 
args = parser.parse_args()

# In[5]:


plt.style.use(FIG_ROOT/"base.mplstyle")
torch.set_default_tensor_type(torch.DoubleTensor)


# In[ ]:





# In[ ]:





# # Param and fns

# In[ ]:





# ## Params

# In[6]:


cs = [args.cs+0.5, 0.5, 0.0] # for sinica paper
np.random.seed(0)
obt_bsp = obt_bsp_obasis_Rfn
paras = edict(def_paras.copy())



# Others
paras.num_rep = 200 
paras.num_rep_CV = 200 
paras.init_noise_sd = -1 # the sd of the noise added to the true value for initial values, if -1, make init 0
paras.SIS_ratio = 0.2 # the ratio to keep with SIS procedure
paras.SIS_pen = 0.02 
paras.linear_theta_update="cholesky_inv"

# candidate sets of tuning parameters, only two 
# lambda: penalty term
# N: num of basis
paras.can_lams = [0.1, 0.2, 0.3]
#paras.can_lams = [0.001, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0, 2.0, 4.0, 16.0]
paras.can_Ns = [4, 6, 8]


# generating dataset
paras.n = 200 # num of data obs to be genareted
paras.npts = 100 # num of pts to evaluate X(s)
paras.d = 68 # num of ROIs
paras.q = 3 # num of other covariates
paras.types_ = ["int", "c", 2]
paras.is_std = False # std PSD or not


# b-spline
paras.x = np.linspace(0, 1, paras.npts)
paras.basis_mats = []
for N in paras.can_Ns:
    paras.basis_mats.append(
        torch.tensor(obt_bsp(paras.x, N, paras.ord)).to(torch.get_default_dtype())
    )
paras.SIS_basis_mat = torch.tensor(obt_bsp(paras.x, 8, paras.ord)).to(torch.get_default_dtype())

# True parameters
# Need to adaptively change intercept for each setting
# to make such 1 and 0 is balanced
paras.alp_GT0 = np.array([-1, 2]) # not include intercept
paras.intercept_cans = np.linspace(-30, 1, 20) 
# fourier basis
paras.fourier_basis = fourier_basis_fn(paras.x)[:, :]
paras.fourier_basis_coefs = ([cs[0]*coef_fn(0.2), cs[1]*coef_fn(0.2), cs[2]*coef_fn(0.2)] + 
                             [np.zeros(50)] * (paras.d-3-1) +
                             [coef_fn(0.2)]
                             )
paras.fourier_basis_coefs = np.array(paras.fourier_basis_coefs).T 
paras.beta_GT = paras.fourier_basis @ paras.fourier_basis_coefs * 2
print(np.linalg.norm(paras.beta_GT, axis=0))

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
paras.max_iter = 2000
paras.num_cv_fold = 5

paras.N_eps = 1e-4 # the stop criteria for Newton-Ralpson method, only for logistic model
paras.N_maxit = 100
paras.is_BFGS = "adaptive"

paras.cv_is_center = True
paras.cv_SIS_ratio = paras.SIS_ratio
paras.cv_SIS_pen = paras.SIS_pen
paras.cv_SIS_basis_mat = paras.SIS_basis_mat
paras.num_cv_fold = 5
paras.cv_init_noise_sd = paras.init_noise_sd


# hypothesis test
#without loss of generality, we assume the idxs in M is the first m betas
paras.sel_idx = np.arange(2, paras.d) # M^c set, 
paras.M_idxs = np.delete(np.arange(paras.d), paras.sel_idx) # the M set
paras.Cmats = [
    np.array([1, -1]).reshape(1, -1), # beta1-beta2=0
]

paras.svdinv_eps_Q = 1e-7 # now 0 means inverse, small value like 0.01 means remove small eig vals.
paras.svdinv_eps_Psi = 1e-7

paras.save_dir = RES_ROOT/f"simu_logi_setting3c_{cs[0]*1000:.0f}"
if not paras.save_dir.exists():
    paras.save_dir.mkdir()


# In[ ]:





# ## Fns

# In[ ]:





# In[7]:


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
def _gen_simu_data_all(seed, paras, verbose=False, is_gen=False):
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
    con_idxs = [typ =="c" for typ in _paras.types_]
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
    simu_curvs = (simu_curvs + np.random.randn(*simu_curvs.shape)*10)*1 # larger
    #simu_curvs = np.random.randn(_paras.n, _paras.d, _paras.npts)* 10
    simu_covs = gen_covs(_paras.n, _paras.types_)
    
    # linear term and Y
    int_part = np.sum(_paras.beta_GT.T* simu_curvs[:, :, :], axis=1).mean(axis=1)
    cov_part = simu_covs @ _paras.alp_GT 
    
    # linear term
    lin_term = cov_part + int_part
    probs = logit_fn(lin_term)
    
    # Y 
    Y = np.random.binomial(1, probs, size=len(probs))
    
    # center
    X_centered = simu_curvs - simu_curvs.mean(axis=0, keepdims=True)
    # this step is not necessary for simulation as I did so in generating data step
    # but for real data, plz do this
    Z_std = simu_covs.copy()
    Z_std[:, con_idxs] = ((simu_covs[:, con_idxs] - simu_covs[:, con_idxs].mean(axis=0, keepdims=True))
                          /simu_covs[:, con_idxs].std(axis=0, keepdims=True))
    
    
    # To torch
    X = torch.Tensor(X_centered) # n x d x npts
    Z = torch.Tensor(Z_std) # n x q
    Y = torch.Tensor(Y)
    
    all_data = edict()
    all_data.X = X
    all_data.Y = Y
    all_data.Z = Z
    #all_data.lin_term = lin_term
    return all_data


# In[8]:


ress = []
for inte in tqdm(paras.intercept_cans):
    tmp_paras = edict(paras.copy())
    tmp_paras.alp_GT = np.concatenate([[inte], paras.alp_GT0])
    _run_fn = lambda seed: _gen_simu_data_all(seed, tmp_paras).Y.numpy()
    with Parallel(n_jobs=30) as parallel:
        res = parallel(delayed(_run_fn)(seed) for seed in range(100))
    ress.append(np.array(res))


# In[9]:


# get the intercept
Yms = np.array([res.mean() for res in ress])
intercept = paras.intercept_cans[np.argmin(np.abs(Yms-0.5))]
paras.alp_GT = np.concatenate([[intercept], paras.alp_GT0])
print(f"The mean of Y is {Yms[np.argmin(np.abs(Yms-0.5))]:.3f} under intercept {intercept:.3f}.")


# In[ ]:





# # Simu




# ## Simulation

# In[ ]:





# In[10]:



def _run_fn(seed, lam, N, paras, is_save=False, is_cv=False, verbose=False):
    torch.set_default_tensor_type(torch.DoubleTensor)
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
        if _paras.SIS_ratio < 1:
            keep_idxs, _ = SIS_GLIM(Y=cur_data.Y, X=cur_data.X, Z=cur_data.Z, 
                                    basis_mat=_paras.SIS_basis_mat, 
                                    keep_ratio=_paras.SIS_ratio, 
                                    model_type="logi", 
                                    SIS_pen=_paras.SIS_pen, 
                                    input_paras=_paras)
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
            
        model = LogisticModel(Y=cur_data_SIS.Y, 
                              X=cur_data_SIS.X, 
                              Z=cur_data_SIS.Z, 
                              basis_mat=_paras.basis_mat)
        pen = SCAD(lams=_paras.lam, a=_paras.a,  sel_idx=_paras.sel_idx_SIS)
                    
                
        main_res = optimization(model=model, 
                                penalty=pen, 
                                inits=[alp_init, Gam_init, theta_init, rhok_init],
                                is_prg=verbose,
                                save_paras=False,    
                                input_paras=_paras)
        opt = main_res[0]
        est_beta = (_paras.basis_mat @ opt.Gamk).numpy()
        
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
        
        res._paras = _paras
        res.Q_mat_part = Q_mat_part
        res.Sig_mat_part = Sig_mat_part
        res.est_alp = est_alp
        res.est_Gam = est_Gam
        res.success = True
        
        if torch.isnan(est_alp.sum()) or torch.isnan(est_Gam.sum()):
            res.success = False
            is_cv = False
        
        if is_cv:
            cv_probs = CV_err_logi_fn(data=cur_data, 
                                     num_cv_fold=_paras.num_cv_fold,
                                     is_prg=verbose, 
                                     save_paras=False,    
                                     input_paras=_paras)
                
            res.cv_probs = cv_probs
            if np.isnan(cv_probs.sum()):
                res.success = False
        else:
            res.cv_probs = None
        
        res.Y = cur_data.Y.numpy()
        
        if is_save:
            save_pkl(_paras.save_dir/f_name, res, verbose=verbose)

    return None


# In[11]:



all_coms = itertools.product(range(0, paras.num_rep_CV), paras.can_lams, paras.can_Ns)
with Parallel(n_jobs=30) as parallel:
    ress = parallel(delayed(_run_fn)(seed, lam=lam, N=N, paras=paras, is_save=True, is_cv=True, verbose=False) 
                    for seed, lam, N 
                    in tqdm(all_coms, total=len(paras.can_Ns)*len(paras.can_lams)*paras.num_rep_CV))
