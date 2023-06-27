#!/usr/bin/env python
# coding: utf-8

# This file contains python code for convergence test under linear setting

# In[1]:


import sys
sys.path.append("../mypkg")


# In[2]:


import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
from scipy.stats import pearsonr
from numbers import Number
import multiprocessing as mp

from easydict import EasyDict as edict
from tqdm import trange
from scipy.io import loadmat
from pprint import pprint
import argparse

plt.style.use('ggplot')
plt.rcParams["savefig.bbox"] = "tight"
# to solve possible error from mp
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


# In[3]:


import importlib
import optimization.opt
importlib.reload(optimization.opt)


# In[4]:


from constants import DATA_ROOT, RES_ROOT, FIG_ROOT
from default_paras import def_paras

from hdf_utils.data_gen import gen_covs, gen_simu_psd
from hdf_utils.fns import fn1, fn2, fn3, fn4, fn5, zero_fn
from hdf_utils.fns_sinica import coef_fn, fourier_basis_fn
from hdf_utils.likelihood import obt_lin_tm
from hdf_utils.SIS import SIS_linear
from utils.matrix import col_vec_fn, col_vec2mat_fn, conju_grad, svd_inverse
from utils.functions import logit_fn
from utils.misc import save_pkl
from splines import obt_bsp_basis_Rfn, obt_bsp_basis_Rfn_wrapper
from projection import euclidean_proj_l1ball
from optimization.one_step_opt import OneStepOpt
from optimization.cross_validation import CV_err_linear_fn
from optimization.opt import optimization

from penalties.scad_pen import SCAD
from models.logistic_model import LogisticModel
from models.linear_model import LinearModel


# In[5]:


torch.set_default_tensor_type(torch.DoubleTensor)


# In[ ]:

parser = argparse.ArgumentParser(description="Input args")
parser.add_argument("-n", type=int, default=2700, help="The sample size")
parser.add_argument("--num_rep", type=int, default=10, help="Num of repetitions")

args = parser.parse_args()




# ## Param and fns

# ### Params

# In[6]:


np.random.seed(0)
paras = edict(def_paras.copy())

# Others
paras.num_rep = args.num_rep
paras.ns = [100, 300, 900, 2700, 8100, 24300]
paras.can_lams = [2e-1, 3e-1, 5e-1, 0.75, 1e0, 1.25, 1.5, 2, 2.5, 3e0, 9e0]
paras.ind_freq = np.linspace(1, 40, 40) # freqs
paras.init_noise_sd = 5 # the sd of the noise added to the true value for initial values
paras.max_iter = 5000 # maximal iteration number
paras.SIS_ratio = 0.1 # the ratio to keep with SIS procedure

# generating dataset
paras.n = args.n # num of data obs to be genareted
paras.npts = 40 # num of pts to evaluate X(s)
paras.d = 200 # num of ROIs
paras.q = 5 # num of other covariates
paras.sigma2 = 1 # variance of the error
paras.types_ = ["int", 2, 2, "c", "c"]

# b-spline
def tmp_fn(n):
    ns = [100, 300, 900, 2700, 8100, 24300]
    Ns = [7, 8, 9, 12, 13, 15]
    return Ns[ns.index(n)]
paras.N = tmp_fn(paras.n)
#paras.N = int(np.round(3.3*paras.n**(1/paras.ord/2), 0)) # num of basis for bsp
paras.can_Ns = [paras.N]
paras.x = np.linspace(0, 1, paras.npts)
paras.basis_mat = torch.DoubleTensor(obt_bsp_basis_Rfn_wrapper(paras.x, paras.N, paras.ord))
assert paras.N == paras.basis_mat.shape[1]
print(f"The number of B-spline basis is {paras.N:.0f}.")

# True parameters
paras.alp_GT = np.array([5, 1, -2, 3, -4])

# first way
#beta_type_GT = [fn1, fn2, fn2] + [zero_fn]*(paras.d-3-2) + [fn1, fn2]
#paras.beta_GT = np.array([_fn(paras.x) for _fn in beta_type_GT]).T
# second way, fourier basis
paras.fourier_basis = fourier_basis_fn(paras.x)
paras.fourier_basis_coefs = ([coef_fn(0.1), coef_fn(0.3), coef_fn(0.5)] + 
                             [np.zeros(50)] * (paras.d-3-2) +
                             [coef_fn(0.7), coef_fn(0.9)]
                             )
paras.fourier_basis_coefs = np.array(paras.fourier_basis_coefs).T * 5
paras.beta_GT = paras.fourier_basis @ paras.fourier_basis_coefs
# 
paras.Gam_est = (np.linalg.inv(paras.basis_mat.numpy().T 
                               @ paras.basis_mat.numpy()) 
                 @ paras.basis_mat.numpy().T 
                 @ paras.beta_GT)

# optimization
paras.Rmin = 2*(np.linalg.norm(paras.Gam_est/np.sqrt(paras.N), axis=0).sum() + np.abs(paras.alp_GT).sum())
paras.sel_idx = np.arange(0, paras.d) # M^c set, for convergence, I penalize all beta fns
paras.num_cv_fold = 5
paras.Rfct = 2


# In[7]:


paras.save_dir = RES_ROOT/"linear_SIS_100_Nmanual_penalize_all"
if not paras.save_dir.exists():
    paras.save_dir.mkdir()


# In[ ]:





# ### Fns

# In[8]:


def _gen_simu_data_all(seed, paras):
    np.random.seed(seed)
    _paras = edict(paras.copy())
    # simulated PSD
    assert len(_paras.types_) == _paras.q
    assert len(_paras.alp_GT) == _paras.q
    
    simu_psd = gen_simu_psd(_paras.n, _paras.d, 10)
    simu_covs = gen_covs(_paras.n, _paras.types_)
    
    # linear term and Y
    int_part = np.sum(_paras.beta_GT.T* simu_psd[:, :, :], axis=1).mean(axis=1)
    cov_part = simu_covs @ _paras.alp_GT 
    
    # linear term
    lin_term = cov_part + int_part
    
    # Y 
    rvs = np.random.rand(_paras.n)
    Y = lin_term + np.random.randn(_paras.n)*np.sqrt(_paras.sigma2)
    
    # To torch
    X = torch.tensor(simu_psd) # n x d x npts
    Z = torch.tensor(simu_covs) # n x q
    Y = torch.tensor(Y)
    
    all_data = edict()
    all_data.X = X
    all_data.Y = Y
    all_data.Z = Z
    return all_data


# In[ ]:





# ## Simu

# In[9]:


def _run_fn(seed, lam, N, paras):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    _paras = edict(paras.copy())
    _paras.lam = lam
    _paras.N = N
    _paras.basis_mat = torch.DoubleTensor(obt_bsp_basis_Rfn_wrapper(_paras.x, _paras.N, _paras.ord))
    _paras.Gam_est = (np.linalg.inv(_paras.basis_mat.numpy().T 
                               @ _paras.basis_mat.numpy()) 
                               @ _paras.basis_mat.numpy().T 
                               @ _paras.beta_GT)
    _paras.Rmin = 2*(np.linalg.norm(_paras.Gam_est/np.sqrt(_paras.N), axis=0).sum() + np.abs(_paras.alp_GT).sum())
    _paras.seed = seed
    _paras.Rv = _paras.Rfct * _paras.Rmin
    print((_paras.seed, lam, N), "\n")
    
    
    cur_data = _gen_simu_data_all(_paras.seed, _paras)

    # do sure independent screening for dim reduction
    if _paras.SIS_ratio < 1:
        keep_idxs, _  = SIS_linear(cur_data.Y, cur_data.X, cur_data.Z, _paras.SIS_ratio, _paras)
    else:
        keep_idxs = np.arange(_paras.d)
        
    _paras.keep_idxs = np.sort(keep_idxs)
    _paras.sel_idx_SIS = np.where(np.array([keep_idx in _paras.sel_idx for keep_idx in _paras.keep_idxs]))[0]
    _paras.d_SIS = len(_paras.keep_idxs)
    
    cur_data_SIS = edict(cur_data.copy())
    cur_data_SIS.X = cur_data.X[:, _paras.keep_idxs, :]
    
    
    alp_init = torch.tensor(_paras.alp_GT) + torch.randn(_paras.q)*_paras.init_noise_sd
    Gam_init = (torch.tensor(_paras.Gam_est[:, _paras.keep_idxs]) 
                + torch.randn(_paras.N, _paras.d_SIS)*_paras.init_noise_sd)
    theta_init = torch.cat([alp_init, col_vec_fn(Gam_init)/np.sqrt(_paras.N)])
    rhok_init = torch.randn(_paras.d_SIS*_paras.N)
        
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
                            is_prg=False,
                            save_paras=False,    
                            input_paras=_paras)
    
    cv_errs = CV_err_linear_fn(data=cur_data_SIS, 
                               penalty=pen, 
                               num_cv_fold=_paras.num_cv_fold,
                               inits=[alp_init, Gam_init, theta_init, rhok_init], 
                               is_prg=False, 
                               save_paras=False,    
                               input_paras=_paras)
    return main_res, cv_errs, _paras

# In[31]:


num_core = 30
if __name__ == "__main__":
    for cur_lam in paras.can_lams:
        for cur_N in paras.can_Ns:
            fil_name = f"result_lam-{cur_lam*100:.0f}_N-{cur_N:.0f}_n-{paras.n:.0f}.pkl"
            with mp.Pool(num_core) as pool:
                res_proc = []
                for seed in range(paras.num_rep):
                    res_proc.append(pool.apply_async(_run_fn, (seed, cur_lam, cur_N, paras)))
                opt_results = [ix.get() for ix in res_proc]
            pool.join()
            save_pkl(paras.save_dir/fil_name, opt_results)


# In[10]:

