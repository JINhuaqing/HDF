#!/usr/bin/env python
# coding: utf-8

# This file contains python code to compare with sinica paper
# 
# It is under the linear setting
# 
# Now, I use the same X and beta from the paper (on Sep 4, 2023)
# 
# I analyze the results under new code, here I save the results seperately.

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



from constants import DATA_ROOT, RES_ROOT, FIG_ROOT, MIDRES_ROOT

from hdf_utils.hypo_test import obt_test_stat
from utils.matrix import col_vec_fn, col_vec2mat_fn, conju_grad, svd_inverse, cholesky_inv
from utils.misc import save_pkl, load_pkl, get_local_min_idxs
from optimization.one_step_opt import OneStepOpt

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


# In[6]:


plt.style.use(FIG_ROOT/"base.mplstyle")
torch.set_default_tensor_type(torch.DoubleTensor)


# In[ ]:





# In[ ]:





# # Param and fns

# In[ ]:





# ## Params

# In[7]:


paras = edict()
paras.save_dir = RES_ROOT/"simu_linear_sinica_samebetaX_test1"
if not paras.save_dir.exists():
    paras.save_dir.mkdir()


# ## Fns

# In[8]:


def _get_min_idx(x):
    """Get the index of the minimal values among the local minimals.
       If there are multiple ones, return the largest index
       args:
           x: a vec
        
    """
    x = np.array(x)
    lmin_idxs = get_local_min_idxs(x);
    if len(lmin_idxs) == 0:
        lmin_idxs = np.arange(len(x))
    lmin_idxs_inv =  lmin_idxs[::-1]
    lmins_inv = x[lmin_idxs_inv];
    return  lmin_idxs_inv[np.argmin(lmins_inv)]
_err_fn = lambda x: np.nanmean(x**2)


# In[ ]:







# # Analysis

# In[9]:


can_Ns = [4, 6, 8, 10, 12]
c1s = [0, 0.1, 0.2, 0.4]
can_lams = [0.01, 0.1, 0.2, 0.3, 0.4, 0.6, 1, 2, 8];


# In[ ]:





# ## Step 1 (only need once)

# In[18]:


for c1 in cs:
    opt_lamNs = []
    for cur_seed in trange(200, desc=f"c1: {c1*1000:.0f}"):
        errs_N = []
        for cur_N in can_Ns:
            errs = []
            for cur_lam in can_lams:
                f2_fil = list(paras.save_dir.glob(f"seed_{cur_seed:.0f}-lam_{cur_lam*1000:.0f}-N_{cur_N:.0f}-c1_{c1*1000:.0f}_cv.pkl"))[0]
                t_res2 = load_pkl(f2_fil, verbose=0)
                errs.append(_err_fn(t_res2.cv_errs))
            min_idx = _get_min_idx(errs);
            errs_N.append((cur_seed, cur_N, can_lams[min_idx], errs[min_idx]))
            errs_N_sorted = sorted(errs_N, key=lambda x:x[-1]);
        opt_lamNs.append(errs_N_sorted[0])
    save_pkl(paras.save_dir/f"opt_lamNs_200_c1_{c1*1000:.0f}.pkl", opt_lamNs)


# ## Step 2 (only need once)

# In[24]:


def _test_fn(Cmat, res1,  rtols=[0, 0], is_verbose=False):
    torch.set_default_tensor_type(torch.DoubleTensor)
    _paras = res1._paras
    
    est_Gam_full = torch.zeros_like(torch.tensor(_paras.Gam_GT_est)).to(torch.get_default_dtype());
    opt = res1.main_res[0]
        
    est_Gam = opt.Gamk
    est_alp = opt.alpk
    est_theta = torch.cat([est_alp, col_vec_fn(est_Gam)/np.sqrt(_paras.N)])
    est_Gam_full[:, _paras.keep_idxs] = est_Gam.clone()
    _paras.svdinv_eps_Q = rtols[0]
    _paras.svdinv_eps_Psi = rtols[1]
    T_v = obt_test_stat(res1.model, est_alp, est_Gam, Cmat, _paras).item() 
    pval = chi2.sf(T_v, Cmat.shape[0]*_paras.N)
    
    res3 = edict()
    res3.T_v = T_v
    res3.pval = pval
    res3.Cmat = Cmat
    #if is_save:
    #    save_pkl(_paras.save_dir/f3_name, res3, verbose=is_verbose)
    return res3.T_v, res3.pval



# In[25]:


Cmat = np.eye(2) # change it depending on test1, test2, test3
cans = [1e-5, 3e-6, 1e-6, 3e-7, 1e-7, 3e-8, 1e-8, 3e-9, 1e-9, 3e-10, 0]
all_coms = list(itertools.product(cans, cans))
for c1 in [0, 0.1, 0.2, 0.4]:
    opt_lamNs =  load_pkl(paras.save_dir/f"opt_lamNs_200_c1_{c1*1000:.0f}.pkl");
    ress = []
    for cur_seed, cur_N, cur_lam, _ in tqdm(opt_lamNs, desc=f"c1: {c1*1000:.0f}"):
        f1_name = f"seed_{cur_seed:.0f}-lam_{cur_lam*1000:.0f}-N_{cur_N:.0f}-c1_{c1*1000:.0f}_est.pkl"
        res1 = load_pkl(paras.save_dir/f1_name, verbose=False);
        res = []
        with Parallel(n_jobs=20) as parallel:
            res = parallel(delayed(_test_fn)(Cmat=Cmat, res1=res1, rtols=rtols) for rtols
                            in all_coms)
    
        ress.append(res)
    ress = np.array(ress);
    if not (paras.save_dir/f"opt_lamNs_200_c1_{c1*1000:.0f}_ress.pkl").exists():
        save_pkl(paras.save_dir/f"opt_lamNs_200_c1_{c1*1000:.0f}_ress.pkl", ress)






