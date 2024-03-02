#!/usr/bin/env python
# coding: utf-8

# This file contains python code to mimic real setting
# 
# It is under the linear setting
# 
# Now, I use the same beta and X from the paper 

# In[1]:


import sys
sys.path.append("../mypkg")

import numpy as np
import torch
import itertools
from easydict import EasyDict as edict
from tqdm import tqdm
from pprint import pprint
from joblib import Parallel, delayed

from constants import RES_ROOT 
from hdf_utils.data_gen import gen_simu_sinica_dataset
from utils.misc import save_pkl, load_pkl
from optimization.opt import HDFOpt

import argparse
parser = argparse.ArgumentParser(description='run')
parser.add_argument('--n', type=int, help='samplesize') 
args = parser.parse_args()



torch.set_default_dtype(torch.double)

from easydict import EasyDict as edict
from hdf_utils.fns_sinica import coef_fn, fourier_basis_fn
from copy import deepcopy
from scenarios.base_params import get_base_params

base_params = get_base_params("linear") 
base_params.data_gen_params = edict()
base_params.data_gen_params.d = 200 # num of ROIs
base_params.data_gen_params.q = 3 # num of other covariates
base_params.data_gen_params.npts = 100 # num of pts to evaluate X(s)
base_params.data_gen_params.types_ = ["int", "c", 2]
base_params.data_gen_params.gt_alp = np.array([5, -1, 2]) 
base_params.data_gen_params.data_params={"sigma2":1, "srho":0.3, "basis_type":"bsp"}
base_params.data_gen_params.data_type = base_params.model_type
base_params.can_Ns = [4, 6, 8, 10, 12, 14]
base_params.SIS_params = edict({"SIS_pen": 0.02, "SIS_basis_N":8, "SIS_ws":"simpson"})
base_params.opt_params.beta = 10 
base_params.can_lams = [0.60,  0.80,  1,  1.2, 1.4, 1.6, 2.0, 4.0]
base_params.can_lams = np.sort([0.60,  0.80,  1,  1.2, 1.4, 1.6, 2.0, 4.0] + [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5])

def _get_gt_beta_wrapper(d, fct=2):
    def _get_gt_beta(cs):
        x = np.linspace(0, 1, 100)
        fourier_basis = fourier_basis_fn(x)
        fourier_basis_coefs = ([cs[0]*coef_fn(0.2), cs[1]*coef_fn(0.2), cs[2]*coef_fn(0.2)] + 
                                     [np.zeros(50)] * (d-3-1) +
                                     [coef_fn(0.2)]
                                     )
        fourier_basis_coefs = np.array(fourier_basis_coefs).T 
        gt_beta = fourier_basis @ fourier_basis_coefs * fct
        return gt_beta
    return _get_gt_beta



setting = edict(deepcopy(base_params))
add_params = edict({})
add_params.data_gen_params = edict(deepcopy(base_params.data_gen_params))
add_params.data_gen_params.ns = [100, 300, 900, 2700, 8100]
add_params.data_gen_params.cs = [0, 0, 0]
add_params.data_gen_params.npts = 100 # num of pts to evaluate X(s)
add_params.data_gen_params.data_params["err_dist"] = "normal"
beta_fn = _get_gt_beta_wrapper(add_params.data_gen_params.d, 
                                                          fct=2)
add_params.data_gen_params.gt_beta = beta_fn(add_params.data_gen_params.cs)

add_params.setting = "alpconv"
add_params.sel_idx =  np.arange(0, add_params.data_gen_params.d)
add_params.SIS_ratio = 0.2
setting.update(add_params)

opt_lamNs = {
100: (12, 1.4),
200: (12, 1.0),
400: (10, 0.6),
800: (10, 0.5),
1600: (10, 0.4),
3200: (10, 0.3),
}


data_gen_params = setting.data_gen_params
x = np.linspace(0, 1, data_gen_params.npts)
opt_lamN = opt_lamNs[args.n]


num_rep = 1000
num_rep0 = 200
n_jobs = 30
save_dir = RES_ROOT/f"simu_alpconv_n{args.n}"
if not save_dir.exists():
    save_dir.mkdir()

pprint(setting)
print(f"Save to {save_dir}")


def _main_run_fn(seed, n, lam, N, setting, is_save=False, is_cv=False, verbose=2):
    """Now (on Aug 25, 2023), if we keep seed the same, the cur_data is the same. 
       If you want to make any changes, make sure this. 
    """
    torch.set_default_dtype(torch.double)
    np.random.seed(seed)
    torch.manual_seed(seed)
        
    _setting = edict(setting.copy())
    _setting.seed = seed
    _setting.lam = lam
    _setting.N = N
    
    data_gen_params = setting.data_gen_params
    x = np.linspace(0, 1, data_gen_params.npts)
    
    f_name = f"seed_{seed:.0f}-lam_{lam*1000:.0f}-N_{N:.0f}_fit.pkl"
    
    
    if not (save_dir/f_name).exists():
        cur_data = gen_simu_sinica_dataset(n=n, 
                                   d=data_gen_params.d, 
                                   q=data_gen_params.q, 
                                   types_=data_gen_params.types_, 
                                   gt_alp=data_gen_params.gt_alp, 
                                   gt_beta=data_gen_params.gt_beta, 
                                   x = x,
                                   data_type=data_gen_params.data_type,
                                   data_params=data_gen_params.data_params, 
                                   seed=seed, 
                                   verbose=verbose);
        hdf_fit = HDFOpt(lam=_setting.lam, 
                         sel_idx=_setting.sel_idx, 
                         model_type=_setting.model_type,
                         verbose=verbose, 
                         SIS_ratio=_setting.SIS_ratio, 
                         N=_setting.N,
                         is_std_data=True, 
                         cov_types=None, 
                         inits=None,
                         model_params = _setting.model_params, 
                         SIS_params = _setting.SIS_params, 
                         opt_params = _setting.opt_params,
                         bsp_params = _setting.bsp_params, 
                         pen_params = _setting.pen_params
               );
        hdf_fit.add_data(cur_data.X, cur_data.Y, cur_data.Z)
        opt_res = hdf_fit.fit()
        
        if is_cv:
            hdf_fit.get_cv_est(_setting.num_cv_fold)
        if is_save:
            hdf_fit.save(save_dir/f_name, is_compact=True, is_force=True)
    else:
        hdf_fit = load_pkl(save_dir/f_name, verbose>=2);
        
    return None




N, lam = opt_lamN
with Parallel(n_jobs=n_jobs) as parallel:
    ress = parallel(delayed(_main_run_fn)(seed, n=args.n, lam=lam, N=N, setting=setting, is_save=True, is_cv=False, verbose=1) 
                    for seed
                    in tqdm(range(num_rep0, num_rep), total=num_rep-num_rep0))

