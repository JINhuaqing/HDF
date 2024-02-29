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
from scenarios.simu_linear_sinica0 import settings



import argparse
parser = argparse.ArgumentParser(description='run')
parser.add_argument('-c', '--cs', type=float, help='cs value') 
parser.add_argument('-s', '--setting', type=str, help='Setting') 
args = parser.parse_args()

torch.set_default_dtype(torch.double)


opt_lamNs = {
"cmpn0s1":  {"0.0": (8, 1.1), "0.1": (8, 1.1), "0.2": (8, 1.1), "0.4": (8, 1.1)},
"cmpn0s1b":  {"0.0": (8, 1.1), "0.1": (8, 1.1), "0.2": (8, 1.1), "0.4": (8, 1.1)},
"cmpn0s2":  {"0.0": (6, 0.9), "0.1": (6, 0.9), "0.2": (6, 0.9), "0.4": (6, 0.9)},
"cmpn0s2b":  {"0.0": (6, 0.9), "0.1": (6, 0.9), "0.2": (6, 0.9), "0.4": (8, 1.0)},
}




# # Params

# In[6]:


np.random.seed(0)
c = args.cs

setting = settings[args.setting]
data_gen_params = setting.data_gen_params
data_gen_params.cs = data_gen_params.cs_fn(c)
data_gen_params.gt_beta = data_gen_params.beta_fn(data_gen_params.cs)
opt_lamN = opt_lamNs[args.setting][str(c)]

num_rep0 = 200
num_rep1 = 1000
n_jobs = 30
save_dir = RES_ROOT/f"simu_setting{setting.setting}_{c*1000:.0f}"
if not save_dir.exists():
    save_dir.mkdir()

pprint(setting)
print(f"Save to {save_dir}")


def _main_run_fn(seed, lam, N, setting, is_save=False, is_cv=False, verbose=2):
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
        cur_data = gen_simu_sinica_dataset(n=data_gen_params.n, 
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




with Parallel(n_jobs=n_jobs) as parallel:
    ress = parallel(delayed(_main_run_fn)(seed, lam=opt_lamN[1], N=opt_lamN[0], setting=setting, is_save=True, is_cv=False, verbose=1) 
                    for seed
                    in tqdm(range(num_rep0, num_rep1), total=(num_rep1-num_rep0)))

