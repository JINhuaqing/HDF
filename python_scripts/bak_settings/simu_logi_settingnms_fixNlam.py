#!/usr/bin/env python
# coding: utf-8

# This file contains python code to mimic real setting
# 
# It is under the logi setting
# 
# Now, I use the same beta and X from MEG

# In[1]:


import sys
sys.path.append("../mypkg")


# In[2]:


import numpy as np
import torch
import itertools
from easydict import EasyDict as edict
from tqdm import  tqdm
from pprint import pprint
from joblib import Parallel, delayed


from constants import DATA_ROOT, RES_ROOT, FIG_ROOT, MIDRES_ROOT
from hdf_utils.data_gen import gen_simu_meg_dataset
from utils.misc import save_pkl, load_pkl, bcross_entropy_loss
from optimization.opt import HDFOpt
from scenarios.real_simu_logi_meg import settings


import argparse
parser = argparse.ArgumentParser(description='run')
parser.add_argument('-c', '--cs', type=float, help='cs value') 
parser.add_argument('-s', '--setting', type=str, help='Setting') 
args = parser.parse_args()

torch.set_default_dtype(torch.double)

opt_lamNs = {
"nm1" : {"0.0": (6, 0.2), "0.1": (6, 0.2), "0.2": (6, 0.2), "0.4": (6, 0.2)}, 
"nm1a": {"0.0": (12, 0.2), "0.1": (12, 0.2), "0.2": (12, 0.2), "0.4": (12, 0.2)}, 
"nm2" : {"0.0": (6, 0.2), "0.1": (6, 0.2), "0.2": (6, 0.2), "0.4": (6, 0.2)}, 
"nm2a": {"0.0": (8, 0.2), "0.1": (8, 0.2), "0.2": (8, 0.2), "0.4": (8, 0.2)}, 
}


np.random.seed(0)
c = args.cs

setting = settings[args.setting]
data_gen_params = setting.data_gen_params
data_gen_params.cs = data_gen_params.cs_fn(c)
data_gen_params.gt_beta = data_gen_params.beta_fn(data_gen_params.cs)
opt_lamN = opt_lamNs[args.setting][str(c)]

AD_ts = load_pkl(DATA_ROOT/"AD_vs_Ctrl_ts/AD88_all.pkl")
Ctrl_ts = load_pkl(DATA_ROOT/"AD_vs_Ctrl_ts/Ctrl92_all.pkl")
ts_data = np.concatenate([AD_ts, Ctrl_ts], axis=0)
stds = ts_data.std(axis=(1, 2));
ts_data_filter = ts_data[np.sort(np.where(stds>100)[0])];

num_rep0 = 200
num_rep1 = 1000
n_jobs = 30
Cmat = np.eye(data_gen_params.d - len(setting.sel_idx))
save_dir = RES_ROOT/f"simu_logi_setting{setting.setting}_{c*1000:.0f}"
if not save_dir.exists():
    save_dir.mkdir()



def _get_logi_int(data_gen_params, n_jobs=30, num_rep=100):
    ress = []
    for inte in tqdm(data_gen_params.intercept_cans):
        gt_alp = np.concatenate([[inte], data_gen_params.gt_alp0])
        def _tmp_fn(seed, data_gen_params=data_gen_params):
            data = gen_simu_meg_dataset(n=data_gen_params.n, 
                                        q=data_gen_params.q, 
                                        types_=data_gen_params.types_, 
                                        gt_alp=gt_alp, 
                                        gt_beta=data_gen_params.gt_beta, 
                                        npts=data_gen_params.npts, 
                                        base_data=ts_data_filter,
                                        data_type=data_gen_params.data_type,
                                        data_params=data_gen_params.data_params, 
                                        seed=seed, 
                                        verbose=1);
            return data.Y.numpy()
        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(_tmp_fn)(seed) for seed in range(num_rep))
        ress.append(np.array(res))


    # get the intercept
    Yms = np.array([res.mean() for res in ress])
    intercept = data_gen_params.intercept_cans[np.argmin(np.abs(Yms-0.5))]
    print(f"The mean of Y is {Yms[np.argmin(np.abs(Yms-0.5))]:.3f} under intercept {intercept:.3f}.")
    gt_alp = np.concatenate([[intercept], data_gen_params.gt_alp0])
    return gt_alp


# In[8]:


data_gen_params.gt_alp = _get_logi_int(data_gen_params, n_jobs=n_jobs);
pprint(setting)
print(f"Save at {save_dir}")


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
    
    f_name = f"seed_{seed:.0f}-lam_{lam*1000:.0f}-N_{N:.0f}_fit.pkl"
    
    
    if not (save_dir/f_name).exists():
        cur_data = gen_simu_meg_dataset(n=data_gen_params.n, 
                                   q=data_gen_params.q, 
                                   types_=data_gen_params.types_, 
                                   gt_alp=data_gen_params.gt_alp, 
                                   gt_beta=data_gen_params.gt_beta, 
                                   npts=data_gen_params.npts, 
                                   base_data=ts_data_filter,
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









