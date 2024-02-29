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
from scenarios.simu_linear_sinica import settings



import argparse
parser = argparse.ArgumentParser(description='run')
parser.add_argument('-s', '--setting', type=str, help='Setting') 
args = parser.parse_args()

torch.set_default_dtype(torch.double)






# # Params

# In[6]:


np.random.seed(0)
setting = settings[args.setting]
num_rep = 200
n_jobs = 30


def _get_valset_metric_fn(res):
    valsel_metrics = edict()
    valsel_metrics.mse_loss = np.mean((res.cv_Y_est- res.Y.numpy())**2);
    valsel_metrics.mae_loss = np.mean(np.abs(res.cv_Y_est-res.Y.numpy()));
    valsel_metrics.cv_Y_est = res.cv_Y_est
    valsel_metrics.tY = res.Y.numpy()
    return valsel_metrics
def _run_fn_extract(seed, N, lam, c):
    f_name = f"seed_{seed:.0f}-lam_{lam*1000:.0f}-N_{N:.0f}_fit.pkl"
    res = load_pkl(save_dir/f_name, verbose=0)
    return (seed, N, lam), _get_valset_metric_fn(res)

for c in [0, 0.1]:
#for c in [0, 0.1, 0.2, 0.4]:
    save_dir = RES_ROOT/f"simu_setting{setting.setting}_{c*1000:.0f}"
    all_coms = itertools.product(range(0, num_rep), setting.can_lams, setting.can_Ns)
    with Parallel(n_jobs=n_jobs) as parallel:
        all_cv_errs_list = parallel(delayed(_run_fn_extract)(cur_seed, cur_N, cur_lam, c=c)  for cur_seed, cur_lam, cur_N in tqdm(all_coms, total=num_rep*len(setting.can_Ns)*len(setting.can_lams), desc=f"c: {c}"))
    all_cv_errs = {res[0]:res[1] for res in all_cv_errs_list};
    save_pkl(save_dir/f"all-valsel-metrics.pkl", all_cv_errs, is_force=1)
