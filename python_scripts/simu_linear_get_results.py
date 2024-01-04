#!/usr/bin/env python
# coding: utf-8


import sys
sys.path.append("../mypkg")


import numpy as np
from scipy.stats import pearsonr
from numbers import Number

from easydict import EasyDict as edict
from tqdm import trange, tqdm
from pprint import pprint
import itertools

from constants import DATA_ROOT, RES_ROOT, FIG_ROOT, MIDRES_ROOT
from utils.misc import save_pkl, load_pkl
from joblib import Parallel, delayed



num_rep_CV = 200 
can_lams = [0.001, 0.2, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1, 2, 8]
can_Ns = [4, 6, 8, 10, 12]
c1s = [0.0, 0.2, 0.4]
setting = "10a"





for c1 in c1s:
    save_dir = RES_ROOT/f"simu_setting{setting}_{c1*1000:.0f}"
    all_coms = itertools.product(range(0, num_rep_CV), can_lams, can_Ns)
    assert save_dir.exists()
    def _run_fn(seed, N, lam, c1):
        f_fil = list(save_dir.glob(f"seed_{seed:.0f}-lam_{lam*1000:.0f}-N_{N:.0f}-c1_{c1*1000:.0f}_est.pkl"))[0]
        t_res = load_pkl(f_fil, verbose=0)
        valsel_metrics = edict()
        valsel_metrics.cv_errs = t_res.cv_errs
        valsel_metrics.AIC = t_res.AIC
        valsel_metrics.BIC = t_res.BIC
        valsel_metrics.GCV = t_res.GCV
        return (seed, N, lam), valsel_metrics
    with Parallel(n_jobs=30) as parallel:
        all_cv_errs_list = parallel(delayed(_run_fn)(cur_seed, cur_N, cur_lam, c1=c1)  
                                     for cur_seed, cur_lam, cur_N
                                     in tqdm(all_coms, total=num_rep_CV*len(can_Ns)*len(can_lams), 
                                                          desc=f"c1: {c1}"))
    all_cv_errs = {res[0]:res[1] for res in all_cv_errs_list};
    save_pkl(save_dir/f"all-valsel-metrics_c1_{c1*1000:.0f}.pkl", all_cv_errs, is_force=1)
