import numpy as np
from easydict import EasyDict as edict

def _err_t_fn(n, df=3, sigma2=1):
    errs_raw = np.random.standard_t(df=df, size=n)
    errs = np.sqrt(sigma2)*(errs_raw - errs_raw.mean())/errs_raw.std()
    return errs

def _err_norm_fn(n, sigma2=1):
    errs = np.random.randn(n)*np.sqrt(sigma2)
    return errs
    


setting10 = edict({
    "setting": "10", 
    "sel_idx": np.arange(1, 68),
    "cs_fn": lambda c: [c, 0, 0],
    "err_fn": lambda n: _err_norm_fn(n, 1),
    "can_Ns": [4, 6, 8],
    "can_lams": [0.001, 0.2, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1, 2, 8],
    "n": 200, 
    "d": 68,
    "SIS_ratio": 0.2, 
    "beta": 1.2,
    "cs": [0.0, 0.2, 0.4], 
    "num_cv_fold": 5
 })


setting10a = edict({
    "setting": "10a", 
    "sel_idx": np.arange(1, 68),
    "cs_fn": lambda c: [c, 0, 0],
    "err_fn": lambda n: _err_norm_fn(n, 1),
    "cs": [0.0, 0.2, 0.4], 
    "can_Ns": [4, 6, 8, 10, 12],
    "can_lams": [0.001, 0.2, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1, 2, 8],
    "n": 200, 
    "d": 68,
    "SIS_ratio": 0.2, 
    "beta": 10,
    "num_cv_fold": 10
 })


setting10c = edict({
    "setting": "10c", 
    "sel_idx": np.arange(2, 68),
    "cs_fn": lambda c: [c+0.5, 0.5, 0],
    "err_fn": lambda n: _err_norm_fn(n, 1),
    "cs": [0.0, 0.2, 0.4], 
    "can_Ns": [4, 6, 8],
    "can_lams": [0.001, 0.1, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1, 1.1, 1.2, 2, 8],
    "n": 200, 
    "d": 68,
    "SIS_ratio": 0.2, 
    "beta": 1.2,
    "num_cv_fold": 5
 })

setting10d = edict({
    "setting": "10d", 
    "sel_idx": np.arange(3, 68),
    "cs_fn": lambda c: [c, c, c],
    "err_fn": lambda n: _err_norm_fn(n, 1),
    "cs": [0.0, 0.2, 0.4], 
    "can_Ns": [4, 6, 8],
    "can_lams": [0.001, 0.1, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1, 2, 8],
    "n": 200, 
    "d": 68,
    "SIS_ratio": 0.2, 
    "beta": 1.2,
    "num_cv_fold": 5
 })

setting10e = edict({
    "setting": "10e", 
    "sel_idx": np.arange(1, 68),
    "cs_fn": lambda c: [c, 0, 0],
    "err_fn": lambda n: _err_t_fn(n, df=3, sigma2=1),
    "cs": [0.0, 0.2, 0.4], 
    "can_Ns": [4, 6, 8, 10, 12],
    "can_lams": [0.001, 0.1, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.1, 1.2, 2, 8],
    "n": 200, 
    "d": 68,
    "SIS_ratio": 0.2, 
    "beta": 10,
    "num_cv_fold": 10
 })

setting10f = edict({
    "setting": "10f", 
    "sel_idx": np.arange(2, 68),
    "cs_fn": lambda c: [c+0.5, 0.5, 0],
    "err_fn": lambda n: _err_t_fn(n, df=3, sigma2=1),
    "cs": [0.0, 0.2, 0.4], 
    "can_Ns": [4, 6, 8, 10, 12],
    "can_lams": [0.001, 0.1, 0.6, 0.7, 0.8,  0.9, 0.95, 1, 1.05, 1.1, 1.2, 2, 8],
    "n": 200, 
    "d": 68,
    "SIS_ratio": 0.2, 
    "beta": 10,
    "num_cv_fold": 10
 })

setting10g = edict({
    "setting": "10g", 
    "sel_idx": np.arange(3, 68),
    "cs_fn": lambda c: [c, c, c],
    "err_fn": lambda n: _err_t_fn(n, df=3, sigma2=1),
    "cs": [0.0, 0.2, 0.4], 
    "can_Ns": [4, 6, 8, 10, 12],
    "can_lams": [0.001, 0.1, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1, 2, 8],
    "n": 200, 
    "d": 68,
    "SIS_ratio": 0.2, 
    "beta": 10,
    "num_cv_fold": 10
 })