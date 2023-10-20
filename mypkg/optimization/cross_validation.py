# this file contains fns for CV
import numpy as np
from easydict import EasyDict as edict
from tqdm import trange
from models.logistic_model import LogisticModel
from models.linear_model import LinearModel
from hdf_utils.likelihood import obt_lin_tm
from optimization.opt import optimization
from penalties.scad_pen import SCAD
from hdf_utils.SIS import SIS_linear
from utils.matrix import col_vec_fn
import torch
import pdb


def CV_err_linear_fn(data, num_cv_fold, is_prg=False, save_paras=False, input_paras={}):
    """This fn is to do the cross validation for select parameters for the optimization procedure 
       for linear model, also include SIS in CV
        args:
            data: The dataset including, Y, Z, X
            num_cv_fold: Num of cross validation folds.
            input_paras: other parameters
    """
    
    # default parameter
    _paras = {
        "is_small": True, 
        "Rv": None, 
        "sigma2": 1,
        "basis_mat": None,
        'alpha': 0.9,
        'beta': 1,
        'NR_eps': 1e-05,
        'NR_maxit': 100,
        'stop_cv': 0.0005,
        'max_iter': 2000, 
        'cv_is_center': False,
        'cv_SIS_ratio': 0.2, 
        'cv_SIS_pen': 1, 
        'cv_SIS_basis_mat': None, 
        'cv_init_noise_sd': -1, 
        "linear_theta_update": "cholesky_inv",
        "linear_mat": None}
    _paras = edict(_paras)
    _paras.update(input_paras)
    
    _paras.n = data.Y.shape[0]
    
    
    num_test = int(_paras.n/num_cv_fold)
    full_idx = np.arange(_paras.n)
    test_Y_err_all = []
    if is_prg:
        prg_bar = trange(num_cv_fold)
    else:
        prg_bar = range(num_cv_fold)
    for ix in prg_bar:
        test_idx = full_idx[(ix*num_test):(ix*num_test+num_test)]
        if ix == num_cv_fold-1:
            test_idx = full_idx[(ix*num_test):] # including all remaining data
        train_idx = np.delete(full_idx, test_idx)
        
        test_set_X = data.X[test_idx]
        test_set_Y = data.Y[test_idx]
        test_set_Z = data.Z[test_idx]
        
        train_set_X = data.X[train_idx]
        train_set_Y = data.Y[train_idx]
        train_set_Z = data.Z[train_idx]
        
        if _paras.cv_is_center:
            test_set_X = test_set_X - train_set_X.mean(axis=0, keepdims=True)
            test_set_Y = test_set_Y - train_set_Y.mean(axis=0, keepdims=True)
            # Now, I do not have time to write code to center Z
            # It is a bit tedious, you should exclude intercept and categorical var
            #test_set_Z = test_set_Z - train_set_Z.mean(axis=0, keepdims=True)
            
            train_set_X = train_set_X - train_set_X.mean(axis=0, keepdims=True)
            train_set_Y = train_set_Y - train_set_Y.mean(axis=0, keepdims=True)
            #train_set_Z = train_set_Z - train_set_Z.mean(axis=0, keepdims=True)
            
        # SIS step
        if _paras.cv_SIS_ratio < 1:
            keep_idxs, _  = SIS_linear(train_set_Y, train_set_X, train_set_Z, _paras.cv_SIS_basis_mat,
                                       _paras.cv_SIS_ratio, _paras, ridge_pen=_paras.cv_SIS_pen)
        else:
            keep_idxs = _paras.sel_idx
        M_idxs = np.delete(np.arange(_paras.d), _paras.sel_idx)
        keep_idxs = np.sort(np.concatenate([M_idxs, keep_idxs]))
            
        sel_idx_SIS = np.where(np.array([keep_idx in _paras.sel_idx for keep_idx in keep_idxs]))[0]
        d_SIS = len(keep_idxs)
        pen = SCAD(lams=_paras.lam, a=_paras.a,  sel_idx=sel_idx_SIS)
        
        train_set_X = train_set_X[:, keep_idxs]
        test_set_X = test_set_X[:, keep_idxs]
        
        # initial value
        if _paras.cv_init_noise_sd < 0:
            alp_init = torch.zeros(_paras.q)
            Gam_init = torch.zeros(_paras.N, d_SIS)
            theta_init = torch.cat([alp_init, col_vec_fn(Gam_init)/np.sqrt(_paras.N)])
            rhok_init = torch.zeros(d_SIS*_paras.N) 
        else:
            alp_init = torch.Tensor(_paras.alp_GT) + torch.randn(_paras.q)*_paras.init_noise_sd
            Gam_init = torch.Tensor(_paras.Gam_GT_est[:, keep_idxs]) + torch.randn(_paras.N, d_SIS)*_paras.init_noise_sd
            theta_init = torch.cat([alp_init, col_vec_fn(Gam_init)/np.sqrt(_paras.N)])
            rhok_init = torch.randn(d_SIS*_paras.N)
        
        cur_model = LinearModel(Y=train_set_Y, X=train_set_X, Z=train_set_Z, 
                        basis_mat=_paras.basis_mat, 
                        sigma2=_paras.norminal_sigma2)
        res = optimization(model=cur_model, 
                           penalty=pen, 
                           inits=[alp_init, Gam_init, theta_init, rhok_init], 
                           is_prg=False,
                           save_paras=False,
                           input_paras=_paras)
        alp_est = res[0].alpk
        gam_est = res[0].Gamk
        test_Y_est = obt_lin_tm(test_set_Z, test_set_X, alp_est, gam_est, _paras.basis_mat)
        test_Y_err = test_set_Y - test_Y_est
        test_Y_err_all.append(test_Y_err.numpy())
    test_Y_err_all = np.concatenate(test_Y_err_all)
    if save_paras:
        return test_Y_err_all, _paras
    else:
        return test_Y_err_all

# backup, no use
'''
def CV_err_linear_fn(data, num_cv_fold, penalty, inits, is_prg=False, save_paras=False, input_paras={}):
    """This fn is to do the cross validation for select parameters for the optimization procedure 
       for linear model
        args:
            data: The dataset including, Y, Z, X
            num_cv_fold: Num of cross validation folds.
            penalty: The penalty fn, SCAD or (to be written)
            inits: Initial values of the parameters, 
                   inits = [alp_init, Gam_init, theta_init, rhok_init]
            input_paras: other parameters
    """
    
    # default parameter
    _paras = {
        "is_small": True, 
        "Rv": None, 
        "sigma2": 1,
        "basis_mat": None,
        'alpha': 0.9,
        'beta': 1,
        'NR_eps': 1e-05,
        'NR_maxit': 100,
        'stop_cv': 0.0005,
        'max_iter': 2000, 
        'cv_is_center': False,
        "linear_theta_update": "cholesky_inv",
        "linear_mat": None}
    _paras = edict(_paras)
    _paras.update(input_paras)
    
    _paras.n = data.Y.shape[0]
    
    # initial value
    alp_init, Gam_init, theta_init, rhok_init = inits
    
    
    num_test = int(_paras.n/num_cv_fold)
    full_idx = np.arange(_paras.n)
    test_Y_err_all = []
    if is_prg:
        prg_bar = trange(num_cv_fold)
    else:
        prg_bar = range(num_cv_fold)
    for ix in prg_bar:
        test_idx = full_idx[(ix*num_test):(ix*num_test+num_test)]
        if ix == num_cv_fold-1:
            test_idx = full_idx[(ix*num_test):] # including all remaining data
        train_idx = np.delete(full_idx, test_idx)
        
        test_set_X = data.X[test_idx]
        test_set_Y = data.Y[test_idx]
        test_set_Z = data.Z[test_idx]
        
        train_set_X = data.X[train_idx]
        train_set_Y = data.Y[train_idx]
        train_set_Z = data.Z[train_idx]
        
        if _paras.cv_is_center:
            test_set_X = test_set_X - train_set_X.mean(axis=0, keepdims=True)
            test_set_Y = test_set_Y - train_set_Y.mean(axis=0, keepdims=True)
            # Now, I do not have time to write code to center Z
            # It is a bit tedious, you should exclude intercept and categorical var
            #test_set_Z = test_set_Z - train_set_Z.mean(axis=0, keepdims=True)
            
            train_set_X = train_set_X - train_set_X.mean(axis=0, keepdims=True)
            train_set_Y = train_set_Y - train_set_Y.mean(axis=0, keepdims=True)
            #train_set_Z = train_set_Z - train_set_Z.mean(axis=0, keepdims=True)
        
        cur_model = LinearModel(Y=train_set_Y, X=train_set_X, Z=train_set_Z, 
                        basis_mat=_paras.basis_mat, 
                        sigma2=_paras.norminal_sigma2)
        res = optimization(model=cur_model, 
                           penalty=penalty, 
                           inits=[alp_init, Gam_init, theta_init, rhok_init], 
                           is_prg=False,
                           save_paras=False,
                           input_paras=_paras)
        alp_est = res[0].alpk
        gam_est = res[0].Gamk
        test_Y_est = obt_lin_tm(test_set_Z, test_set_X, alp_est, gam_est, _paras.basis_mat)
        test_Y_err = test_set_Y - test_Y_est
        test_Y_err_all.append(test_Y_err.numpy())
    test_Y_err_all = np.concatenate(test_Y_err_all)
    if save_paras:
        return test_Y_err_all, _paras
    else:
        return test_Y_err_all
'''
