# this file contains fns for CV
import numpy as np
from easydict import EasyDict as edict
from tqdm import trange
from models.logistic_model import LogisticModel
from models.linear_model import LinearModel
from hdf_utils.likelihood import obt_lin_tm
from optimization.opt import optimization


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
        'max_iter': 2000}
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
        
        cur_model = LinearModel(Y=train_set_Y, X=train_set_X, Z=train_set_Z, 
                        basis_mat=_paras.basis_mat, 
                        sigma2=_paras.sigma2)
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