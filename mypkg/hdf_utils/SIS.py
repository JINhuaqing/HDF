# this file contains fns for sure independent screening from fan
import numpy as np
from easydict import EasyDict as edict
import torch
#import pdb

def SIS_linear(Y, X, Z, basis_mat, keep_ratio=0.3, input_paras={}, ridge_pen=1):
    """The function is to do the sure ind screening when d (num of ROIs) is large under linear model
       Ref to Fan_and_Lv_JRSSB_2008
       args:
            Y: Response
            X: The psd 
            Z: Covariates
            keep_ratio: The ratio between the keeped rois and all rois
            basis_mat: Now SIS and main opt can use diff basis_mat (on Sep 1, 2023)
            ridge_pen: A constant added for ridge reg
            input_paras: Other parameters, 
                         require: sel_idx, q
    """
    _paras = edict(input_paras.copy())
    
    num_kp = int(np.round(len(_paras.sel_idx)*keep_ratio, 0))
    N = basis_mat.shape[1]
    
    SIS_gams = []
    for ix in _paras.sel_idx:
        cur_X = X[:, ix, :].unsqueeze(-1)
        tmp_BX = (cur_X * basis_mat).mean(axis=1)
        vec_p2 = tmp_BX*np.sqrt(N)
        vec_p = torch.cat([Z, vec_p2], dim=1)
        
        right_vec = torch.mean(vec_p * Y.unsqueeze(-1), axis=0)
        left_mat = torch.mean(vec_p.unsqueeze(-1) * vec_p.unsqueeze(1), axis=0)
        # ridge penalty
        left_mat = left_mat + torch.eye(left_mat.shape[0])*ridge_pen
        cur_gam = torch.linalg.solve(left_mat, right_vec)[_paras.q:] * np.sqrt(N)
        SIS_gams.append(cur_gam.numpy())
    SIS_gams = np.array(SIS_gams)
    SIS_betas = basis_mat.numpy() @ SIS_gams.T
    norm_vs =  np.sqrt(np.mean(SIS_betas**2, axis=0))
    keep_idxs = np.sort(np.argsort(-norm_vs)[:num_kp])
    return _paras.sel_idx[keep_idxs], norm_vs