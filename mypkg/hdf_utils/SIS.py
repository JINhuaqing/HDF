# this file contains fns for sure independent screening from fan
import numpy as np
from easydict import EasyDict as edict
import torch

def SIS_linear(Y, X, Z, keep_ratio=0.3, input_paras={}):
    """The function is to do the sure ind screening when d (num of ROIs) is large under linear model
       Ref to Fan_and_Lv_JRSSB_2008
       args:
            Y: Response
            X: The psd 
            Z: Covariates
            keep_ratio: The ratio between the keeped rois and all rois
            input_paras: Other parameters, 
                         require: sel_idx, basis_mat, N, q
    """
    _paras = edict(input_paras.copy())
    
    num_kp = int(np.round(len(_paras.sel_idx)*keep_ratio, 0))
    
    SIS_gams = []
    for ix in _paras.sel_idx:
        cur_X = X[:, ix, :].unsqueeze(-1)
        tmp_BX = (cur_X * _paras.basis_mat).mean(axis=1)
        vec_p2 = tmp_BX*np.sqrt(_paras.N)
        vec_p = torch.cat([Z, vec_p2], dim=1)
        
        right_vec = torch.sum(vec_p * Y.unsqueeze(-1), axis=0)
        left_mat = torch.sum(vec_p.unsqueeze(-1) * vec_p.unsqueeze(1), axis=0)
        cur_gam = torch.linalg.solve(left_mat, right_vec)[_paras.q:] * np.sqrt(_paras.N)
        SIS_gams.append(cur_gam.numpy())
    SIS_gams = np.array(SIS_gams)
    SIS_betas = _paras.basis_mat.numpy() @ SIS_gams.T
    norm_vs =  np.sqrt(np.mean(SIS_betas**2, axis=0))
    keep_idxs = np.sort(np.argsort(-norm_vs)[:num_kp])
    return _paras.sel_idx[keep_idxs], norm_vs
