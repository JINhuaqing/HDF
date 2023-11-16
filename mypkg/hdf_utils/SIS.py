# this file contains fns for sure independent screening from fan
import numpy as np
from easydict import EasyDict as edict
import torch
from rpy2 import robjects as robj
import pdb

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
        #pdb.set_trace()
        # ridge penalty
        left_mat = left_mat + torch.eye(left_mat.shape[0])*ridge_pen
        cur_gam = torch.linalg.solve(left_mat, right_vec)[_paras.q:] * np.sqrt(N)
        #cur_gam = conju_grad(left_mat, right_vec)[_paras.q:] * np.sqrt(N)
        SIS_gams.append(cur_gam.numpy())
    SIS_gams = np.array(SIS_gams)
    SIS_betas = basis_mat.numpy() @ SIS_gams.T
    norm_vs =  np.sqrt(np.mean(SIS_betas**2, axis=0))
    keep_idxs = np.sort(np.argsort(-norm_vs)[:num_kp])
    return _paras.sel_idx[keep_idxs], norm_vs

def SIS_ballcor(Y, X, sel_idx, keep_ratio=0.3):
    """
    Perform SIS (Sure Independence Screening) using ball correlation.
    It can be used for any regressions in GLIM

    Parameters:
    -----------
    Y : numpy.ndarray or torch.Tensor
        The response variable.
    X : numpy.ndarray or torch.Tensor
        The predictor variables.
    sel_idx : list or numpy.ndarray
        The indices of the selected variables.
    keep_ratio : float, optional (default=0.3)
        The ratio of variables to keep after screening.

    Returns:
    --------
    numpy.ndarray
        The selected variables.
    numpy.ndarray
        The ball cors
    """
    if isinstance(Y, torch.Tensor):
        Y = Y.numpy()
    if isinstance(X, torch.Tensor):
        X = X.numpy()
    num_kp = int(np.round(len(sel_idx)*keep_ratio, 0))
    
    bcors = []
    for fn_ix in sel_idx:
        cur_x = X[:, fn_ix]
        bcors.append(get_ball_cor(cur_x, Y))
    bcors = np.array(bcors);
    keep_idxs = np.sort(np.argsort(-bcors)[:num_kp])
    return sel_idx[keep_idxs], bcors


# some utils 
def array2d2Robj(mat):
    """
    Converts a 2D numpy array to an R matrix object.

    Args:
        mat (numpy.ndarray): A 2D numpy array.

    Returns:
        r.matrix: An R matrix object.
    """
    mat_vec = mat.reshape(-1)
    mat_vecR = robj.FloatVector(mat_vec)
    matR = robj.r.matrix(mat_vecR, nrow=mat.shape[0], ncol=mat.shape[1], byrow=True)
    return matR
def get_ball_cor(x, y):
    """
    Calculates the ball correlation coefficient between two arrays x and y.

    Parameters:
    x (numpy.ndarray): A 1-D or 2-D array.
    y (numpy.ndarray): A 1-D array with the same number of samples as x.

    Returns:
    float: The ball correlation coefficient between x and y.
    """
    r = robj.r
    r["library"]("Ball");
    assert y.shape[0] == x.shape[0], "There should be the same number of samples"
    assert x.ndim <= 2, "x is at most 2-d array"
    assert y.ndim == 1, "y is 1-d array"
    yR = robj.FloatVector(y);
    if x.ndim == 1:
        xR = robj.FloatVector(x);
    elif x.ndim == 2:
        xR = array2d2Robj(x)
    bcor_v = np.array(r['bcor'](yR, xR))[0]
    return bcor_v