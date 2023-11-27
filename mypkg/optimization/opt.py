# this file contains function for optimization
import numpy as np
from tqdm import trange
from easydict import EasyDict as edict
import torch
from optimization.one_step_opt import OneStepOpt
from models.linear_model import LinearModel
from utils.matrix import  col_vec2mat_fn
import pdb


def optimization(model, penalty, inits, is_prg=False, save_paras=False, input_paras={}):
    """The function to do the optimization
        args:
            model: likelihood model: LogisticModel or LinearModel
            penalty: The penalty fn, SCAD or (to be written)
            inits: Initial values of the parameters, 
                   inits = [alp_init, Gam_init, theta_init, rhok_init]
            input_paras: other parmaters
                is_small: Whether remove model or not after optmization. If not, the output can be very large
                linear_theta_update: 
                                if conjugate, use conj_grad
                                if cholesky_solve, use cholesky_solve
                                if cholesky_inv, get the cholesky inverse
                linear_mat: The matrix for linear_theta_update:
                                if conjugate, None
                                if cholesky_solve, L where the cholesky decom of left_mat=LL^T under linear, 
                                if cholesky_inv, inverse of left_mat
                is_BFGS: Whether using BFGS for updating theta or not, defualt is True
        
    """
    eps = 1e-10 # a small number to avoid divided-by-zero issue
    # default parameters
    _paras = {
              'is_small': True,
              'alpha': 0.9,
              'beta': 1,
              'N_eps': 1e-05,
              'R_maxit': 100,
              'stop_cv': 0.0005,
              'max_iter': 2000, 
              "linear_theta_update": "cholesky_inv",
              "linear_mat": None, 
              "is_BFGS": True
              }
    _paras = edict(_paras)
    _paras.update(input_paras)
    _paras.q = model.Z.shape[-1]
    _paras.N = model.basis_mat.shape[-1]
        
    last_Gamk = 0
    last_rhok = 0
    last_thetak = 0
    last_alpk = 0
    alp_init, Gam_init, theta_init, rhok_init = inits
    if is_prg:
        prg_bar = trange(_paras.max_iter)
    else:
        prg_bar = range(_paras.max_iter)
    for ix in prg_bar:
            opt = OneStepOpt(Gamk=Gam_init, 
                          rhok=rhok_init, 
                          theta_init=theta_init, 
                          alpha=_paras.alpha, 
                          beta=_paras.beta, 
                          model=model, 
                          penalty=penalty, 
                          N_eps=_paras.N_eps, 
                          N_maxit=_paras.N_maxit, 
                          R=_paras.Rv, 
                          linear_theta_update=_paras.linear_theta_update,
                          linear_mat=_paras.linear_mat, 
                          is_BFGS=_paras.is_BFGS
                          )
            opt()
            Gam_init = opt.Gamk
            rhok_init = opt.rhok
            theta_init = opt.thetak
            
            
            # converge cv
            alp_diff = opt.alpk - last_alpk
            alp_diff_norm = torch.norm(alp_diff)/(torch.norm(opt.alpk)+eps)
            
            Gam_diff = opt.Gamk - last_Gamk
            Gam_diff_norm = torch.norm(Gam_diff)/(torch.norm(opt.Gamk)+eps)
            
            theta_diff = opt.thetak - last_thetak
            theta_diff_norm = torch.norm(theta_diff)/(torch.norm(opt.thetak)+eps)
            
            Gam_theta_diff = opt.Gamk - col_vec2mat_fn(opt.thetak[_paras.q:], nrow=_paras.N)*np.sqrt(_paras.N)
            Gam_theta_diff_norm = torch.norm(Gam_theta_diff)/(torch.norm(opt.Gamk)+eps)
            
            stop_v = np.max([alp_diff_norm.item(),
                             Gam_diff_norm.item(), 
                             theta_diff_norm.item(), 
                             Gam_theta_diff_norm.item()])
            if stop_v < _paras.stop_cv:
                break
            if np.isnan(stop_v):
                print(f"The optimization encounters nan")
                break
                
            if is_prg:
                if ix % 10 == 0:
                    prg_bar.set_postfix({'error': stop_v, 
                                         'GamL0': torch.sum(torch.norm(opt.Gamk, dim=0)!=0).item(),
                                         "CV":_paras.stop_cv}, 
                                        refresh=True)
                
            last_alpk = opt.alpk
            last_Gamk = opt.Gamk
            last_rhok = opt.rhok
            last_thetak = opt.thetak
            if isinstance(model, LinearModel):
                _paras.linear_mat = opt.linear_mat
        
    if ix == (_paras.max_iter-1):
        print(f"The optimization may not converge with stop value {stop_v:.3E}")
    if _paras.is_small:
        opt.model = None
    if save_paras:
        return opt, (ix+1, _paras.max_iter), _paras
    else:
        return opt, (ix+1, _paras.max_iter)