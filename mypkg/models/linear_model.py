import torch
import numpy as np
from .base_model import BaseModel


class LinearModel(BaseModel):
    """Linear model
    """
    
    def __init__(self, Y, Z, X, basis_mat, ws=None, sigma2=1):
        """
        args:
               Y: response values: M
               Z: matrix or vector of other covariates, (M) x q
               X: freq of data: (M) x d x npts
               basis_mat: Basis matrix of B-spline evaluated at some pts: npts x N
               ws: the weights used for approximating the integration: npts. 
               sigma2: Variance of the data Y
        """
        super().__init__(Y=Y, Z=Z, X=X, basis_mat=basis_mat, ws=ws)
        self.sigma2 = sigma2
        
    def log_lik(self, alp, Gam):
        """Up to a constant"""
        Os = self._obt_lin_tm(alp, Gam)
        tm1 = -(self.Y - Os)**2
        rev = torch.mean(tm1/2/self.sigma2)
        return rev
    
    def log_lik_der1(self, alp, Gam):
        """First dervative of log_likelihood w.r.t theta = [alp^T, N^{-1/2}*col_vec(Gam)^T]^T
        """
        Os = self._obt_lin_tm(alp, Gam) # linear term
        
        
        if self.lin_tm_der is None:
            self._linear_term_der()
        tm2 = self.lin_tm_der #M x (q+dxN)
        
        tm1 = -(Os-self.Y)/self.sigma2 # M
        
        log_lik_der1_vs = tm1.unsqueeze(-1) * tm2 #M x (q+dxN)
        log_lik_der1_v = log_lik_der1_vs.mean(axis=0) # (q+dxN)
        return log_lik_der1_v
    
    def log_lik_der2(self, alp, Gam):
        """Second dervative of log_likelihood w.r.t theta = [alp^T, N^{-1/2}*col_vec(Gam)^T]^T
        """
        if self.lin_tm_der is None:
            self._linear_term_der()
        tm1 = self.lin_tm_der #M x (q+dxN)
        
        log_lik_der2_vs = - tm1.unsqueeze(1) * tm1.unsqueeze(2)/self.sigma2
        log_lik_der2_v = log_lik_der2_vs.mean(axis=0) # (q+dxN) x (q+dxN)
        return log_lik_der2_v