import torch
import numpy as np
from .base_model import BaseModel

class LogisticModel(BaseModel):
    """Logistic model
    """
    def log_lik(self, alp, Gam):
        Os = self._obt_lin_tm(alp, Gam)
        tm1 = self.Y * Os
        tm2 = -torch.log(1+torch.exp(Os))
        rev = torch.mean(tm1+tm2)
        return rev
    
    def log_lik_der1(self, alp, Gam):
        """First dervative of log_likelihood w.r.t theta = [alp^T, N^{-1/2}*col_vec(Gam)^T]^T
        """
        Os = self._obt_lin_tm(alp, Gam) # linear term
        
        if self.lin_tm_der is None:
            self._linear_term_der()
        tm2 = self.lin_tm_der #M x (q+dxN)
        #M x (q+dxN)
        
        tm1 = self.Y - torch.exp(Os)/(1+torch.exp(Os)) # M
        
        log_lik_der1_vs = tm1.unsqueeze(-1) * tm2 #M x (q+dxN)
        log_lik_der1_v = log_lik_der1_vs.mean(axis=0) # (q+dxN)
        return log_lik_der1_v
    
    def log_lik_der2(self, alp, Gam):
        """Second dervative of log_likelihood w.r.t theta = [alp^T, N^{-1/2}*col_vec(Gam)^T]^T
        """
        Os = self._obt_lin_tm(alp, Gam) # linear term
        
        tm1 = - torch.exp(Os)/((1+torch.exp(Os))**2) #M

        if self.lin_tm_der is None:
            self._linear_term_der()
        tm2 = self.lin_tm_der #M x (q+dxN)
        
        log_lik_der2_vs = tm1.unsqueeze(-1).unsqueeze(-1) * tm2.unsqueeze(1) * tm2.unsqueeze(2)
        log_lik_der2_v = log_lik_der2_vs.mean(axis=0) # (q+dxN) x (q+dxN)
        return log_lik_der2_v