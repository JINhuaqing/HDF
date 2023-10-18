import torch
import numpy as np
from easydict import EasyDict as edict
from utils.matrix import col_vec_fn, col_vec2mat_fn, gen_Dmat, svd_inverse, conju_grad,  cholesky_inv
from projection import euclidean_proj_l1ball
from models.linear_model import LinearModel
import time
import pdb

def theta_proj(thetal, q, N, R):
    """Proj theta to the space \|theta_1\|_1 + \sum_j \|theta_2j\|_2 < R
       args:
           thetal: the theta to be projected.
           q: Num of covaraites
           N: The bspline space dim
           R: the radius of the space
    """
    alpl = thetal[:q]
    GamlN= col_vec2mat_fn(thetal[q:], nrow=N)
    GamlN_l2norm = torch.norm(GamlN, dim=0)
    cat_vec = torch.cat([alpl, GamlN_l2norm])
    cat_vec_after_proj = torch.tensor(euclidean_proj_l1ball(cat_vec.numpy(), R))
    alpl_after_proj = cat_vec_after_proj[:q]
    GamlN_after_proj = cat_vec_after_proj[q:]*GamlN/GamlN_l2norm
    thetal_after_proj = torch.cat([alpl_after_proj, col_vec_fn(GamlN_after_proj)])
    return thetal_after_proj

class OneStepOpt():
    """One step for the optimization"""
    def __init__(self, Gamk, rhok, alpha, beta, 
                 model, penalty, 
                 theta_init=None, **paras):
        """args:
                Gamk: The Gam from last update, matrix of N x d
                rhok: The vector rho from last update, vector of dN
                alpha : the parameter for CPRSM, scalar, alpha and alp are different, alpha \in (0, 1)
                beta: the parameter for CPRSM, scalar, beta>0
                model: likelihood model: LogisticModel or LinearModel
                penalty: The penalty fn, SCAD or (to be written)
                theta_init: The initial value of theta, vec of q+dN
                            Note theta = [alp, Gam/sqrt(N)]
                paras: other parameters
                    R: The radius of the projection space
                    NR_eps: the stop criteria for Newton-Ralpson method
                    NR_maxit: the max num of iterations for Newton-Ralpson method
                    linear_theta_update: 
                                if conjugate, use conj_grad
                                if cholesky_solve, use cholesky_solve
                                if cholesky_inv, get the cholesky inverse
                    linear_mat: The matrix for linear_theta_update:
                                if conjugate, None
                                if cholesky_solve, L where the cholesky decom of left_mat=LL^T under linear, 
                                if cholesky_inv, inverse of left_mat
        """
        self.paras = edict({
            "linear_theta_update": "cholesky_inv",
            "linear_mat": None,
                           })
        self.paras.update(paras)
        
        self.N, self.d = Gamk.shape
        assert len(rhok) == self.N * self.d
        if theta_init is None:
            self.q = model.Z.shape[-1]
            theta_init = torch.randn(self.q+self.d*self.N)
        else:
            self.q = len(theta_init) - self.N*self.d
    
        self.model = model
        self.penalty = penalty
    
        self.theta_init = theta_init
        self.Gamk = Gamk
        self.rhok = rhok
        self.thetak = None
        self.alpk = None
        
        self.D = gen_Dmat(self.d, self.N, self.q)
        self.beta = beta
        self.alpha = alpha
        
        if isinstance(self.model, LinearModel):
            self.linear_theta_update = self.paras.linear_theta_update.lower()
            assert self.linear_theta_update in ["conjugate", "cholesky_solve", "cholesky_inv"]
            self.linear_mat = self.paras.linear_mat
    
    def _update_theta_linearmodel(self):
        """First step of optimization, update theta under linear model.
            Note that under the linear case, we do not need iterations.
        """
        if self.model.lin_tm_der is None:
            self.model._linear_term_der()
            
        right_vec = (self.D.T@self.rhok +
                     self.beta*self.D.T@col_vec_fn(self.Gamk)/np.sqrt(self.N) + 
                     (self.model.Y.unsqueeze(-1)*self.model.lin_tm_der).mean(dim=0)/self.model.sigma2)
        if self.linear_theta_update.startswith("cholesky_inv"):
            if self.linear_mat is None:
                tmp_mat = self.model.lin_tm_der.T@self.model.lin_tm_der/len(self.model.Y) # (q+dN) x (q+dN)
                left_mat = tmp_mat/self.model.sigma2 + self.beta * self.D.T@self.D
                self.linear_mat = cholesky_inv(left_mat)
            thetak_raw = self.linear_mat @ right_vec;
                
        elif self.linear_theta_update.startswith("conjugate"):
            tmp_mat = self.model.lin_tm_der.T@self.model.lin_tm_der/len(self.model.Y) # (q+dN) x (q+dN)
            left_mat = tmp_mat/self.model.sigma2 + self.beta * self.D.T@self.D
            
            thetak_raw = conju_grad(left_mat, right_vec)
            
        elif self.linear_theta_update.startswith("cholesky_solve"):
            if self.linear_mat is None:
                tmp_mat = self.model.lin_tm_der.T@self.model.lin_tm_der/len(self.model.Y) # (q+dN) x (q+dN)
                left_mat = tmp_mat/self.model.sigma2 + self.beta * self.D.T@self.D
                self.linear_mat = torch.linalg.cholesky(left_mat)
            thetak_raw = torch.cholesky_solve(right_vec.reshape(-1, 1), self.linear_mat).reshape(-1);
        else:
            raise TypeError(f"No such type, {self.linear_theta_update}")
            
        
        self.thetak = theta_proj(thetak_raw, self.q, self.N, self.paras.R) # projection
    
    def _update_theta(self):
        """First step of optimization, update theta 
           This step can be slow
        """
        thetal = self.theta_init
        alpl = self.theta_init[:self.q]
        Gaml = col_vec2mat_fn(self.theta_init[self.q:], nrow=self.N)*np.sqrt(self.N)
        for ix in range(self.paras.NR_maxit):
            der1_p1 = -self.model.log_lik_der1(alpl, Gaml)
            der1_p2 = -self.D.T @ self.rhok
            der1_p3 = self.beta * (self.D.T@self.D@thetal - self.D.T@col_vec_fn(self.Gamk)/np.sqrt(self.N))
            der1 = der1_p1 + der1_p2 + der1_p3
            
            der2_p1 = -self.model.log_lik_der2(alpl, Gaml)
            der2_p2 = self.beta * self.D.T @ self.D 
            der2 = der2_p1 + der2_p2 
            
            theta_last = thetal.clone()
            der2_inv = svd_inverse(der2)
            #der2_inv = torch.inverse(der2) # simply inverse will cause numerical problem
            thetal = theta_last - der2_inv @ der1 # update 
            thetal = theta_proj(thetal, self.q, self.N, self.paras.R) # projection
            alpl = thetal[:self.q]
            Gaml = col_vec2mat_fn(thetal[self.q:], nrow=self.N)*np.sqrt(self.N)
            
            stop_cv = torch.norm(thetal-theta_last)/torch.norm(thetal)
            if stop_cv <= self.paras.NR_eps:
                break
        if ix == (self.paras.NR_maxit-1):
            print("The NR algorithm may not converge")
        self.thetak = thetal
            
    
    def _update_rho(self):
        """Second/Fourth step, update rho to get rho_k+1/2/rho_k"""
        assert self.thetak is not None
        self.rhok = self.rhok - self.alpha*self.beta*(self.D@self.thetak-col_vec_fn(self.Gamk)/np.sqrt(self.N))
        
    
    def _update_Gam(self):
        """Third step, update Gam to get Gam_k+1"""
        GamNk_raw = col_vec2mat_fn(self.D@self.thetak - self.rhok/self.beta, nrow=self.N)
        GamNk = self.penalty(GamNk_raw, C=self.beta)
        self.Gamk = GamNk * np.sqrt(self.N)
    
    def __call__(self):
        t0 = time.time()
        if isinstance(self.model, LinearModel):
            self._update_theta_linearmodel()
        else:
            self._update_theta()
        t1 = time.time()
        self._update_rho()
        t2 = time.time()
        self._update_Gam()
        t3 = time.time()
        self._update_rho()
        t4 = time.time()
        self.alpk = self.thetak[:self.q]
        #print(np.diff([t0, t1, t2, t3, t4]))
