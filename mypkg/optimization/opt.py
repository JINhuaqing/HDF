# this file contains function for optimization
import numpy as np
from tqdm import trange
from easydict import EasyDict as edict
import torch

from .one_step_opt import OneStepOpt
from models.linear_model import LinearModel
from models.logistic_model import LogisticModel
from utils.misc import save_pkl
from penalties.scad_pen import SCAD
from splines import obt_bsp_obasis_Rfn, obt_bsp_basis_Rfn_wrapper
from hdf_utils.likelihood import obt_lin_tm
from hdf_utils.SIS import SIS_GLIM
from utils.functions import logit_fn
from utils.matrix import  col_vec2mat_fn, col_vec_fn
from utils.misc import  _set_verbose_level, _update_params
from scipy.stats import chi2
from hdf_utils.hypo_test import  MS2idxs
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    ch = logging.StreamHandler() # for console. 
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch) 
    

    
class HDFOpt():
    """
    Perform HDF optimization.
    """
    def __init__(self, lam, 
                 sel_idx, 
                 model_type="linear",
                 verbose=2, 
                 SIS_ratio=0.2, 
                 N=8,
                 is_std_data=True, 
                 cov_types=None, 
                 inits=None,
                 model_params = {}, 
                 SIS_params = {}, 
                 opt_params = {},
                 bsp_params = {}, 
                 pen_params = {}
               ):
        """
        Args:
            lam (float): The penalty term for SCAD. Scale > 0.
            sel_idx (list): The set of betas to be regularized, an array of beta indices (starting from zero).
            N (int, optional): The number of basis functions. Defaults to 8.
            verbose (int, optional): Verbosity level. Defaults to 2.
            model_type (str, optional): The regression type. Currently supports "linear" or "logistic". Defaults to "linear".
            is_std_data (bool, optional): Whether to standardize Z, center Y, and X across subjects. Defaults to True.
            cov_types (list, optional): The covariate types. A list of length num_covs. 
                                        If None, infer from data. Only used when is_std_data=True. Defaults to None.
            SIS_ratio (float, optional): The ratio for the SIS step. A number between (0, 1]. 
                                         If 1, no SIS is performed. Defaults to 0.2.
            inits (list, optional): The initial values for optimization. 
                                    A list [Gam_init, theta_init, rhok_init] or None. 
                                    If None, use zero initial values. Defaults to None.
            model_params (dict, optional): Other parameters for the model. 
            SIS_params (dict, optional): Other parameters for SIS.
            opt_params (dict, optional): Other parameters for optimization. Refer to the optimization function. 
            bsp_params (dict, optional): Other parameters for bspline
                - basis_ord (int, optional): The B-spline basis order. Defaults to 4.
                - is_orth_basis (bool, optional): Whehter use orthogonal basis or not. Default to True
            pen_params (dict, optional): Other parameters for penalty
                - a (float, optional): The parameter for SCAD. A positive number. Defaults to 3.7.
        """
        _set_verbose_level(verbose, logger)
        
        
        bsp_params_def = {
            "basis_ord": 4, 
            "is_orth_basis": True, 
            "N": N
        }
        
        if model_type.startswith("linear"):
            model_params_def = {
                "norminal_sigma2": 1
                    }
            opt_params_def = {
               'stop_cv': 0.0005,
               'max_iter': 2000, 
               "one_step_verbose":0,
               'alpha': 0.9,
               'beta': 1,
               'R': 2e5,
               "linear_theta_update": "cholesky_inv",
             }
        elif model_type.startswith("logi"):
            model_params_def = {}
            opt_params_def = {
               'stop_cv': 0.0005,
               'max_iter': 2000, 
               "one_step_verbose":0,
               'alpha': 0.9,
               'beta': 1,
               'R': 2e5,
               'N_eps': 1e-4,
               'N_maxit': 100,
               "is_BFGS": "adaptive", 
             }
            
        pen_params_def = {
            "a": 3.7, 
            "lam": lam
                }
            
        opt_params = _update_params(opt_params, opt_params_def, logger)
        bsp_params = _update_params(bsp_params, bsp_params_def, logger)
        model_params = _update_params(model_params, model_params_def, logger)
        pen_params = _update_params(pen_params, pen_params_def, logger)
        SIS_params_def = {
             "SIS_pen": 0.02,  
             "SIS_basis_N": 8,  
             "SIS_basis_ord": bsp_params.basis_ord, 
             "SIS_ratio": SIS_ratio,
         }
            
        SIS_params = _update_params(SIS_params, SIS_params_def, logger)
        
        # In case, these params are provided in {} which override main inputs
        bsp_params["N"] = N
        SIS_params["SIS_ratio"] = SIS_ratio
        pen_params["lam"] = lam
        
        logger.info(f"opt parmas is {opt_params}.")
        logger.info(f"SIS parmas is {SIS_params}.")
        logger.info(f"model parmas is {model_params}.")
        logger.info(f"penalty parmas is {pen_params}.")
        logger.info(f"bspline parmas is {bsp_params}.")
        
        if bsp_params.is_orth_basis:
            self.obt_bsp = obt_bsp_obasis_Rfn
        else:
            self.obt_bsp = obt_bsp_basis_Rfn_wrapper
            
        self.model_type = model_type.lower()
        self.opt_params = opt_params
        self.model_params = model_params
        self.bsp_params = bsp_params 
        self.pen_params = pen_params 
        self.SIS_params = SIS_params
        
        
        self.N = N
        self.is_std_data = is_std_data
        self.cov_types = cov_types
        self.inits = inits
        self.verbose = verbose
        self.sel_idx = sel_idx
        
        self.X, self.Y, self.Z = None, None, None
        self.keep_idxs = None
        self.opt_res = None
        self.est_alp, self.est_Gam = None, None
        self.hypo_utils = None
        self.data_params = edict({})
    
    def add_data(self, X, Y, Z):
        """
        Args:
            X (torch.Tensor): The functional part, a tensor with shape (num_sub, num_roi, num_pts).
            Y (torch.Tensor): The response, a tensor of shape (num_sub,).
            Z (torch.Tensor): The covariates, a tensor of shape (num_sub, num_covs).
        """
        n, q = Z.shape
        _, _, npts = X.shape
        # Standardize Z and center Y and X across subjects
        if self.is_std_data:
            if self.cov_types is None:
                logger.info(f"As cov_types is not provided, inferring the continuous covariates.")
                con_idxs = [len(torch.unique(Z[:, cov_ix])) >= (n*0.9) for cov_ix in range(q)]
            else:
                con_idxs = [typ =="c" for typ in cov_types]
                
            if self.model_type.startswith("linear"):
                Y = Y - Y.mean(axis=0, keepdims=True)
            X = X - X.mean(axis=0, keepdims=True)
            Z[:, con_idxs] = ((Z[:, con_idxs] - Z[:, con_idxs].mean(axis=0, keepdims=True))
                                  /Z[:, con_idxs].std(axis=0, keepdims=True))
        self.X, self.Y, self.Z = X, Y, Z
        self.data_params.n = X.shape[0]
        self.data_params.d = X.shape[1]
        self.data_params.q = Z.shape[1]
        self.con_idxs = con_idxs
        
        xs = np.linspace(0, 1, npts)
        if self.SIS_params.SIS_ratio < 1:
            SIS_basis_mat = torch.tensor(self.obt_bsp(xs, self.SIS_params.SIS_basis_N, self.SIS_params.SIS_basis_ord))
            self.SIS_basis_mat = SIS_basis_mat.to(torch.get_default_dtype())
        self.basis_mat = torch.tensor(self.obt_bsp(xs, self.bsp_params.N, self.bsp_params.basis_ord)).to(torch.get_default_dtype())
        
    def _fit(self, lam, N, X, Y, Z, is_cv=False, is_pbar=True):
        """
        do the opt
        """
        n, q = Z.shape
        _, d, _= X.shape
        
        if self.SIS_params.SIS_ratio < 1:
            keep_idxs, _ = SIS_GLIM(Y=Y, X=X, Z=Z, 
                                    basis_mat=self.SIS_basis_mat,
                                    sel_idx=self.sel_idx,
                                    keep_ratio=self.SIS_params.SIS_ratio, 
                                    model_type=self.model_type, 
                                    SIS_pen=self.SIS_params.SIS_pen, 
                                    )
        else:
            keep_idxs = self.sel_idx
            
        # Get data after SIS
        M_idxs = np.delete(np.arange(d), self.sel_idx) # The set of indices of betas not penalized
        keep_idxs = np.sort(np.concatenate([M_idxs, keep_idxs]))
        sel_idx_SIS = np.where(np.array([keep_idx in self.sel_idx for keep_idx in keep_idxs]))[0]
        d_SIS = len(keep_idxs)
        X_SIS = X[:, keep_idxs, :]
        
        
        # Penalty
        pen = SCAD(lams=lam, a=self.pen_params.a, sel_idx=sel_idx_SIS);
        
        # Get model
        if self.model_type.startswith("linear"):
            model = LinearModel(Y=Y, 
                                X=X_SIS, 
                                Z=Z, 
                                basis_mat=self.basis_mat, 
                                sigma2=self.model_params["norminal_sigma2"])
        elif self.model_type.startswith("logi"):
            model = LogisticModel(Y=Y, 
                                  X=X_SIS, 
                                  Z=Z, 
                                  basis_mat=self.basis_mat)
        else:
            raise ValueError(f"{self.model_type} is not supported now.")
            
        if self.inits is None:
            Gam_init = torch.zeros(N, d_SIS)
            theta_init = torch.cat([torch.zeros(q), col_vec_fn(Gam_init)/np.sqrt(N)])
            rhok_init = torch.zeros(d_SIS*N)
        else:
            Gam_init, theta_init, rhok_init = self.inits
        opt_res = self._optimization(model=model,  penalty=pen,  inits=[Gam_init, theta_init, rhok_init], is_pbar=is_pbar)
        
        if not is_cv:
            self.keep_idxs = keep_idxs
            return opt_res
        else:
            return opt_res, keep_idxs
        
    def _optimization(self, model, penalty, inits, is_pbar=True):
        is_pbar = (self.verbose >= 2) and is_pbar
        eps = 1e-10
        q = model.Z.shape[-1]
        N = model.basis_mat.shape[-1]
        opt_params = self.opt_params
        
        last_Gamk = 0
        last_rhok = 0
        last_thetak = 0
        last_alpk = 0
        Gam_init, theta_init, rhok_init = inits
        if is_pbar:
            prg_bar = trange(opt_params.max_iter, desc="Main Loop")
        else:
            prg_bar = range(opt_params.max_iter)
        one_step_params = edict(opt_params.copy())
        one_step_params.pop("stop_cv")
        one_step_params.pop("max_iter")
        one_step_params.pop("one_step_verbose")
        one_step_params["verbose"] = opt_params.one_step_verbose
        one_step_params["linear_mat"] = None
        for ix in prg_bar:
            opt = OneStepOpt(Gamk=Gam_init, 
                             rhok=rhok_init, 
                             model=model, 
                             penalty=penalty, 
                             theta_init=theta_init, 
                             **one_step_params,
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
            
            Gam_theta_diff = opt.Gamk - col_vec2mat_fn(opt.thetak[q:], nrow=N)*np.sqrt(N)
            Gam_theta_diff_norm = torch.norm(Gam_theta_diff)/(torch.norm(opt.Gamk)+eps)
            
            stop_v = np.max([alp_diff_norm.item(),
                             Gam_diff_norm.item(), 
                             theta_diff_norm.item(), 
                             Gam_theta_diff_norm.item()])
            if stop_v < opt_params.stop_cv:
                break
            if np.isnan(stop_v):
                logger.error(f"The optimization encounters nan")
                break
                
            if is_pbar:
                if ix % 10 == 0:
                    prg_bar.set_postfix({'error': stop_v, 
                                         'GamL0': torch.sum(torch.norm(opt.Gamk, dim=0)!=0).item(),
                                         "CV":opt_params.stop_cv}, 
                                        refresh=True)
                
            last_alpk = opt.alpk
            last_Gamk = opt.Gamk
            last_rhok = opt.rhok
            last_thetak = opt.thetak
            if isinstance(model, LinearModel):
                one_step_params.linear_mat = opt.linear_mat
            
        if ix == (opt_params.max_iter-1):
            logger.warning(f"The optimization may not converge with stop value {stop_v:.3E}")
            opt.conv_iter = -1
        else:
            opt.conv_iter = ix + 1
        logger.debug(f"The optimization takes {ix+1} iters to converge with stop value {stop_v:.3E}.")
        return opt
    
    def fit(self):
        self.opt_res = self._fit(self.pen_params.lam, self.bsp_params.N, self.X, self.Y, self.Z)
        self.est_alp, self.est_Gam = self.opt_res.alpk, self.opt_res.Gamk
        self.conv_iter = self.opt_res.conv_iter
        return self.opt_res
    
    def get_cv_est(self, num_cv_fold=5):
        """
        Get CV estimate for each Y
        """
        assert self.X is not None, f"Plz add data by .add_data() first."
        n = self.data_params.n
        
        num_test = int(n/num_cv_fold)
        full_idx = np.arange(n)
        test_Y_est_all = []
        if self.verbose >= 2:
            prg_bar = trange(num_cv_fold, desc="Cross Validation")
        else:
            prg_bar = range(num_cv_fold)
            
        for ix in prg_bar:
            test_idx = full_idx[(ix*num_test):(ix*num_test+num_test)]
            if ix == num_cv_fold-1:
                test_idx = full_idx[(ix*num_test):] # including all remaining data
            train_idx = np.delete(full_idx, test_idx)
            
            test_set_X = self.X[test_idx]
            test_set_Y = self.Y[test_idx]
            test_set_Z = self.Z[test_idx]
            
            train_set_X = self.X[train_idx]
            train_set_Y = self.Y[train_idx]
            train_set_Z = self.Z[train_idx]
            
            if self.is_std_data:
                con_idxs = self.con_idxs
                test_set_X = test_set_X - train_set_X.mean(axis=0, keepdims=True)
                test_set_Z[:, con_idxs] = (test_set_Z[:, con_idxs] - train_set_Z[:, con_idxs].mean(axis=0, keepdims=True))/train_set_Z[:, con_idxs].std(axis=0, keepdims=True)
                
                train_set_X = train_set_X - train_set_X.mean(axis=0, keepdims=True)
                train_set_Z[:, con_idxs] = (train_set_Z[:, con_idxs] - train_set_Z[:, con_idxs].mean(axis=0, keepdims=True))/train_set_Z[:, con_idxs].std(axis=0, keepdims=True)
                
                
                if self.model_type.startswith("linear"):
                    test_set_Y = test_set_Y - train_set_Y.mean(axis=0, keepdims=True)
                    train_set_Y = train_set_Y - train_set_Y.mean(axis=0, keepdims=True)
            cv_res, cur_keep_idxs = self._fit(self.pen_params.lam, self.bsp_params.N, train_set_X, train_set_Y, train_set_Z, 
                                              is_pbar=False, is_cv=True)
            est_alp = cv_res.alpk
            est_Gam = cv_res.Gamk
            test_Y_est = obt_lin_tm(test_set_Z, test_set_X[:, cur_keep_idxs], est_alp, est_Gam, self.basis_mat)
            if self.model_type.startswith("linear"):
                test_Y_est_all.append(test_Y_est.numpy())
            elif self.model_type.startswith("logi"):
                test_Y_est_all.append(logit_fn(test_Y_est.numpy()))
                
        self.cv_Y_est = np.concatenate(test_Y_est_all)
        return self.cv_Y_est
    
    def _prepare_hypotest(self):
        """Calculate necessary quantities for hypo test
        """
        est_alp, est_Gam = self.est_alp, self.est_Gam
        model = self.opt_res.model
        d, q = self.data_params.d, self.data_params.q
        
        # get Q and Sig mat
        Q_mat = -model.log_lik_der2(est_alp, est_Gam)
        model.log_lik_der1(est_alp, est_Gam);
        Sig_mat = (model.log_lik_der1_vs.unsqueeze(-1) * model.log_lik_der1_vs.unsqueeze(1)).mean(axis=0) 
        
        
        # get the non-zeros index
        M_idxs = np.delete(np.arange(d), self.sel_idx) # the M set
        est_theta = torch.cat([est_alp, col_vec_fn(est_Gam)/np.sqrt(self.bsp_params.N)])
        nonzero_idxs = torch.nonzero(torch.norm(est_Gam, dim=0)).reshape(-1).numpy()
        MS_unions = np.sort(np.union1d(M_idxs, nonzero_idxs))
        keep_idxs_test = MS2idxs(q, self.bsp_params.N, MS_unions)
        
        Q_mat_part = Q_mat[keep_idxs_test][:, keep_idxs_test]
        Sig_mat_part = Sig_mat[keep_idxs_test][:, keep_idxs_test]
        
        self.hypo_utils = edict({})
        self.hypo_utils.Q_mat_part = Q_mat_part
        self.hypo_utils.Sig_mat_part = Sig_mat_part
        self.hypo_utils.keep_idxs_test = keep_idxs_test
        self.hypo_utils.k = len(np.setdiff1d(nonzero_idxs, M_idxs))
        if self.model_type.startswith("linear"):
            est_sigma2 = torch.mean((self.opt_res.model.Y - self.opt_res.model._obt_lin_tm(est_alp, est_Gam))**2)
            self.hypo_utils.est_sigma2 = est_sigma2
        
    def _get_Amat(self, k, Cmat):
        """Get A matrix for hypothesis test
            k: Num of elements in S
            Cmat: Hypothesis matrix
            other parameters
                    required: N, m, q
        """
        m = Cmat.shape[-1]
        N = self.bsp_params.N
        q = self.data_params.q
        
        part1 = np.kron(Cmat, np.eye(N))
        part2 = np.zeros((m*N, q+(m+k)*N))
        part2[:, q:(q+m*N)] = np.eye(m*N)
        A = part1 @ part2
        return A
        
    def hypo_test(self, Cmat, is_simpler=False, hypo_params={}):
        """
        Conduct hypothesis test based on the fitting resutls
        args:
            Cmat(np.ndarray): r x m matrix for the hypotest problem
            is_simpler(bool): only used for linear model, calculate the test stat in a simpler way to 
                              avoid too many matrix inverses.
            hypo_params(dict): other hypo parameters
                - svdinv_eps_Q: default to 1e-7, rtol for Q
                - svdinv_eps_Psi: default to 1e-7, rtol for Psi 
        """
        
        d = self.data_params.d
        n = self.data_params.n
        assert d == (Cmat.shape[1]+self.sel_idx.shape[0]), "Cmat should be compatible with sel_idx." 
        
        hypo_params_def = edict({
            "svdinv_eps_Q": 1e-7,
            "svdinv_eps_Psi": 1e-7,
        })
        hypo_params = _update_params(hypo_params, hypo_params_def, logger)
        hypo_params["Cmat"] = Cmat
        logger.info(f"opt parmas is {hypo_params}.")
        self.hypo_params = hypo_params
        
        
        
        if is_simpler: 
            assert self.model_type.startswith("linear"), "is_simpler only supports linear model"
        
        
        if self.hypo_utils is None:
            self._prepare_hypotest()
        est_alp, est_Gam = self.est_alp, self.est_Gam
        
        est_theta = torch.cat([est_alp, col_vec_fn(est_Gam)/np.sqrt(self.bsp_params.N)])
    
        # A mat
        Amat = torch.Tensor(self._get_Amat(self.hypo_utils.k, hypo_params.Cmat))
        
        # calculate Test stats
        Q_mat_part_inv = torch.linalg.pinv(self.hypo_utils.Q_mat_part, hermitian=True, rtol=hypo_params.svdinv_eps_Q)
        if is_simpler:
            Psi = self.hypo_utils.est_sigma2*Amat @ Q_mat_part_inv @ Amat.T/self.model_params.norminal_sigma2
        else:
            Psi = Amat @ Q_mat_part_inv @ self.hypo_utils.Sig_mat_part @ Q_mat_part_inv @ Amat.T
        Psi_inv = torch.linalg.pinv(Psi, hermitian=True, rtol=hypo_params.svdinv_eps_Psi)
        
        T_p1 = Amat @ est_theta[self.hypo_utils.keep_idxs_test]
        T_v = T_p1 @ Psi_inv @ T_p1 * n 
        
        
        pval = chi2.sf(T_v, Cmat.shape[0]*self.bsp_params.N)
        
        hypo_test_res = edict()
        hypo_test_res.pval = pval
        hypo_test_res.T_v = T_v
        self.hypo_test_res = hypo_test_res
        return hypo_test_res
    
    def save(self, path, is_compact=True, is_force=True):
        """Save the HDF object
        args:
            is_compact(bool): whether remove all large vars or not to save the storage space.
        """
        if self.hypo_utils is None:
            self._prepare_hypotest()
            
        if is_compact:
            self.X = None
            self.Z = None
            self.opt_res = None
            self.basis_mat = None
            self.SIS_basis_mat = None
        save_pkl(path, self, verbose=self.verbose>=2, is_force=is_force)
    
    def GIC_fn(self, fct="BIC"):
        """
        This function calculates the generalized information criterion (GIC)
        
        Args:
            fct: The function used to adjust the penalty. 
                 If fct=2, GIC is equivalent to AIC.
                 If fct=log(n), GIC is equivalent to BIC.
        
        Returns:
            The calculated GIC value.
        """
        assert self.model_type.startswith("linear"), "this fn is only for linear model"
        if self.hypo_utils is None:
            self._prepare_hypotest()
        if isinstance(fct, str):
            if fct.lower().startswith("bic"):
                fct = np.log(self.data_params.n)
            elif fct.lower().startswith("aic"):
                fct = 2
        # 1 is the variance
        DoF = torch.sum(self.est_Gam.sum(axis=0)!=0) * self.est_Gam.shape[0] + len(self.est_alp) + 1
        return np.log(self.hypo_utils.est_sigma2) + fct*DoF/self.data_params.n
    
    
    def GCV_fn(self):
        """
        This function calculates the generalized crossvalidation (GCV) 
        
        Returns:
            The calculated GCV value.
        """
        assert self.model_type.startswith("linear"), "this fn is only for linear model"
        if self.hypo_utils is None:
            self._prepare_hypotest()
        # 1 is the variance
        DoF = torch.sum(self.est_Gam.sum(axis=0)!=0) * self.est_Gam.shape[0] + len(self.est_alp) + 1
        den = (1-(DoF/self.data_params.n))**2
        return self.hypo_utils.est_sigma2/den

# the function below is not used as I have HDFOpt now
# def HDF_opt(X, Y, Z, lam, sel_idx, N=8, verbose=2, 
#             model_type="linear",
#             is_std=True, 
#             cov_types=None, 
#             SIS_ratio=0.2, 
#             is_orth_basis=True, 
#             inits=None, basis_ord=4, a=3.7,
#             model_params = {}, 
#             SIS_params = {}, 
#             opt_params = {},
#            ):
#     """
#     Perform HDF optimization.
# 
#     Args:
#         X (torch.Tensor): The functional part, a tensor with shape (num_sub, num_roi, num_pts).
#         Y (torch.Tensor): The response, a tensor of shape (num_sub,).
#         Z (torch.Tensor): The covariates, a tensor of shape (num_sub, num_covs).
#         lam (float): The penalty term for SCAD. Scale > 0.
#         sel_idx (list): The set of betas to be regularized, an array of beta indices (starting from zero).
#         N (int, optional): The number of basis functions. Defaults to 8.
#         verbose (int, optional): Verbosity level. Defaults to 2.
#         model_type (str, optional): The regression type. Currently supports "linear" or "logistic". Defaults to "linear".
#         is_std (bool, optional): Whether to standardize Z, center Y, and X across subjects. Defaults to True.
#         cov_types (list, optional): The covariate types. A list of length num_covs. If None, infer from data. Only used when is_std=True. Defaults to None.
#         SIS_ratio (float, optional): The ratio for the SIS step. A number between (0, 1]. If 1, no SIS is performed. Defaults to 0.2.
#         is_orth_basis (bool, optional): Whehter use orthogonal basis or not. Default to True
#         inits (list, optional): The initial values for optimization. A list [Gam_init, theta_init, rhok_init] or None. If None, use zero initial values. Defaults to None.
#         basis_ord (int, optional): The B-spline basis order. Defaults to 4.
#         a (float, optional): The parameter for SCAD. A positive number. Defaults to 3.7.
#         model_params (dict, optional): Other parameters for the model. 
#         SIS_params (dict, optional): Other parameters for SIS.
#         opt_params (dict, optional): Other parameters for optimization. Refer to the optimization function. 
# 
#     Returns:
#         main_res: The result of the main optimization process.
#     """
#     
#     _set_verbose_level(verbose)
#     
#   
#     
#     opt_params_def = {
#        'stop_cv': 0.0005,
#        'max_iter': 2000, 
#        "inner_loop_verbose":0,
#        'alpha': 0.9,
#        'beta': 1,
#        'R': 1e5,
#        'N_eps': 1e-4,
#        'N_maxit': 100,
#        "is_BFGS": "adaptive", 
#        "linear_theta_update": "cholesky_inv",
#        "linear_mat": None, 
#        "is_full": False
#      }
#     
#     SIS_params_def = {
#          "SIS_pen": 0.02,  
#          "SIS_basis_N": 8,  
#          "SIS_basis_ord": basis_ord
#      }
#         
#     model_params_def = {
#         "norminal_sigma2": 1
#             }
#         
#     opt_params = _update_params(opt_params, opt_params_def)
#     SIS_params = _update_params(SIS_params, SIS_params_def)
#     model_params = _update_params(model_params, model_params_def)
#     logger.info(f"opt parmas is {opt_params}.")
#     logger.info(f"SIS parmas is {SIS_params}.")
#     logger.info(f"model parmas is {model_params}.")
#     
#     model_type = model_type.lower()
#     if is_orth_basis:
#         obt_bsp = obt_bsp_obasis_Rfn
#     else:
#         obt_bsp = obt_bsp_basis_Rfn_wrapper
# 
#     n, q = Z.shape
#     _, d, npts = X.shape
#     
#         
#     # Standardize Z and center Y and X across subjects
#     if is_std:
#         if cov_types is None:
#             logger.info(f"As cov_types is not provided, inferring the continuous covariates.")
#             con_idxs = [len(torch.unique(Z[:, cov_ix])) >= (n*0.9) for cov_ix in range(q)]
#         else:
#             con_idxs = [typ =="c" for typ in cov_types]
#             
#         if model_type.startswith("linear"):
#             Y = Y - Y.mean(axis=0, keepdims=True)
#         X = X - X.mean(axis=0, keepdims=True)
#         Z[:, con_idxs] = ((Z[:, con_idxs] - Z[:, con_idxs].mean(axis=0, keepdims=True))
#                               /Z[:, con_idxs].std(axis=0, keepdims=True))
#         
#     
#     # SIS to reduce the number of functionals
#     SIS_params = edict(SIS_params)
#     
#     xs = np.linspace(0, 1, npts)
#     SIS_basis_mat = torch.tensor(obt_bsp(xs, SIS_params.SIS_basis_N, SIS_params.SIS_basis_ord)).to(torch.get_default_dtype())
#     if SIS_ratio < 1:
#         keep_idxs, _ = SIS_GLIM(Y=Y, X=X, Z=Z, 
#                                 basis_mat=SIS_basis_mat,
#                                 sel_idx=sel_idx,
#                                 keep_ratio=SIS_ratio, 
#                                 model_type=model_type, 
#                                 SIS_pen=SIS_params.SIS_pen, 
#                                 )
#     else:
#         keep_idxs = sel_idx
#         
#     # Get data after SIS
#     M_idxs = np.delete(np.arange(d), sel_idx) # The set of indices of betas not penalized
#     keep_idxs = np.sort(np.concatenate([M_idxs, keep_idxs]))
#     sel_idx_SIS = np.where(np.array([keep_idx in sel_idx for keep_idx in keep_idxs]))[0]
#     d_SIS = len(keep_idxs)
#     X_SIS = X[:, keep_idxs, :]
#     
#     
#     # Penalty
#     pen = SCAD(lams=lam, a=a, sel_idx=sel_idx_SIS);
#     
#     # Get model
#     basis_mat = torch.tensor(obt_bsp(xs, N, basis_ord)).to(torch.get_default_dtype())
#     if model_type.startswith("linear"):
#         model = LinearModel(Y=Y, 
#                             X=X_SIS, 
#                             Z=Z, 
#                             basis_mat=basis_mat, 
#                             sigma2=model_params["norminal_sigma2"])
#     elif model_type.startswith("logi"):
#         model = LogisticModel(Y=Y, 
#                               X=X_SIS, 
#                               Z=Z, 
#                               basis_mat=basis_mat)
#     else:
#         raise ValueError(f"{model_type} is not supported now.")
#         
#     if inits is None:
#         Gam_init = torch.zeros(N, d_SIS)
#         theta_init = torch.cat([torch.zeros(q), col_vec_fn(Gam_init)/np.sqrt(N)])
#         rhok_init = torch.zeros(d_SIS*N)
#     else:
#         Gam_init, theta_init, rhok_init = inits
#     main_res = optimization(model=model, 
#                             penalty=pen, 
#                             inits=[Gam_init, theta_init, rhok_init],
#                             verbose=verbose,
#                             **opt_params
#                            )
# 
#    return main_res
# def optimization(model, penalty, inits, is_full=False, verbose=0, **input_params):
#     """The function to do the optimization
#         args:
#             model: likelihood model: LogisticModel or LinearModel
#             penalty: The penalty fn, SCAD or (to be written)
#             inits: Initial values of the parameters, 
#                    inits = [Gam_init, theta_init, rhok_init]
#             verbose: log level, from 0 to 3, increase the output
#             input_params: other parmaters
#                 linear_theta_update: 
#                                 if conjugate, use conj_grad
#                                 if cholesky_solve, use cholesky_solve
#                                 if cholesky_inv, get the cholesky inverse
#                 linear_mat: The matrix for linear_theta_update:
#                                 if conjugate, None
#                                 if cholesky_solve, L where the cholesky decom of left_mat=LL^T under linear, 
#                                 if cholesky_inv, inverse of left_mat
#                 is_BFGS: Whether using BFGS for updating theta or not, defualt is True
#         
#     """
#     _set_verbose_level(verbose)
#         
#     eps = 1e-10 # a small number to avoid divided-by-zero issue
#     # default parameters
#     _paras = edict({
#               'stop_cv': 0.0005,
#               'max_iter': 2000, 
#               "inner_loop_verbose":verbose,
#               'alpha': 0.9,
#               'beta': 1,
#               'R': 1e5,
#               'N_eps': 1e-4,
#               'N_maxit': 100,
#               "is_BFGS": "adaptive", 
#               "linear_theta_update": "cholesky_inv",
#               "linear_mat": None, 
#               })
#     
#     _paras = _update_params(input_params, _paras)
#     
#     _paras.q = model.Z.shape[-1]
#     _paras.N = model.basis_mat.shape[-1]
#     
#     logger.debug(f"The paras is {_paras}.")
#         
#     last_Gamk = 0
#     last_rhok = 0
#     last_thetak = 0
#     last_alpk = 0
#     Gam_init, theta_init, rhok_init = inits
#     if verbose==3:
#         prg_bar = trange(_paras.max_iter)
#     else:
#         prg_bar = range(_paras.max_iter)
#     for ix in prg_bar:
#             opt = OneStepOpt(Gamk=Gam_init, 
#                              rhok=rhok_init, 
#                              alpha=_paras.alpha, 
#                              beta=_paras.beta, 
#                              model=model, 
#                              penalty=penalty, 
#                              theta_init=theta_init, 
#                              verbose=_paras.inner_loop_verbose,
#                              R=_paras.R, 
#                              linear_theta_update=_paras.linear_theta_update,
#                              linear_mat=_paras.linear_mat, 
#                              is_BFGS=_paras.is_BFGS,
#                              N_eps=_paras.N_eps, 
#                              N_maxit=_paras.N_maxit, 
#                              )
#             opt()
#             Gam_init = opt.Gamk
#             rhok_init = opt.rhok
#             theta_init = opt.thetak
#             
#             
#             # converge cv
#             alp_diff = opt.alpk - last_alpk
#             alp_diff_norm = torch.norm(alp_diff)/(torch.norm(opt.alpk)+eps)
#             
#             Gam_diff = opt.Gamk - last_Gamk
#             Gam_diff_norm = torch.norm(Gam_diff)/(torch.norm(opt.Gamk)+eps)
#             
#             theta_diff = opt.thetak - last_thetak
#             theta_diff_norm = torch.norm(theta_diff)/(torch.norm(opt.thetak)+eps)
#             
#             Gam_theta_diff = opt.Gamk - col_vec2mat_fn(opt.thetak[_paras.q:], nrow=_paras.N)*np.sqrt(_paras.N)
#             Gam_theta_diff_norm = torch.norm(Gam_theta_diff)/(torch.norm(opt.Gamk)+eps)
#             
#             stop_v = np.max([alp_diff_norm.item(),
#                              Gam_diff_norm.item(), 
#                              theta_diff_norm.item(), 
#                              Gam_theta_diff_norm.item()])
#             if stop_v < _paras.stop_cv:
#                 break
#             if np.isnan(stop_v):
#                 logger.error(f"The optimization encounters nan")
#                 break
#                 
#             if verbose==3:
#                 if ix % 10 == 0:
#                     prg_bar.set_postfix({'error': stop_v, 
#                                          'GamL0': torch.sum(torch.norm(opt.Gamk, dim=0)!=0).item(),
#                                          "CV":_paras.stop_cv}, 
#                                         refresh=True)
#                 
#             last_alpk = opt.alpk
#             last_Gamk = opt.Gamk
#             last_rhok = opt.rhok
#             last_thetak = opt.thetak
#             if isinstance(model, LinearModel):
#                 _paras.linear_mat = opt.linear_mat
#         
#     if ix == (_paras.max_iter-1):
#         logger.warning(f"The optimization may not converge with stop value {stop_v:.3E}")
#     if is_full:
#         return opt, (ix+1, _paras.max_iter), _paras
#     else:
#         opt.model = None
#         return opt, (ix+1, _paras.max_iter)
