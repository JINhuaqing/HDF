import numpy as np
from easydict import EasyDict as edict
from hdf_utils.fns_sinica import coef_fn, fourier_basis_fn
from copy import deepcopy
from .base_params import get_base_params
from constants import RES_ROOT

base_params = get_base_params("linear") 
base_params.data_gen_params = edict()
base_params.data_gen_params.d = 68 # num of ROIs
base_params.data_gen_params.q = 3 # num of other covariates
base_params.data_gen_params.npts = 100 # num of pts to evaluate X(s)
base_params.data_gen_params.types_ = ["int", "c", 2]
base_params.data_gen_params.gt_alp = np.array([5, -1, 2]) 
base_params.data_gen_params.data_params={"sigma2":1, "srho":0.3, "basis_type":"bsp"}
base_params.data_gen_params.data_type = base_params.model_type
base_params.can_Ns = [4, 6, 8, 10, 12]
base_params.SIS_params = edict({"SIS_pen": 0.02, "SIS_basis_N":8, "SIS_ws":"simpson"})
base_params.opt_params.beta = 10 
base_params.can_lams = [0.5, 0.60,  0.70, 0.80, 0.9, 1, 1.1, 1.2, 1.3, 1.4]

def _get_gt_beta_wrapper(d, fct=2):
    def _get_gt_beta(cs):
        x = np.linspace(0, 1, 100)
        fourier_basis = fourier_basis_fn(x)
        fourier_basis_coefs = ([cs[0]*coef_fn(0.2), cs[1]*coef_fn(0.2), cs[2]*coef_fn(0.2)] + 
                                     [np.zeros(50)] * (d-3-1) +
                                     [coef_fn(0.2)]
                                     )
        fourier_basis_coefs = np.array(fourier_basis_coefs).T 
        gt_beta = fourier_basis @ fourier_basis_coefs * fct
        return gt_beta
    return _get_gt_beta



#---settings------------------------------
#========================================================================================================
settingn0s1 = edict(deepcopy(base_params))
add_params = edict({})
add_params.data_gen_params = edict(deepcopy(base_params.data_gen_params))
add_params.data_gen_params.n = 200 # num of data obs to be genareted
add_params.data_gen_params.cs_fn = lambda c: [c, 0, 0]
add_params.data_gen_params.npts = 100 # num of pts to evaluate X(s)
add_params.data_gen_params.data_params["err_dist"] = "normal"
add_params.data_gen_params.beta_fn = _get_gt_beta_wrapper(add_params.data_gen_params.d, fct=2)

add_params.setting = "n0s1"
add_params.sel_idx =  np.arange(1, add_params.data_gen_params.d)
add_params.SIS_ratio = 0.2
settingn0s1.update(add_params)


## -------

settingn0s1b = edict(deepcopy(base_params))
add_params = edict({})
add_params.data_gen_params = edict(deepcopy(base_params.data_gen_params))
add_params.data_gen_params.n = 200 # num of data obs to be genareted
add_params.data_gen_params.cs_fn = lambda c: [c, 0, 0]
add_params.data_gen_params.data_params["err_dist"] = "t"
add_params.data_gen_params.beta_fn =  _get_gt_beta_wrapper(add_params.data_gen_params.d, fct=2)

add_params.setting = "n0s1b"
add_params.sel_idx =  np.arange(1, add_params.data_gen_params.d)
add_params.SIS_ratio = 0.2
settingn0s1b.update(add_params)


#========================================================================================================

settingn0s2 = edict(deepcopy(base_params))
add_params = edict({})
add_params.data_gen_params = edict(deepcopy(base_params.data_gen_params))
add_params.data_gen_params.n = 200 # num of data obs to be genareted
add_params.data_gen_params.cs_fn = lambda c: [c+0.5, 0.5, 0]
add_params.data_gen_params.data_params["err_dist"] = "normal"
add_params.data_gen_params.beta_fn = lambda cs: _get_gt_beta_wrapper(cs, 
                                                             add_params.data_gen_params.d, 
                                                             add_params.data_gen_params.npts, 
                                                             fct=2)

add_params.setting = "n0s2"
add_params.sel_idx =  np.arange(2, add_params.data_gen_params.d)
add_params.SIS_ratio = 0.2
settingn0s2.update(add_params)


## -------

settingn0s2b = edict(deepcopy(base_params))
add_params = edict({})
add_params.data_gen_params = edict(deepcopy(base_params.data_gen_params))
add_params.data_gen_params.n = 200 # num of data obs to be genareted
add_params.data_gen_params.cs_fn = lambda c: [c+0.5, 0.5, 0]
add_params.data_gen_params.data_params["err_dist"] = "t"
add_params.data_gen_params.beta_fn = lambda cs: _get_gt_beta_wrapper(cs, 
                                                             add_params.data_gen_params.d, 
                                                             add_params.data_gen_params.npts, 
                                                             fct=2)

add_params.setting = "n0s2b"
add_params.sel_idx =  np.arange(2, add_params.data_gen_params.d)
add_params.SIS_ratio = 0.2
settingn0s2b.update(add_params)

#========================================================================================================


#========================================================================================================

#### n = 500
settingn0s1a = edict(deepcopy(settingn0s1))
settingn0s1a.setting = "n0s1a"
settingn0s1a.data_gen_params.n = 500
settingn0s1e = edict(deepcopy(settingn0s1b))
settingn0s1e.setting = "n0s1e"
settingn0s1e.data_gen_params.n = 500

settingn0s2a = edict(deepcopy(settingn0s2))
settingn0s2a.setting = "n0s2a"
settingn0s2a.data_gen_params.n = 500
settingn0s2e = edict(deepcopy(settingn0s2b))
settingn0s2e.setting = "n0s2e"
settingn0s2e.data_gen_params.n = 500



#========================================================================================================
settings = edict()
settings.n0s1 = settingn0s1
settings.n0s1a = settingn0s1a
settings.n0s1b = settingn0s1b
settings.n0s1e = settingn0s1e

settings.n0s2 = settingn0s2
settings.n0s2a = settingn0s2a
settings.n0s2b = settingn0s2b
settings.n0s2e = settingn0s2e
