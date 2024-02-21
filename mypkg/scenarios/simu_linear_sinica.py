import numpy as np
from easydict import EasyDict as edict
from hdf_utils.fns_sinica import coef_fn, fourier_basis_fn
from copy import deepcopy
from .base_params import get_base_params
from constants import RES_ROOT

base_params = get_base_params("linear") 
base_params.data_gen_params = edict()
base_params.data_gen_params.d = 200 # num of ROIs
base_params.data_gen_params.q = 1 # num of other covariates
base_params.data_gen_params.npts = 11 # num of pts to evaluate X(s)
base_params.data_gen_params.types_ = ["int"]
base_params.data_gen_params.gt_alp = np.array([0]) # we will determine intercept later
base_params.data_gen_params.data_type = base_params.model_type
base_params.data_gen_params.data_params={"sigma2":1, "srho":0.3}
base_params.SIS_params = edict({"SIS_pen": 100, "SIS_basis_N":4, "SIS_ws":"simpson"})
base_params.opt_params.beta = 1 
base_params.can_Ns = [4, 6, 8, 10, 12, 14]
def _get_gt_beta(cs, d, fct=1):
    x = np.linspace(0, 1, 101)
    fourier_basis = fourier_basis_fn(x)
    fourier_basis_coefs = ([cs[0]*coef_fn(0.2), cs[1]*coef_fn(0.2), cs[2]*coef_fn(0.2)] + 
                                 [np.zeros(50)] * (d-3-1) +
                                 [coef_fn(0.2)]
                                 )
    fourier_basis_coefs = np.array(fourier_basis_coefs).T 
    gt_beta = fourier_basis @ fourier_basis_coefs * fct
    return gt_beta



##---settings------------------------------
#========================================================================================================
settingcmpns1 = edict(deepcopy(base_params))
add_params = edict({})
add_params.data_gen_params = edict(deepcopy(base_params.data_gen_params))
add_params.data_gen_params.n = 100 # num of data obs to be genareted
add_params.data_gen_params.cs_fn = lambda c: [c, 0, 0]
add_params.data_gen_params.data_params["err_dist"] = "normal"
add_params.data_gen_params.beta_fn = lambda cs: _get_gt_beta(cs,
                                                add_params.data_gen_params.d,
                                                fct=1) 
add_params.setting = "cmpns1"
add_params.sel_idx =  np.arange(1, add_params.data_gen_params.d)
add_params.can_lams = [0.001, 0.2, 0.3, 0.4, 0.5, 0.60,  0.70, 0.80, 0.9, 2.0]
add_params.SIS_ratio = 0.2
settingcmpns1.update(add_params)

##---- 
settingcmpns2 = edict(deepcopy(base_params))
add_params = edict({})
add_params.data_gen_params = edict(deepcopy(base_params.data_gen_params))
add_params.data_gen_params.n = 100 # num of data obs to be genareted
add_params.data_gen_params.cs_fn = lambda c: [c, c, 0]
add_params.data_gen_params.data_params["err_dist"] = "normal"
add_params.data_gen_params.beta_fn = lambda cs: _get_gt_beta(cs,
                                                add_params.data_gen_params.d,
                                                fct=1) 
add_params.setting = "cmpns2"
add_params.sel_idx =  np.arange(2, add_params.data_gen_params.d)
add_params.can_lams = [0.001, 0.2, 0.3, 0.4, 0.5, 0.60,  0.70, 0.80, 0.9, 2.0]
add_params.SIS_ratio = 0.2
settingcmpns2.update(add_params)

## -------
settingcmpns3 = edict(deepcopy(base_params))
add_params = edict({})
add_params.data_gen_params = edict(deepcopy(base_params.data_gen_params))
add_params.data_gen_params.n = 100 # num of data obs to be genareted
add_params.data_gen_params.cs_fn = lambda c: [c, c, c]
add_params.data_gen_params.data_params["err_dist"] = "normal"
add_params.data_gen_params.beta_fn = lambda cs: _get_gt_beta(cs,
                                                add_params.data_gen_params.d,
                                                fct=1) 
add_params.setting = "cmpns3"
add_params.sel_idx =  np.arange(3, add_params.data_gen_params.d)
add_params.can_lams = [0.001, 0.2, 0.3, 0.4, 0.5, 0.60,  0.70, 0.80, 0.9, 2.0]
add_params.SIS_ratio = 0.2
settingcmpns3.update(add_params)


#========================================================================================================
### t(3) error
settingcmpns1b = edict(deepcopy(settingcmpns1))
settingcmpns1b.setting = "cmpns1b"
settingcmpns1b.data_gen_params.data_params["err_dist"] = "t"

settingcmpns2b = edict(deepcopy(settingcmpns2))
settingcmpns2b.setting = "cmpns2b"
settingcmpns2b.data_gen_params.data_params["err_dist"] = "t"

settingcmpns3b = edict(deepcopy(settingcmpns3))
settingcmpns3b.setting = "cmpns3b"
settingcmpns3b.data_gen_params.data_params["err_dist"] = "t"


#========================================================================================================

settings = edict()
settings.cmpns1 = settingcmpns1
settings.cmpns2 = settingcmpns2
settings.cmpns3 = settingcmpns3
settings.cmpns1b = settingcmpns1b
settings.cmpns2b = settingcmpns2b
settings.cmpns3b = settingcmpns3b
