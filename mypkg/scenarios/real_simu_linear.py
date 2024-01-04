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
base_params.data_gen_params.freqs = np.linspace(2, 45, base_params.data_gen_params.npts) # freqs
base_params.data_gen_params.types_ = ["int", "c", 2]
base_params.data_gen_params.is_std = False
base_params.data_gen_params.gt_alp = np.array([5, -1, 2]) # we will determine intercept later
base_params.data_gen_params.data_params={"psd_noise_sd":10, "sigma2":1}
base_params.data_gen_params.data_type = base_params.model_type
base_params.can_Ns = [4, 6, 8, 10, 12]
def _get_gt_beta(cs, d, npts, fct=2):
    x = np.linspace(0, 1, npts)
    fourier_basis = fourier_basis_fn(x)
    fourier_basis_coefs = ([cs[0]*coef_fn(0.2), cs[1]*coef_fn(0.2), cs[2]*coef_fn(0.2)] + 
                                 [np.zeros(50)] * (d-3-1) +
                                 [coef_fn(0.2)]
                                 )
    fourier_basis_coefs = np.array(fourier_basis_coefs).T 
    gt_beta = fourier_basis @ fourier_basis_coefs * fct
    return gt_beta



##---settings------------------------------
settingn1 = edict(deepcopy(base_params))
add_params = edict({})
add_params.data_gen_params = edict(deepcopy(base_params.data_gen_params))
add_params.data_gen_params.n = 200 # num of data obs to be genareted
add_params.data_gen_params.cs_fn = lambda c: [c, 0, 0]
add_params.data_gen_params.data_params["err_dist"] = "normal"
add_params.data_gen_params.beta_fn = lambda cs: _get_gt_beta(cs, 
                                                             add_params.data_gen_params.d, 
                                                             add_params.data_gen_params.npts, 
                                                             fct=2)

add_params.setting = "n1"
add_params.sel_idx =  np.arange(1, add_params.data_gen_params.d)
add_params.can_lams = [0.001, 0.2, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1, 2, 8]
add_params.can_Ns = [4, 6, 8, 10, 12, 14]
add_params.SIS_ratio = 0.2
settingn1.update(add_params)

## -------

settingn1b = edict(deepcopy(base_params))
add_params = edict({})
add_params.data_gen_params = edict(deepcopy(base_params.data_gen_params))
add_params.data_gen_params.n = 200 # num of data obs to be genareted
add_params.data_gen_params.cs_fn = lambda c: [c, 0, 0]
add_params.data_gen_params.data_params = edict(base_params.data_gen_params.data_params.copy())
add_params.data_gen_params.data_params["err_dist"] = "t1"
add_params.data_gen_params.beta_fn = lambda cs: _get_gt_beta(cs, 
                                                             add_params.data_gen_params.d, 
                                                             add_params.data_gen_params.npts, 
                                                             fct=2)

add_params.setting = "n1b"
add_params.sel_idx =  np.arange(1, add_params.data_gen_params.d)
add_params.can_lams = [0.001, 0.1, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.1, 1.2, 2, 8]
add_params.can_Ns = [4, 6, 8, 10, 12, 14]
add_params.SIS_ratio = 0.2
settingn1b.update(add_params)


#-------

settings = edict()
settings.n1 = settingn1
settings.n1b = settingn1b
