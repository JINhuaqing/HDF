import numpy as np
from easydict import EasyDict as edict
from hdf_utils.fns_sinica import coef_fn, fourier_basis_fn
from .base_params import get_base_params
from copy import deepcopy
from constants import RES_ROOT

base_params = get_base_params("logi") 
base_params.data_gen_params = edict()
base_params.data_gen_params.d = 68 # num of ROIs
base_params.data_gen_params.q = 3 # num of other covariates
base_params.data_gen_params.npts = 100 # num of pts to evaluate X(s)
base_params.data_gen_params.freqs = np.linspace(2, 45, base_params.data_gen_params.npts) # freqs
base_params.data_gen_params.types_ = ["int", "c", 2]
base_params.data_gen_params.is_std = False
base_params.data_gen_params.gt_alp0 = np.array([-1, 2]) # we will determine intercept later
base_params.data_gen_params.intercept_cans = np.linspace(-30, 1, 20)
base_params.data_gen_params.data_params={"psd_noise_sd":10}
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
add_params.data_gen_params.beta_fn = _get_gt_beta
add_params.data_gen_params.cs_fn = lambda c: [c, 0, 0]

add_params.setting = "n1"
add_params.sel_idx =  np.arange(1, add_params.data_gen_params.d)
add_params.can_lams = [0.001, 0.05, 0.1, 0.13, 0.15, 0.18, 0.2, 0.25, 0.3, 0.35, 2.0, 4.0, 16.0] 
add_params.SIS_ratio = 0.2
settingn1.update(add_params)

#----------------------------------

settingn1a = edict(deepcopy(base_params))
add_params = edict({})
add_params.data_gen_params = edict(base_params.data_gen_params.copy())
add_params.data_gen_params.n = 500 # num of data obs to be genareted
add_params.data_gen_params.beta_fn = _get_gt_beta
add_params.data_gen_params.cs_fn = lambda c: [c, 0, 0]

add_params.setting = "n1a"
add_params.sel_idx =  np.arange(1, add_params.data_gen_params.d)
add_params.can_lams = [0.001, 0.05, 0.1, 0.13, 0.15, 0.18, 0.2, 0.25, 0.3, 0.35, 2.0, 4.0, 16.0] 
add_params.SIS_ratio = 0.2
settingn1a.update(add_params)

#----------------------------------

settingn2 = edict(deepcopy(base_params))
add_params = edict({})
add_params.data_gen_params = edict(deepcopy(base_params.data_gen_params))
add_params.data_gen_params.n = 200 # num of data obs to be genareted
add_params.data_gen_params.cs_fn = lambda c: [c+0.5, 0.5, 0]
add_params.data_gen_params.beta_fn = _get_gt_beta

add_params.setting = "n2"
add_params.sel_idx =  np.arange(2, add_params.data_gen_params.d)
add_params.can_lams = [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0, 2.0, 4.0, 16.0]
add_params.SIS_ratio = 0.2
settingn2.update(add_params)

#----------------------------------

settingn2a = edict(deepcopy(base_params))
add_params = edict({})
add_params.data_gen_params = edict(deepcopy(base_params.data_gen_params))
add_params.data_gen_params.n = 500 # num of data obs to be genareted
add_params.data_gen_params.cs_fn = lambda c: [c+0.5, 0.5, 0]
add_params.data_gen_params.beta_fn = _get_gt_beta

add_params.setting = "n2a"
add_params.sel_idx =  np.arange(2, add_params.data_gen_params.d)
add_params.can_lams = [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0, 2.0, 4.0, 16.0]
add_params.SIS_ratio = 0.2
settingn2a.update(add_params)

#----------------------------------

settingn3 = edict(deepcopy(base_params))
add_params = edict({})
add_params.data_gen_params = edict(deepcopy(base_params.data_gen_params))
add_params.data_gen_params.n = 200 # num of data obs to be genareted
add_params.data_gen_params.cs_fn = lambda c: [c, c, c]
add_params.data_gen_params.beta_fn = _get_gt_beta

add_params.setting = "n3"
add_params.sel_idx =  np.arange(3, add_params.data_gen_params.d)
add_params.can_lams = [0.001, 0.1, 0.3, 0.4, 0.45, 0.5, 0.6,  0.7, 0.8, 1.0, 2.0, 4.0, 16.0]
add_params.SIS_ratio = 0.2
settingn3.update(add_params)


#----------------------------------

settingn3a = edict(deepcopy(base_params))
add_params = edict({})
add_params.data_gen_params = edict(deepcopy(base_params.data_gen_params))
add_params.data_gen_params.n = 500 # num of data obs to be genareted
add_params.data_gen_params.cs_fn = lambda c: [c, c, c]
add_params.data_gen_params.beta_fn = _get_gt_beta

add_params.setting = "n3a"
add_params.sel_idx =  np.arange(3, add_params.data_gen_params.d)
add_params.can_lams = [0.001, 0.1, 0.3, 0.4, 0.45, 0.5, 0.6,  0.7, 0.8, 1.0, 2.0, 4.0, 16.0]
add_params.SIS_ratio = 0.2
settingn3a.update(add_params)



#-------

settings = edict()
settings.n1 = settingn1
settings.n2 = settingn2
settings.n3 = settingn3
settings.n1a = settingn1a
settings.n2a = settingn2a
settings.n3a = settingn3a