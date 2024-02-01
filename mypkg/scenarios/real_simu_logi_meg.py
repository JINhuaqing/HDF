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
base_params.data_gen_params.types_ = ["int", "c", 2]
base_params.data_gen_params.gt_alp0 = np.array([-1, 2]) # we will determine intercept later
base_params.data_gen_params.intercept_cans = np.linspace(-30, 1, 20)
base_params.data_gen_params.data_params=edict({})
base_params.data_gen_params.data_type = base_params.model_type
base_params.can_Ns = [4, 6, 8, 10, 12]
base_params.can_lams = [0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0, 2, 8]
#base_params.can_lams = [0.01, 0.03, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0, 2, 8]
base_params.SIS_params = edict({"SIS_pen": 5, "SIS_basis_N":10})
base_params.opt_params.beta = 1.0
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



##====================================================================================================
settingnm1 = edict(deepcopy(base_params))
add_params = edict({})
add_params.data_gen_params = edict(deepcopy(base_params.data_gen_params))
add_params.data_gen_params.n = 200 # num of data obs to be genareted
add_params.data_gen_params.cs_fn = lambda c: [c, 0, 0]
add_params.data_gen_params.beta_fn = lambda cs: _get_gt_beta(cs, 
                                                             add_params.data_gen_params.d, 
                                                             add_params.data_gen_params.npts, 
                                                             fct=2)

add_params.setting = "nm1"
add_params.sel_idx =  np.arange(1, add_params.data_gen_params.d)
add_params.SIS_ratio = 0.2
settingnm1.update(add_params)


#----------------------------------------------------------------------------------------------------

settingnm2 = edict(deepcopy(base_params))
add_params = edict({})
add_params.data_gen_params = edict(deepcopy(base_params.data_gen_params))
add_params.data_gen_params.n = 200 # num of data obs to be genareted
add_params.data_gen_params.cs_fn = lambda c: [c+0.5, 0.5, 0]
add_params.data_gen_params.beta_fn = lambda cs: _get_gt_beta(cs, 
                                                             add_params.data_gen_params.d, 
                                                             add_params.data_gen_params.npts, 
                                                             fct=2)

add_params.setting = "nm2"
add_params.sel_idx =  np.arange(2, add_params.data_gen_params.d)
add_params.SIS_ratio = 0.2
settingnm2.update(add_params)


##====================================================================================================
# n = 500

settingnm1a = edict(deepcopy(base_params))
add_params = edict({})
add_params.data_gen_params = edict(base_params.data_gen_params.copy())
add_params.data_gen_params.n = 500 # num of data obs to be genareted
add_params.data_gen_params.cs_fn = lambda c: [c, 0, 0]
add_params.data_gen_params.beta_fn = lambda cs: _get_gt_beta(cs, 
                                                             add_params.data_gen_params.d, 
                                                             add_params.data_gen_params.npts, 
                                                             fct=2)

add_params.setting = "nm1a"
add_params.sel_idx =  np.arange(1, add_params.data_gen_params.d)
add_params.SIS_ratio = 0.2
settingnm1a.update(add_params)

#----------------------------------------------------------------------------------------------------
settingnm2a = edict(deepcopy(base_params))
add_params = edict({})
add_params.data_gen_params = edict(deepcopy(base_params.data_gen_params))
add_params.data_gen_params.n = 500 # num of data obs to be genareted
add_params.data_gen_params.cs_fn = lambda c: [c+0.5, 0.5, 0]
add_params.data_gen_params.beta_fn = lambda cs: _get_gt_beta(cs, 
                                                             add_params.data_gen_params.d, 
                                                             add_params.data_gen_params.npts, 
                                                             fct=2)

add_params.setting = "nm2a"
add_params.sel_idx =  np.arange(2, add_params.data_gen_params.d)
add_params.SIS_ratio = 0.2
settingnm2a.update(add_params)


##====================================================================================================

settings = edict()
settings.nm1 = settingnm1
settings.nm2 = settingnm2
settings.nm1a = settingnm1a
settings.nm2a = settingnm2a
