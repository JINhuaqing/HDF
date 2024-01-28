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
base_params.data_gen_params.is_std = False
base_params.data_gen_params.gt_alp = np.array([5, -1, 2]) # we will determine intercept later
base_params.data_gen_params.data_params={"sigma2":1}
base_params.data_gen_params.data_type = base_params.model_type
base_params.can_Ns = [4, 6, 8, 10, 12, 14]
base_params.can_lams = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2, 8]
base_params.SIS_params = edict({"SIS_pen": 5, "SIS_basis_N":10})
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



#---settings------------------------------
#========================================================================================================
settingnm1 = edict(deepcopy(base_params))
add_params = edict({})
add_params.data_gen_params = edict(deepcopy(base_params.data_gen_params))
add_params.data_gen_params.n = 200 # num of data obs to be genareted
add_params.data_gen_params.cs_fn = lambda c: [c, 0, 0]
add_params.data_gen_params.data_params["err_dist"] = "normal"
add_params.data_gen_params.beta_fn = lambda cs: _get_gt_beta(cs, 
                                                             add_params.data_gen_params.d, 
                                                             add_params.data_gen_params.npts, 
                                                             fct=2)

add_params.setting = "nm1"
add_params.sel_idx =  np.arange(1, add_params.data_gen_params.d)
#add_params.can_lams = [0.001, 0.1, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7,  2, 8]
add_params.SIS_ratio = 0.2
settingnm1.update(add_params)

## -------

settingnm1b = edict(deepcopy(base_params))
add_params = edict({})
add_params.data_gen_params = edict(deepcopy(base_params.data_gen_params))
add_params.data_gen_params.n = 200 # num of data obs to be genareted
add_params.data_gen_params.cs_fn = lambda c: [c, 0, 0]
add_params.data_gen_params.data_params["err_dist"] = "t"
add_params.data_gen_params.beta_fn = lambda cs: _get_gt_beta(cs, 
                                                             add_params.data_gen_params.d, 
                                                             add_params.data_gen_params.npts, 
                                                             fct=2)

add_params.setting = "nm1b"
add_params.sel_idx =  np.arange(1, add_params.data_gen_params.d)
#add_params.can_lams = [0.001, 0.1, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7,  2, 8]
add_params.SIS_ratio = 0.2
settingnm1b.update(add_params)


#========================================================================================================

settingnm2 = edict(deepcopy(base_params))
add_params = edict({})
add_params.data_gen_params = edict(deepcopy(base_params.data_gen_params))
add_params.data_gen_params.n = 200 # num of data obs to be genareted
add_params.data_gen_params.cs_fn = lambda c: [c+0.5, 0.5, 0]
add_params.data_gen_params.data_params["err_dist"] = "normal"
add_params.data_gen_params.beta_fn = lambda cs: _get_gt_beta(cs, 
                                                             add_params.data_gen_params.d, 
                                                             add_params.data_gen_params.npts, 
                                                             fct=2)

add_params.setting = "nm2"
add_params.sel_idx =  np.arange(2, add_params.data_gen_params.d)
#add_params.can_lams = [0.001, 0.1, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7,  2, 8]
add_params.SIS_ratio = 0.2
settingnm2.update(add_params)


## -------

settingnm2b = edict(deepcopy(base_params))
add_params = edict({})
add_params.data_gen_params = edict(deepcopy(base_params.data_gen_params))
add_params.data_gen_params.n = 200 # num of data obs to be genareted
add_params.data_gen_params.cs_fn = lambda c: [c+0.5, 0.5, 0]
add_params.data_gen_params.data_params["err_dist"] = "t"
add_params.data_gen_params.beta_fn = lambda cs: _get_gt_beta(cs, 
                                                             add_params.data_gen_params.d, 
                                                             add_params.data_gen_params.npts, 
                                                             fct=2)

add_params.setting = "nm2b"
add_params.sel_idx =  np.arange(2, add_params.data_gen_params.d)
#add_params.can_lams = [0.001, 0.1, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7,  2, 8]
add_params.SIS_ratio = 0.2
settingnm2b.update(add_params)

#========================================================================================================

settingnm3 = edict(deepcopy(base_params))
add_params = edict({})
add_params.data_gen_params = edict(deepcopy(base_params.data_gen_params))
add_params.data_gen_params.n = 200 # num of data obs to be genareted
add_params.data_gen_params.cs_fn = lambda c: [c, c, c]
add_params.data_gen_params.data_params["err_dist"] = "normal"
add_params.data_gen_params.beta_fn = lambda cs: _get_gt_beta(cs, 
                                                             add_params.data_gen_params.d, 
                                                             add_params.data_gen_params.npts, 
                                                             fct=2)

add_params.setting = "nm3"
add_params.sel_idx =  np.arange(3, add_params.data_gen_params.d)
#add_params.can_lams = [0.001, 0.1, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7,  2, 8]
add_params.SIS_ratio = 0.2
settingnm3.update(add_params)

## -------

settingnm3b = edict(deepcopy(base_params))
add_params = edict({})
add_params.data_gen_params = edict(deepcopy(base_params.data_gen_params))
add_params.data_gen_params.n = 200 # num of data obs to be genareted
add_params.data_gen_params.cs_fn = lambda c: [c, c, c]
add_params.data_gen_params.data_params["err_dist"] = "t"
add_params.data_gen_params.beta_fn = lambda cs: _get_gt_beta(cs, 
                                                             add_params.data_gen_params.d, 
                                                             add_params.data_gen_params.npts, 
                                                             fct=2)
add_params.setting = "nm3b"
add_params.sel_idx =  np.arange(3, add_params.data_gen_params.d)
#add_params.can_lams = [0.001, 0.1, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7,  2, 8]
add_params.SIS_ratio = 0.2
settingnm3b.update(add_params)



#========================================================================================================

#### SIS_ratio = 1
settingnm1c = edict(deepcopy(settingnm1))
settingnm1c.setting = "nm1c"
settingnm1c.SIS_ratio = 1
settingnm1d = edict(deepcopy(settingnm1b))
settingnm1d.setting = "nm1d"
settingnm1d.SIS_ratio = 1

settingnm2c = edict(deepcopy(settingnm2))
settingnm2c.setting = "nm2c"
settingnm2c.SIS_ratio = 1
settingnm2d = edict(deepcopy(settingnm2b))
settingnm2d.setting = "nm2d"
settingnm2d.SIS_ratio = 1

settingnm3c = edict(deepcopy(settingnm3))
settingnm3c.setting = "nm3c"
settingnm3c.SIS_ratio = 1
settingnm3d = edict(deepcopy(settingnm3b))
settingnm3d.setting = "nm3d"
settingnm3d.SIS_ratio = 1

#### n = 500
settingnm1a = edict(deepcopy(settingnm1))
settingnm1a.setting = "nm1a"
#settingnm1a.can_lams = [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 3, 4, 5, 6]
settingnm1a.data_gen_params.n = 500
settingnm1e = edict(deepcopy(settingnm1b))
#settingnm1e.can_lams = [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 3, 4, 5, 6]
settingnm1e.setting = "nm1e"
settingnm1e.data_gen_params.n = 500

settingnm2a = edict(deepcopy(settingnm2))
settingnm2a.setting = "nm2a"
#settingnm2a.can_lams = [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 3, 4, 5, 6]
settingnm2a.data_gen_params.n = 500
settingnm2e = edict(deepcopy(settingnm2b))
settingnm2e.setting = "nm2e"
#settingnm2e.can_lams = [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 3, 4, 5, 6]
settingnm2e.data_gen_params.n = 500

settingnm3a = edict(deepcopy(settingnm3))
settingnm3a.setting = "nm3a"
settingnm3a.data_gen_params.n = 500
settingnm3e = edict(deepcopy(settingnm3b))
settingnm3e.setting = "nm3e"
settingnm3e.data_gen_params.n = 500


#### n = 400
settingnm1f = edict(deepcopy(settingnm1))
settingnm1f.setting = "nm1f"
settingnm1f.data_gen_params.n = 400
settingnm1g = edict(deepcopy(settingnm1b))
settingnm1g.setting = "nm1g"
settingnm1g.data_gen_params.n = 400

settingnm2f = edict(deepcopy(settingnm2))
settingnm2f.setting = "nm2f"
settingnm2f.data_gen_params.n = 400
settingnm2g = edict(deepcopy(settingnm2b))
settingnm2g.setting = "nm2g"
settingnm2g.data_gen_params.n = 400

settingnm3f = edict(deepcopy(settingnm3))
settingnm3f.setting = "nm3f"
settingnm3f.data_gen_params.n = 400
settingnm3g = edict(deepcopy(settingnm3b))
settingnm3g.setting = "nm3g"
settingnm3g.data_gen_params.n = 400



#========================================================================================================
settings = edict()
settings.nm1 = settingnm1
settings.nm1a = settingnm1a
settings.nm1b = settingnm1b
settings.nm1c = settingnm1c
settings.nm1d = settingnm1d
settings.nm1e = settingnm1e
settings.nm1f = settingnm1f
settings.nm1g = settingnm1g

settings.nm2 = settingnm2
settings.nm2a = settingnm2a
settings.nm2b = settingnm2b
settings.nm2c = settingnm2c
settings.nm2e = settingnm2e
settings.nm2f = settingnm2f
settings.nm2g = settingnm2g

settings.nm3 = settingnm3
settings.nm3a = settingnm3a
settings.nm3b = settingnm3b
settings.nm3c = settingnm3c
settings.nm3e = settingnm3e
settings.nm3f = settingnm3f
settings.nm3g = settingnm3g
