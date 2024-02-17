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
base_params.data_gen_params.data_params={"srho":0.3, "sigma2":1}
base_params.data_gen_params.data_type = base_params.model_type
base_params.can_Ns = [4, 6, 8, 10, 12]
base_params.SIS_params = edict({"SIS_pen": 1, "SIS_basis_N":4})

def _get_gt_beta_wrapper(d, npts, fct=2):
    def _get_gt_beta(cs):
        x = np.linspace(0, 1, npts)
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
settingns1 = edict(deepcopy(base_params))
add_params = edict({})
add_params.data_gen_params = edict(deepcopy(base_params.data_gen_params))
add_params.data_gen_params.n = 200 # num of data obs to be genareted
add_params.data_gen_params.cs_fn = lambda c: [c, 0, 0]
add_params.data_gen_params.npts = 100 # num of pts to evaluate X(s)
add_params.data_gen_params.data_params["err_dist"] = "normal"
add_params.data_gen_params.beta_fn = _get_gt_beta_wrapper(add_params.data_gen_params.d, add_params.data_gen_params.npts, fct=2)

add_params.setting = "ns1"
add_params.sel_idx =  np.arange(1, add_params.data_gen_params.d)
add_params.can_lams = [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1, 2, 8]
add_params.can_Ns = [4, 6, 8, 10, 12]
add_params.SIS_ratio = 0.2
settingns1.update(add_params)


settingns1t = edict(deepcopy(base_params))
add_params = edict({})
add_params.data_gen_params = edict(deepcopy(base_params.data_gen_params))
add_params.data_gen_params.n = 200 # num of data obs to be genareted
add_params.data_gen_params.cs_fn = lambda c: [c, 0, 0]
add_params.data_gen_params.npts = 10 # num of pts to evaluate X(s)
add_params.data_gen_params.data_params["err_dist"] = "normal"
add_params.data_gen_params.beta_fn = _get_gt_beta_wrapper(add_params.data_gen_params.d, add_params.data_gen_params.npts, fct=2)

add_params.setting = "ns1t"
add_params.sel_idx =  np.arange(1, add_params.data_gen_params.d)
add_params.can_lams = [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1, 2, 8]
add_params.can_Ns = [4, 6, 8, 10, 12]
add_params.SIS_ratio = 0.2
settingns1t.update(add_params)

## -------

settingns1b = edict(deepcopy(base_params))
add_params = edict({})
add_params.data_gen_params = edict(deepcopy(base_params.data_gen_params))
add_params.data_gen_params.n = 200 # num of data obs to be genareted
add_params.data_gen_params.cs_fn = lambda c: [c, 0, 0]
add_params.data_gen_params.data_params["err_dist"] = "t"
add_params.data_gen_params.beta_fn =  _get_gt_beta_wrapper(add_params.data_gen_params.d, add_params.data_gen_params.npts, fct=2)

add_params.setting = "ns1b"
add_params.sel_idx =  np.arange(1, add_params.data_gen_params.d)
add_params.can_lams = [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1, 2, 8]
add_params.can_Ns = [4, 6, 8, 10, 12]
add_params.SIS_ratio = 0.2
settingns1b.update(add_params)


#========================================================================================================

settingns2 = edict(deepcopy(base_params))
add_params = edict({})
add_params.data_gen_params = edict(deepcopy(base_params.data_gen_params))
add_params.data_gen_params.n = 200 # num of data obs to be genareted
add_params.data_gen_params.cs_fn = lambda c: [c+0.5, 0.5, 0]
add_params.data_gen_params.data_params["err_dist"] = "normal"
add_params.data_gen_params.beta_fn = lambda cs: _get_gt_beta(cs, 
                                                             add_params.data_gen_params.d, 
                                                             add_params.data_gen_params.npts, 
                                                             fct=2)

add_params.setting = "ns2"
add_params.sel_idx =  np.arange(2, add_params.data_gen_params.d)
add_params.can_lams = [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1, 2, 8] 
add_params.can_Ns = [4, 6, 8, 10, 12]
add_params.SIS_ratio = 0.2
settingns2.update(add_params)


## -------

settingns2b = edict(deepcopy(base_params))
add_params = edict({})
add_params.data_gen_params = edict(deepcopy(base_params.data_gen_params))
add_params.data_gen_params.n = 200 # num of data obs to be genareted
add_params.data_gen_params.cs_fn = lambda c: [c+0.5, 0.5, 0]
add_params.data_gen_params.data_params["err_dist"] = "t"
add_params.data_gen_params.beta_fn = lambda cs: _get_gt_beta(cs, 
                                                             add_params.data_gen_params.d, 
                                                             add_params.data_gen_params.npts, 
                                                             fct=2)

add_params.setting = "ns2b"
add_params.sel_idx =  np.arange(2, add_params.data_gen_params.d)
add_params.can_lams = [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1, 2, 8] 
add_params.can_Ns = [4, 6, 8, 10, 12]
add_params.SIS_ratio = 0.2
settingns2b.update(add_params)

#========================================================================================================

settingns3 = edict(deepcopy(base_params))
add_params = edict({})
add_params.data_gen_params = edict(deepcopy(base_params.data_gen_params))
add_params.data_gen_params.n = 200 # num of data obs to be genareted
add_params.data_gen_params.cs_fn = lambda c: [c, c, c]
add_params.data_gen_params.data_params["err_dist"] = "normal"
add_params.data_gen_params.beta_fn = lambda cs: _get_gt_beta(cs, 
                                                             add_params.data_gen_params.d, 
                                                             add_params.data_gen_params.npts, 
                                                             fct=2)

add_params.setting = "ns3"
add_params.sel_idx =  np.arange(3, add_params.data_gen_params.d)
add_params.can_lams = [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1, 2, 8] 
add_params.can_Ns = [4, 6, 8, 10, 12]
add_params.SIS_ratio = 0.2
settingns3.update(add_params)

## -------

settingns3b = edict(deepcopy(base_params))
add_params = edict({})
add_params.data_gen_params = edict(deepcopy(base_params.data_gen_params))
add_params.data_gen_params.n = 200 # num of data obs to be genareted
add_params.data_gen_params.cs_fn = lambda c: [c, c, c]
add_params.data_gen_params.data_params["err_dist"] = "t"
add_params.data_gen_params.beta_fn = lambda cs: _get_gt_beta(cs, 
                                                             add_params.data_gen_params.d, 
                                                             add_params.data_gen_params.npts, 
                                                             fct=2)
add_params.setting = "ns3b"
add_params.sel_idx =  np.arange(3, add_params.data_gen_params.d)
add_params.can_lams = [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1, 2, 8] 
add_params.can_Ns = [4, 6, 8, 10, 12]
add_params.SIS_ratio = 0.2
settingns3b.update(add_params)



#========================================================================================================

#### SIS_ratio = 1
settingns1c = edict(deepcopy(settingns1))
settingns1c.setting = "ns1c"
settingns1c.SIS_ratio = 1
settingns1d = edict(deepcopy(settingns1b))
settingns1d.setting = "ns1d"
settingns1d.SIS_ratio = 1

settingns2c = edict(deepcopy(settingns2))
settingns2c.setting = "ns2c"
settingns2c.SIS_ratio = 1
settingns2d = edict(deepcopy(settingns2b))
settingns2d.setting = "ns2d"
settingns2d.SIS_ratio = 1

settingns3c = edict(deepcopy(settingns3))
settingns3c.setting = "ns3c"
settingns3c.SIS_ratio = 1
settingns3d = edict(deepcopy(settingns3b))
settingns3d.setting = "ns3d"
settingns3d.SIS_ratio = 1

#### n = 500
settingns1a = edict(deepcopy(settingns1))
settingns1a.setting = "ns1a"
settingns1a.data_gen_params.n = 500
settingns1e = edict(deepcopy(settingns1b))
settingns1e.setting = "ns1e"
settingns1e.data_gen_params.n = 500

settingns2a = edict(deepcopy(settingns2))
settingns2a.setting = "ns2a"
settingns2a.data_gen_params.n = 500
settingns2e = edict(deepcopy(settingns2b))
settingns2e.setting = "ns2e"
settingns2e.data_gen_params.n = 500

settingns3a = edict(deepcopy(settingns3))
settingns3a.setting = "ns3a"
settingns3a.data_gen_params.n = 500
settingns3e = edict(deepcopy(settingns3b))
settingns3e.setting = "ns3e"
settingns3e.data_gen_params.n = 500


#### n = 400
settingns1f = edict(deepcopy(settingns1))
settingns1f.setting = "ns1f"
settingns1f.data_gen_params.n = 400
settingns1g = edict(deepcopy(settingns1b))
settingns1g.setting = "ns1g"
settingns1g.data_gen_params.n = 400

settingns2f = edict(deepcopy(settingns2))
settingns2f.setting = "ns2f"
settingns2f.data_gen_params.n = 400
settingns2g = edict(deepcopy(settingns2b))
settingns2g.setting = "ns2g"
settingns2g.data_gen_params.n = 400

settingns3f = edict(deepcopy(settingns3))
settingns3f.setting = "ns3f"
settingns3f.data_gen_params.n = 400
settingns3g = edict(deepcopy(settingns3b))
settingns3g.setting = "ns3g"
settingns3g.data_gen_params.n = 400



#========================================================================================================
settings = edict()
settings.ns1 = settingns1
settings.ns1t = settingns1t
settings.ns1a = settingns1a
settings.ns1b = settingns1b
settings.ns1c = settingns1c
settings.ns1d = settingns1d
settings.ns1e = settingns1e
settings.ns1f = settingns1f
settings.ns1g = settingns1g

settings.ns2 = settingns2
settings.ns2a = settingns2a
settings.ns2b = settingns2b
settings.ns2c = settingns2c
settings.ns2e = settingns2e
settings.ns2f = settingns2f
settings.ns2g = settingns2g

settings.ns3 = settingns3
settings.ns3a = settingns3a
settings.ns3b = settingns3b
settings.ns3c = settingns3c
settings.ns3e = settingns3e
settings.ns3f = settingns3f
settings.ns3g = settingns3g
