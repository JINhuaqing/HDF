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
base_params.data_gen_params.is_std = False
base_params.data_gen_params.gt_alp0 = np.array([-1, 2]) # we will determine intercept later
base_params.data_gen_params.intercept_cans = np.linspace(-30, 1, 20)
base_params.data_gen_params.data_params={"srho":0.3}
base_params.data_gen_params.data_type = base_params.model_type
base_params.can_Ns = [4, 6, 8, 10, 12]
base_params.SIS_params = edict({"SIS_pen": 1, "SIS_basis_N":4})
base_params.opt_params.beta = 1
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
settingns1 = edict(deepcopy(base_params))
add_params = edict({})
add_params.data_gen_params = edict(deepcopy(base_params.data_gen_params))
add_params.data_gen_params.n = 200 # num of data obs to be genareted
add_params.data_gen_params.cs_fn = lambda c: [c, 0, 0]
add_params.data_gen_params.beta_fn = lambda cs: _get_gt_beta(cs, 
                                                             add_params.data_gen_params.d, 
                                                             add_params.data_gen_params.npts, 
                                                             fct=2)

add_params.setting = "ns1"
add_params.sel_idx =  np.arange(1, add_params.data_gen_params.d)
add_params.can_lams = [0.001, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 1.0, 2.0, 8.0] 
add_params.SIS_ratio = 0.2
settingns1.update(add_params)


#----------------------------------------------------------------------------------------------------

settingns2 = edict(deepcopy(base_params))
add_params = edict({})
add_params.data_gen_params = edict(deepcopy(base_params.data_gen_params))
add_params.data_gen_params.n = 200 # num of data obs to be genareted
add_params.data_gen_params.cs_fn = lambda c: [c+0.5, 0.5, 0]
add_params.data_gen_params.beta_fn = lambda cs: _get_gt_beta(cs, 
                                                             add_params.data_gen_params.d, 
                                                             add_params.data_gen_params.npts, 
                                                             fct=2)

add_params.setting = "ns2"
add_params.sel_idx =  np.arange(2, add_params.data_gen_params.d)
add_params.can_lams = [0.001, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 1.0, 2.0, 8.0] 
add_params.SIS_ratio = 0.2
settingns2.update(add_params)

#----------------------------------------------------------------------------------------------------
settingns3 = edict(deepcopy(base_params))
add_params = edict({})
add_params.data_gen_params = edict(deepcopy(base_params.data_gen_params))
add_params.data_gen_params.n = 200 # num of data obs to be genareted
add_params.data_gen_params.cs_fn = lambda c: [c, c, c]
add_params.data_gen_params.beta_fn = lambda cs: _get_gt_beta(cs, 
                                                             add_params.data_gen_params.d, 
                                                             add_params.data_gen_params.npts, 
                                                             fct=2)

add_params.setting = "ns3"
add_params.sel_idx =  np.arange(3, add_params.data_gen_params.d)
add_params.can_lams = [0.001, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 1.0, 2.0, 8.0] 
add_params.SIS_ratio = 0.2
settingns3.update(add_params)


##====================================================================================================
# n = 500

settingns1a = edict(deepcopy(base_params))
add_params = edict({})
add_params.data_gen_params = edict(base_params.data_gen_params.copy())
add_params.data_gen_params.n = 500 # num of data obs to be genareted
add_params.data_gen_params.cs_fn = lambda c: [c, 0, 0]
add_params.data_gen_params.beta_fn = lambda cs: _get_gt_beta(cs, 
                                                             add_params.data_gen_params.d, 
                                                             add_params.data_gen_params.npts, 
                                                             fct=2)

add_params.setting = "ns1a"
add_params.sel_idx =  np.arange(1, add_params.data_gen_params.d)
add_params.can_lams = [0.001, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 1.0, 2.0, 8.0] 
add_params.SIS_ratio = 0.2
settingns1a.update(add_params)

#----------------------------------------------------------------------------------------------------
settingns2a = edict(deepcopy(base_params))
add_params = edict({})
add_params.data_gen_params = edict(deepcopy(base_params.data_gen_params))
add_params.data_gen_params.n = 500 # num of data obs to be genareted
add_params.data_gen_params.cs_fn = lambda c: [c+0.5, 0.5, 0]
add_params.data_gen_params.beta_fn = lambda cs: _get_gt_beta(cs, 
                                                             add_params.data_gen_params.d, 
                                                             add_params.data_gen_params.npts, 
                                                             fct=2)

add_params.setting = "ns2a"
add_params.sel_idx =  np.arange(2, add_params.data_gen_params.d)
add_params.can_lams = [0.001, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 1.0, 2.0, 8.0] 
add_params.SIS_ratio = 0.2
settingns2a.update(add_params)



#----------------------------------------------------------------------------------------------------

settingns3a = edict(deepcopy(base_params))
add_params = edict({})
add_params.data_gen_params = edict(deepcopy(base_params.data_gen_params))
add_params.data_gen_params.n = 500 # num of data obs to be genareted
add_params.data_gen_params.cs_fn = lambda c: [c, c, c]
add_params.data_gen_params.beta_fn = lambda cs: _get_gt_beta(cs, 
                                                             add_params.data_gen_params.d, 
                                                             add_params.data_gen_params.npts, 
                                                             fct=2)

add_params.setting = "ns3a"
add_params.sel_idx =  np.arange(3, add_params.data_gen_params.d)
add_params.can_lams = [0.001, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 1.0, 2.0, 8.0] 
add_params.SIS_ratio = 0.2
settingns3a.update(add_params)


##====================================================================================================
# SIS_ratio = 1

settingns1c = edict(deepcopy(settingns1))
settingns1c.setting = "ns1c"
settingns1c.SIS_ratio = 1

settingns2c = edict(deepcopy(settingns2))
settingns2c.setting = "ns2c"
settingns2c.SIS_ratio = 1

settingns3c = edict(deepcopy(settingns3))
settingns3c.setting = "ns3c"
settingns3c.SIS_ratio = 1

##====================================================================================================
# n = 400

settingns1f = edict(deepcopy(settingns1a))
settingns1f.setting = "ns1f"
settingns1f.data_gen_params.n = 400 # num of data obs to be genareted

settingns2f = edict(deepcopy(settingns2a))
settingns2f.setting = "ns2f"
settingns2f.data_gen_params.n = 400 # num of data obs to be genareted

settingns3f = edict(deepcopy(settingns3a))
settingns3f.setting = "ns3f"
settingns3f.data_gen_params.n = 400 # num of data obs to be genareted


##====================================================================================================

settings = edict()
settings.ns1 = settingns1
settings.ns2 = settingns2
settings.ns3 = settingns3
settings.ns1a = settingns1a
settings.ns2a = settingns2a
settings.ns3a = settingns3a
settings.ns1c = settingns1c
settings.ns2c = settingns2c
settings.ns3c = settingns3c
settings.ns1f = settingns1f
settings.ns2f = settingns2f
settings.ns3f = settingns3f