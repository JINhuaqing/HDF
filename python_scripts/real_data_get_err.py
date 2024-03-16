#!/usr/bin/env python
# coding: utf-8

# This file contains python code to check the hypothesis testing

# In[1]:


RUN_PYTHON_SCRIPT = False
OUTLIER_IDXS = dict(AD=[49], ctrl=[14, 19, 30, 38])
SAVED_FOLDER = "real_data_nlinear_nostd"
#SAVED_FOLDER = "real_data_nlinear_nostd_X1err"
DATA = ["AD88_matlab_1-45.pkl", "Ctrl92_matlab_1-45.pkl"]


# In[2]:


import sys
sys.path.append("../mypkg")


# In[3]:


import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from numbers import Number
import itertools

from easydict import EasyDict as edict
from tqdm import trange, tqdm
from scipy.io import loadmat
from pprint import pprint
from IPython.display import display
from joblib import Parallel, delayed



from constants import DATA_ROOT, RES_ROOT, FIG_ROOT, MIDRES_ROOT
from hdf_utils.likelihood import obt_lin_tm
from utils.misc import save_pkl, load_pkl


# In[6]:


plt.style.use(FIG_ROOT/"base.mplstyle")


# In[7]:


torch.set_default_tensor_type(torch.DoubleTensor)
def_dtype = torch.get_default_dtype()


# In[ ]:





# # Load  data and prepare

# In[8]:


data_root = DATA_ROOT/"AD_vs_Ctrl_PSD/";
AD_PSD = load_pkl(data_root/DATA[0]);
ctrl_PSD = load_pkl(data_root/DATA[1]);
df0= pd.read_csv(data_root/"AllDataBaselineOrdered_r_ncpt.csv");
df1= pd.read_csv(data_root/"AllDataBaselineOrdered_r_ncpt_more.csv");
df1 = df1.set_index("RID")
df0 = df0.set_index("RID");
df1 = df1.reindex(df0.index)
baseline = df1
baseline["Gender_binary"] = baseline["Gender"].apply(lambda x: 0 if x=="female" else 1);
baseline["Grp_binary"] = baseline["Grp"].apply(lambda x: 1 if x=="AD" else 0);


# In[9]:


# The outlier idxs to rm
outlier_idxs = np.concatenate([OUTLIER_IDXS["AD"], len(AD_PSD.PSDs)+np.array(OUTLIER_IDXS["ctrl"])])
outlier_idxs = outlier_idxs.astype(int)

# make PSD in dB and std 
raw_X = np.concatenate([AD_PSD.PSDs, ctrl_PSD.PSDs]); #n x d x npts
X_dB = 10*np.log10(raw_X);
outlier_idxs2 = np.where(X_dB.mean(axis=(1, 2))<0)
#X = (X_dB - X_dB.mean(axis=-1, keepdims=1))/X_dB.std(axis=-1, keepdims=1);
X = X_dB

Y = np.array(baseline["MMSE"])[:X.shape[0]];
# if logi
#Yb = np.array(baseline["Grp_binary"])[:X.shape[0]];

sel_cov = ["Gender_binary", "MEG_Age","Education"]
Z_raw = np.array(baseline[sel_cov])[:X.shape[0]];

grp_idxs = np.array(baseline["Grp"])[:X.shape[0]];


outlier_idxs = np.sort(np.union1d(outlier_idxs, outlier_idxs2))


# remove outliers
X = np.delete(X, outlier_idxs, axis=0)
Y = np.delete(Y, outlier_idxs, axis=0)
Z_raw = np.delete(Z_raw, outlier_idxs, axis=0)
grp_idxs = np.delete(grp_idxs, outlier_idxs, axis=0)


#remove nan
keep_idx = ~np.bitwise_or(np.isnan(Y), np.isnan(Z_raw.sum(axis=1)));
X = X[keep_idx];
Y = Y[keep_idx]
Z_raw = Z_raw[keep_idx]
grp_idxs = grp_idxs[keep_idx]

Z = np.concatenate([np.ones((Z_raw.shape[0], 1)), Z_raw], axis=1); # add intercept


freqs = AD_PSD.freqs;
# only take PSD between [2, 35] freqs of interest
X = X[:, :, np.bitwise_and(freqs>=2, freqs<=35)]
X = X/X.mean()


print(X.shape, Y.shape, Z.shape)

all_data = edict()
all_data.X = torch.tensor(X)
#all_data.X = torch.tensor(X+np.random.randn(*X.shape)*0.1)
all_data.Y = torch.tensor(Y)
all_data.Z = torch.tensor(Z)


# In[10]:


# # Param and fns

# ## Params

# In[11]:


from easydict import EasyDict as edict
from hdf_utils.fns_sinica import coef_fn, fourier_basis_fn
from copy import deepcopy
from scenarios.base_params import get_base_params

base_params = get_base_params("linear") 
base_params.data_params = edict()
base_params.data_params.d = all_data.X.shape[1]
base_params.data_params.n = all_data.X.shape[0]
base_params.data_params.npts = all_data.X.shape[-1]
base_params.data_params.freqs = AD_PSD.freqs[np.bitwise_and(freqs>=2, freqs<=35)]

base_params.can_Ns = [4, 6, 8, 10, 12, 14]
base_params.SIS_params = edict({"SIS_pen": 0.02, "SIS_basis_N":8, "SIS_ws":"simpson"})
base_params.opt_params.beta = 10 
base_params.can_lams = [0.60,  0.80,  1,  1.2, 1.4, 1.6, 2.0, 4.0]


setting = edict(deepcopy(base_params))
add_params = edict({})
add_params.setting = "real_data_linear"
add_params.SIS_ratio = 1
setting.update(add_params)


# In[12]:


save_dir = RES_ROOT/SAVED_FOLDER
if not save_dir.exists():
    save_dir.mkdir()


# In[13]:


bands_cut = edict()
bands_cut.delta = [2, 4]
bands_cut.theta = [4, 8]
bands_cut.alpha = [8, 12]
bands_cut.beta = [12, 35]
bands_cut.pts = [4, 8, 12, 35]

cut_pts = np.abs(freqs.reshape(-1, 1) - bands_cut.pts).argmin(axis=0)


# # Analysis

# In[14]:




def _filname2set(fil):
    """Based on the file name, reture the setting"""
    res = edict()
    for curstr in fil.stem.split("-"):
        vs = curstr.split("_")
        if vs[0] == "lam":
            res[vs[0]] = int(vs[1])/1000
        else:
            res[vs[0]] = int(vs[1])
    return res
tNs = []
tlams = []
for fil in save_dir.glob(f"roi_*.pkl"):
    tmp = _filname2set(fil)
    tNs.append(tmp["N"])
    tlams.append(tmp["lam"])
np.sort(np.unique(tNs)),  np.sort(np.unique(tlams))



from scipy.stats import chi2
lams = np.sort(np.unique(tlams))
Ns = np.array([4, 6, 8, 10, 12, 14])
log_cv = np.log(0.05/68);
print(lams)

from itertools import product
all_coms = product(range(68), lams, Ns)
def _extract_err(roi_ix, lam, N):
    fil = save_dir/f"roi_{roi_ix}-lam_{lam*1000:.0f}-N_{N:.0f}_fit.pkl"
    hdf_res = load_pkl(fil, verbose=False);
    mse = np.mean((hdf_res.cv_Y_est - hdf_res.Y.numpy())**2)
    return (roi_ix, lam, N), mse

with Parallel(n_jobs=20) as parallel:
    errs = parallel(delayed(_extract_err)(roi_ix=roi_ix, lam=lam, N=N) 
                    for roi_ix, lam, N in tqdm(all_coms, total=68*len(Ns)*len(lams)))
save_pkl(RES_ROOT/f"{SAVED_FOLDER}/errs_mse_roi_N_lam.pkl", errs, is_force=True);

