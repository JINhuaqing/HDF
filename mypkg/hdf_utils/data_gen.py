# this file contains fns for generating data for simulation
import numpy as np
from numbers import Number
from pathlib import Path
from utils.misc import load_pkl
from constants import DATA_ROOT

AD_ts = load_pkl(DATA_ROOT/"AD_vs_Ctrl_ts/AD88_all.pkl")
Ctrl_ts = load_pkl(DATA_ROOT/"AD_vs_Ctrl_ts/Ctrl92_all.pkl")
ts_data = np.concatenate([AD_ts, Ctrl_ts], axis=0)
std_ts_data = (ts_data - ts_data.mean(axis=2)[:, :, np.newaxis])/ts_data.std(axis=2)[:, :, np.newaxis]
#_cur_dir = Path(__file__).parent
#_database_dir = Path(_cur_dir/"../../data/fooof_data_ADvsCtrl")
#_database_fil = Path(_cur_dir/"../../data/ctrl_vs_AD_nooutlier.pkl")
#_data = load_pkl(_database_fil)

def obt_maskmat(sel_seqs):
    """Return the mask matrix for generating new MEG data
    """
    num_sel, num_pt = sel_seqs.shape
    seg_len = int(num_pt/num_sel)
    mask_mat = np.zeros_like(sel_seqs)
    for ix in range(num_sel):
        low, up = ix*seg_len, (ix+1)*seg_len
        if ix == (num_sel-1):
            up = num_pt
        mask_mat[ix, low:up] = 1
    return mask_mat


def gen_ts_single(base_data, num_sel, sub_idx):
    """
    This function is to generate one single MEG seq in time domain. 
        args:
            base_data: The dataset used for generaing
            num_sel: Num of seqs used for jointing
            sub_idx: The data idx used for generating
    """
    tmp_stds = np.zeros(10)
    # some MEG dataset is very bad, to avoid zero-std
    while np.min(tmp_stds) <= 1e-5:
        sel_roi_idx = np.sort(np.random.choice(base_data.shape[1], num_sel, False))
        sel_seqs = base_data[sub_idx, sel_roi_idx, :]
        maskmat = obt_maskmat(sel_seqs)
        sel_seqs_masked = sel_seqs * maskmat
        
        # make each segment standardized
        sel_seqs_masked[maskmat==0] = np.NAN
        tmp_stds = np.nanstd(sel_seqs_masked, axis=1)
        tmp_means = np.nanmean(sel_seqs_masked, axis=1)
    
    simu_seq = np.nansum((sel_seqs_masked - tmp_means[:, np.newaxis])/tmp_stds[:, np.newaxis], axis=0)
    
    simu_seq = (simu_seq - simu_seq.mean())/simu_seq.std()
    
    return simu_seq

def gen_simu_ts(n, d, num_sel, base_data=None, decimate_rate=10):
    """generate time_series data based on AD vs ctrl
        args:
            n: Num of sps to generate
            d: Num of ROIs
            num_sel: Num of selected curve to get the shape
            base_data: The dataset used for generaing
        return:
            simu_ts: n x d x len array
    """
    if base_data is None:
        base_data = std_ts_data[:, :, ::decimate_rate]
    simu_tss = []
    for ix in range(n):
        sel_sub_idx = np.random.choice(base_data.shape[0], 1)
        cur_ts = []
        for iy in range(d):
            cur_ts.append(gen_ts_single(base_data, num_sel, sel_sub_idx))
        cur_ts = np.array(cur_ts)
        simu_tss.append(cur_ts)
    return np.array(simu_tss)



def gen_covs(n, types_):
    """Generate the covariates for simulated datasets
        args:
            n: The num of obs to generate
            types_: A list of types for each col
                    num: Discrete with num classes
                    "c": Continuous type
                    "int" intercept
    """
    covs = []
    for type_ in types_:
        if isinstance(type_, Number):
            cur_cov = np.random.choice(int(type_), n)
        else:
            type_ = type_.lower()
            if type_.startswith("c"):
                cur_cov = np.random.randn(n)
            elif type_.startswith("int"):
                cur_cov = np.ones(n)
            else:
                pass
        covs.append(cur_cov)
    covs = np.array(covs).T
    return covs


