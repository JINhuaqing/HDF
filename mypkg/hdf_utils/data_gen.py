# this file contains fns for generating data for simulation
import numpy as np
from numbers import Number
from fooof import FOOOF
from pathlib import Path
from utils.misc import load_pkl

_cur_dir = Path(__file__).parent
_database_dir = Path(_cur_dir/"../../data/fooof_data_ADvsCtrl")
_database_fil = Path(_cur_dir/"../../data/ctrl_vs_AD_nooutlier.pkl")
_data = load_pkl(_database_fil)

def gen_simu_psd(M, d, num_sel, noise_sd=0.2):
    """generate data based on AD vs ctrl
        args:
            M: Num of sps to generate
            d: Num of ROIs
            num_sel: Num of selected curve to get the shape
            noise_sd: SD of the noise added to the psd
        return:
            simu_psd: M x d x 40 array
    """
    simu_psd = []
    for roi_idx in range(d):
        roi_dat = np.log10(_data.psd[roi_idx%68]) # if d > 68, from 0 again
        
        std_logp = roi_dat.std(axis=0)
        
        psds = []
        for ix in range(M):
            cur_sel_idxs = np.random.choice(168, num_sel, replace=False)
            psd_std = roi_dat[:, cur_sel_idxs].mean(axis=-1)
            psd_std = (psd_std - psd_std.mean())/psd_std.std()
            
            cur_mean = np.random.rand(1) 
            std_sel_idx = np.random.choice(168, 1, replace=False)
            cur_std = std_logp[std_sel_idx]
            
            psd_raw = psd_std * cur_std  + cur_mean
            psd_out = psd_raw + np.random.randn(len(psd_raw))*noise_sd
            psds.append(psd_out)
        psds = np.array(psds)
        simu_psd.append(psds)
    simu_psd = np.array(simu_psd).transpose((1, 0, 2))
    return simu_psd



def gen_covs(M, types_):
    """Generate the covariates for simulated datasets
        args:
            M: The num of obs to generate
            types_: A list of types for each col
                    num: Discrete with num classes
                    "c": Continuous type
                    "int" intercept
    """
    covs = []
    for type_ in types_:
        if isinstance(type_, Number):
            cur_cov = np.random.choice(int(type_), M)
        else:
            type_ = type_.lower()
            if type_.startswith("c"):
                cur_cov = np.random.randn(M)
            elif type_.startswith("int"):
                cur_cov = np.ones(M)
            else:
                pass
        covs.append(cur_cov)
    covs = np.array(covs).T
    return covs




# below are the functions for generate data with fooof
# I do not use them
#def _jitter_fn(n, jit_noises):
#    if isinstance(jit_noises, Number):
#        jit_noises = np.ones(n)*jit_noises
#    jit_noises = np.array(jit_noises)
#    rvs = np.random.rand(n) - 0.5
#    fcts = 1+rvs*jit_noises
#    return fcts
#    
#
#def fooof_rec(freqs, aperiodic_params_, peak_params_):
#    """recontruct the PSD curve with fooof output in log10(power) scale
#        args:
#            freqs: the frequency pts at which you'd like to evaluate
#            aperiodic_params_: exponential part of fooof
#            peak_params_: peak part of fooof
#    """
#    if len(aperiodic_params_) == 3:
#        offset, knee, exponent = aperiodic_params_ 
#    else:
#        offset, exponent = aperiodic_params_ 
#        knee = 0
#    ps = offset - np.log10(knee+freqs**exponent)
#    for peak_param_ in peak_params_:
#        cen, pw, bw = peak_param_ 
#        ps = ps + pw*np.exp(-(freqs-cen)**2*2/bw/bw)
#    return ps
#
#def obt_psm_raw(raw_psd, freqs, is_knee=False):                                                                                                                            
#    """
#         extract features from Power spectrum models
#         two components:
#             1. Peaks, [center freq, power, bandwidth]
#             2. Aperiodic part: offset and Exponent
#         args:
#             raw_psd: the power of spectrum to extract features
#             freqs: corresponding freqs
#    """
#    freq_range = [np.amin(freqs), np.amax(freqs)]
#     
#    # to smooth the raw psd
#    if is_knee:
#        aperiodic_mode='knee'
#    else:
#        aperiodic_mode='fixed'
#    # the fooof obj
#    fm = FOOOF(peak_width_limits=[2*np.diff(freqs)[0], 12.0], aperiodic_mode=aperiodic_mode)
#    fm.fit(freqs, raw_psd, freq_range)
#    return fm
#
#def gen_simu_psd_1roi(N, freqs, jit_noise, noise=0.05, roi_idx=1):
#    """Generate simulated psds from observed data obs_psd in log(power) for one ROI
#        Note that you will map freqs to [0, 1]
#        args:
#        N: Num of psds to generate
#        freqs: The frequency for the observed psd
#        jit_noise: The level of noise fct multiplying on fooof parameters
#        noise: The random noise level added on the final PSD
#        roi_idx: The roi_idx, 1-68
#    """
#    fil = list(_database_dir.glob(f"ROI{roi_idx}.pkl"))[0]
#    data_base = load_pkl(fil)
#
#
#    max_freq_cv = np.max(freqs)
#    peaks_params_pool = np.concatenate(data_base.peaks_params)
#    sorted_peaks_params_pool = np.array(sorted(peaks_params_pool, key=lambda x: x[0]))
#    
#    cen_loc_diff = np.concatenate([np.diff(ix, axis=0) for ix in data_base.peaks_params])[:, 0]
#    min_diff_cv, _ = np.quantile(cen_loc_diff, [0.025, 0.975])
#    
#    cur_nums_peak = np.array(data_base.num_peaks)[np.random.choice(len(data_base.num_peaks), N, replace=False)]
#    
#    psds = []
#    for num_peak in cur_nums_peak:
#        min_diff = 0
#        max_freq = 1000
#        
#        # get peak params
#        while (min_diff < min_diff_cv) or (max_freq > max_freq_cv):
#            sel_idx = np.sort(np.random.choice(sorted_peaks_params_pool.shape[0], num_peak, replace=False))
#            cur_peak_params = sorted_peaks_params_pool[sel_idx, :]
#            # jit 
#            cur_peak_params = np.array([ix*_jitter_fn(3, jit_noise) for ix in cur_peak_params])
#            
#            if cur_peak_params.shape[0] >1:
#                min_diff = np.min(np.diff(cur_peak_params, axis=0)[:, 0])
#            else:
#                min_diff = 100
#            
#            if cur_peak_params.shape[0] >0:
#                max_freq = np.max(cur_peak_params[:, 0])
#            else:
#                max_freq = 0
#        
#        # get ap params
#        sel_idx = np.random.choice(len(data_base.aperiodics_params), 1, replace=False)[0]
#        cur_aperiodic_params = data_base.aperiodics_params[sel_idx]
#        # jit params
#        cur_aperiodic_params = cur_aperiodic_params * _jitter_fn(3, jit_noise/2)
#        
#        psd = fooof_rec(freqs, cur_aperiodic_params, cur_peak_params)
#        
#        # add noise
#        psd = psd + np.random.randn(len(psd))*noise
#        
#        psds.append(psd)
#        
#    return psds
#
#
#
#def gen_simu_psd(N, num_rois, freqs, jit_noise, noise=0.05):
#    """Generate simulated psds from observed data obs_psd in log(power) for all ROIs
#        Note that you will map freqs to [0, 1]
#        args:
#        N: Num of psds to generate
#        num_rois: num of rois used 
#        freqs: The frequency for the observed psd
#        jit_noise: The sd of noise added on the peak parameters
#        noise: The random noise level added on the final PSD
#    """
#    dats = []
#    for ix in range(1, num_rois+1):
#        dat = gen_simu_psd_1roi(N=N, freqs=freqs, 
#                                jit_noise=jit_noise, 
#                                noise=noise, roi_idx=ix)
#        dats.append(dat)
#    dats = np.array(dats)
#    dats = np.transpose(dats, (1, 0, 2))
#    return dats