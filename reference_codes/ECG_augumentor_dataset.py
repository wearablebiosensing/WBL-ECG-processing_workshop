# %%
!pip install resampy

# %%
# %% [markdown]
# # ECG WebDataset Generator (v10) - With Massive Jitter & Full Augmentation Tracking
# 
# This script directly processes WFDB records (HDF5 bypass) and exports PyTorch WebDataset `.tar` archives.
# It features robust peak detection, complex augmenting, and a massive dynamic time jitter to perfectly train translation-invariant RPNet models.

# %%
import os
import sys
import warnings
import numpy as np
import pandas as pd
import wfdb
import wfdb.processing
import webdataset as wds
import matplotlib.pyplot as plt

try:
    from wfdb import RecordNotFoundError
except ImportError:
    try:
        from wfdb.io._record import RecordNotFoundError
    except ImportError:
        RecordNotFoundError = FileNotFoundError

import scipy.signal
import resampy
import pickle
from tqdm.notebook import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import gaussian_kde
from scipy.linalg import toeplitz
import neurokit2 as nk
import pywt
import uuid

warnings.filterwarnings("ignore")

# %%
# --- 1. Configuration ---
class Config:
    BASE_PATH = "/work/pi_kunalm_uri_edu/Arm_ECG_RPnet/"
    RAW_DATA_ROOT = os.path.join(BASE_PATH, 'raw-datasets')
    
    # NEW WEBDATASET EXPORT PATH
    PROCESSED_DATA_ROOT = os.path.join(BASE_PATH, 'webdataset_export_v10')
    NOISE_MODELS_PATH = os.path.join(PROCESSED_DATA_ROOT, 'noise_models.pkl')

    DATASETS_TO_PROCESS = ['mitdb', 'fantasia', 'chfdb', 'apnea-ecg', 'qtdb', 'ludb', 'ltafdb', 'ltdb']
    NSTDB_NAME = 'nstdb'
    RECORD_SUBSET_FRACTION = None
    BEAT_SUBSET_FRACTION = 1.0

    TARGET_FS = 100.0
    WINDOW_SIZE_SAMPLES = 500
    
    # User's massive Jitter window integrated!
    MAX_JITTER_SAMPLES = 200  
    EXP_DT_ALPHA = 0.1

    INTENSE_TSN_PROBABILITY = 0.8
    TSN_SNR_DB_STANDARD = (-15, 5)
    TSN_SNR_DB_INTENSE = (-30, 2)

    LEAD_MAP = {
        'mitdb': 'MLII', 'ltafdb': 'ECG', 'fantasia': 'ECG', 'chfdb': 'ECG1',
        'apnea-ecg': 'ECG', 'qtdb': 'MLII', 'ludb': 'i', 'ecgiddb': 'ECG I filtered',
        'svdb': 'ECG1', 'incartdb': 'i', 'nstdb': 'MLII', 'ltdb': 'ECG1'
    }
    WAVELET_NAME = 'sym4'
    WAVELET_LEVELS = 4
    WAVELET_PT_SUPPRESS_FACTOR = 0.75
    GAUSSIAN_NOISE_SCALE = 0.6
    
    # Enable to dump matplotlib PNGs for visual inspection before heavy cluster runs
    VERIFY_PLOTS = False

config = Config()
os.makedirs(config.PROCESSED_DATA_ROOT, exist_ok=True)

# %%
print(f"Reading raw data from: {config.RAW_DATA_ROOT}")
print(f"Writing WDS Tarballs to: {config.PROCESSED_DATA_ROOT}")
print(f"Max Jitter Shift: ±{config.MAX_JITTER_SAMPLES} samples")

# %%

# %%
# --- 2. Filter Bank Definition ---
def butter_sos(order, Wn, btype, fs):
    nyquist = fs / 2.0; Wn = np.asarray(Wn)
    if np.any(Wn <= 0) or np.any(Wn >= nyquist):
        Wn = np.clip(Wn, 0.01, nyquist - 0.01)
        if Wn.size > 1 and Wn[0] >= Wn[1]: Wn[0] = Wn[1] * 0.9; Wn = np.clip(Wn, 0.01, nyquist - 0.01)
    return scipy.signal.butter(order, Wn, btype=btype, fs=fs, output='sos')

def sosfiltfilt(sos, x):
    if x is None or len(x) == 0: return np.array([], dtype=np.float32)
    min_len = 3 * max(len(sos_i) for sos_i in sos if hasattr(sos_i, '__len__')) + 1
    if len(x) < min_len:
        padlen = min_len
        try: x_padded = np.pad(x, (padlen, padlen), mode='reflect'); filtered = scipy.signal.sosfiltfilt(sos, x_padded); return filtered[padlen:-padlen].astype(np.float32)
        except Exception: return np.zeros_like(x, dtype=np.float32)
    try: filtered = scipy.signal.sosfiltfilt(sos, x); return filtered.astype(np.float32)
    except Exception: return np.zeros_like(x, dtype=np.float32)

class FilterBank:
    def __init__(self, fs):
        self.fs = fs
        self.s_clean = butter_sos(5, [0.5, 45.0], 'bandpass', fs)
        self.s_emg   = butter_sos(5, [10.0, 48.0], 'bandpass', fs)
FB = FilterBank(fs=config.TARGET_FS)

# %%
# --- 3. Label Generation (VERIFIED ALGORITHM FOR ALL PEAKS IN WINDOW) ---
def generate_multipeak_distance_transform(window_size, peak_locations):
    """
    CRITICAL FIX VERIFIED: computes DT relative to ALL peaks that landed inside the current `window_size` (0 to 499).
    If a peak fell out of bounds (<0 or >=window_size), it is explicitly ignored ensuring the label perfectly matches the data view.
    """
    x_indices = np.arange(window_size)
    if peak_locations is None or not hasattr(peak_locations, '__len__') or len(peak_locations) == 0:
        return np.full(window_size, window_size, dtype=np.float32)
    try:
        peak_array = np.array(peak_locations, dtype=int)
        # Filter Peaks ONLY within the visual window bounds
        peak_array = peak_array[(peak_array >= 0) & (peak_array < window_size)]
        
        if peak_array.size == 0: return np.full(window_size, window_size, dtype=np.float32)
        
        # Calculate distance to nearest peak for every sample in the window!
        dist_matrix = np.abs(x_indices[:, np.newaxis] - peak_array)
        min_dist = np.min(dist_matrix, axis=1)
        return min_dist.astype(np.float32)
    except Exception as e:
        return np.full(window_size, window_size, dtype=np.float32)

def generate_exponential_distance_transform(window_size, peak_locations, alpha):
    linear_dist = generate_multipeak_distance_transform(window_size, peak_locations)
    if np.all(linear_dist == window_size): return np.zeros(window_size, dtype=np.float32)
    exp_term = np.exp(-np.clip(alpha * linear_dist, 0, 50))
    return (1.0 - exp_term).astype(np.float32)


# %%
# --- 4. Authentic Augmentations Pipeline ---
def convolve_with_random_kernel(signal, rng):
    if len(signal) == 0: return signal, "conv_k:err_zero_input"
    kernel_len=rng.choice([3,5,7]); kernel=rng.standard_normal(kernel_len);
    kernel_sum = np.sum(kernel);
    if np.abs(kernel_sum) < 1e-6: kernel.fill(1.0/kernel_len)
    else: kernel=kernel/kernel_sum
    try: convolved=scipy.signal.convolve(signal, kernel, mode='same')
    except Exception as conv_err: return signal, "conv_k:err_convolve"
    if np.any(~np.isfinite(convolved)): return signal, "conv_k:err_nan_output"
    return convolved.astype(np.float32), f"conv_k:{kernel_len}"
def random_morphology_transform(signal, fs, rng, max_strength=0.1): 
    n=len(signal);
    if n == 0: return signal, "morph:err_zero_input"
    t=np.arange(n)/fs; warped_signal=signal.copy(); num_sines=rng.integers(1,3)
    signal_std=np.std(signal);
    if signal_std < 1e-6: return signal, "morph:zero_std"
    log=[]
    for _ in range(num_sines):
        f0=rng.uniform(0.1,0.7); phase=rng.uniform(0,2*np.pi); amplitude=rng.uniform(0.01,max_strength)*signal_std
        warp=amplitude*np.sin(2*np.pi*f0*t+phase); warped_signal+=warp; log.append(f"morph_f:{f0:.2f}")
    if np.any(~np.isfinite(warped_signal)): return signal, "morph:err_nan_output"
    return warped_signal.astype(np.float32), ";".join(log)

def simulate_chest_to_arm_scaling(signal, rng): scale=rng.uniform(0.07,0.5); return signal*scale, f"arm_scale:{scale:.2f}"
def random_rescale(signal, rng): scale=rng.uniform(0.7,1.4); return signal*scale, f"random_rescale:{scale:.2f}"

def add_lpc_kde_noise(signal, fs, rng, local_lpc_kde_models, intensity='standard'):
    if not local_lpc_kde_models or not isinstance(local_lpc_kde_models, dict) or not local_lpc_kde_models: return signal, "tsn:no_models"
    valid_noise_types = list(local_lpc_kde_models.keys())
    if not valid_noise_types: return signal, "tsn:empty_models"
    n=len(signal); noise_type=rng.choice(valid_noise_types)
    lpc_coeffs, kde_model = local_lpc_kde_models[noise_type]
    try: synthetic_residual=kde_model.resample(n).reshape(-1)
    except Exception as kde_err: return signal, f"tsn:kde_err_{noise_type}"
    a_filter=np.concatenate(([1.0],lpc_coeffs)); synthetic_noise=scipy.signal.lfilter([1.0],a_filter,synthetic_residual).astype(np.float32)
    noise_std = np.std(synthetic_noise)
    if noise_std < 1e-6: return signal, f"tsn:zero_std_{noise_type}"
    synthetic_noise /= noise_std
    signal_rms=np.sqrt(np.mean(signal**2))+1e-6
    snr_range = config.TSN_SNR_DB_INTENSE if intensity=='intense' else config.TSN_SNR_DB_STANDARD
    intensity_label = "intense" if intensity=='intense' else "standard"
    snr_db=rng.uniform(snr_range[0],snr_range[1]); noise_rms=signal_rms/(10**(snr_db/20))
    result = signal+(synthetic_noise*noise_rms);
    if np.any(~np.isfinite(result)): return signal, f"tsn:nan_out_{noise_type}"
    return result, f"tsn:{intensity_label}:{noise_type}:{snr_db:.1f}dB"

def add_complex_baseline_wander(signal, fs, rng, severity=1.0):
    n=len(signal); t=np.arange(n)/fs; wander=np.zeros(n,dtype=np.float32)
    num_sines=rng.integers(2,4); mad=np.median(np.abs(signal-np.median(signal)))+1e-6
    for _ in range(num_sines): f0=rng.uniform(0.05,0.4); phase=rng.uniform(0,2*np.pi); amplitude=rng.uniform(0.2,0.8)*mad*severity; wander+=amplitude*np.sin(2*np.pi*f0*t+phase)
    result = signal+wander;
    if np.any(~np.isfinite(result)): return signal
    return result

def add_structured_emg_noise(signal, fs, rng, fb:FilterBank, severity=1.0):
    n=len(signal); x=signal.copy(); raw_noise=rng.standard_normal(n).astype(np.float32)
    try: emg_core=sosfiltfilt(fb.s_emg,raw_noise)
    except Exception: return signal
    emg_std = np.std(emg_core);
    if emg_std < 1e-6: return signal
    emg_core /= emg_std; env=np.zeros(n,dtype=np.float32); i=0
    while i < n:
        gap=int(rng.uniform(0.2,3.0)*fs); i+=gap;
        if i>=n: break
        dur=int(rng.uniform(0.5,10.0)*fs); j=min(n,i+dur)
        if j<=i: continue
        burst_len=j-i; burst_shape=scipy.signal.windows.tukey(burst_len,alpha=rng.uniform(0.1,0.4))
        burst_time = burst_len/fs
        if burst_time < 1e-6: continue
        mod_freq=rng.uniform(0.5,2.0)/burst_time; t_burst=np.arange(burst_len)/fs
        modulator=0.7+0.3*np.sin(2*np.pi*mod_freq*t_burst+rng.uniform(0,2*np.pi))
        env[i:j]=burst_shape*modulator
        if rng.random()<(0.15*severity): x[i:j]*=rng.uniform(0.2,0.5)
        i=j
    signal_rms=np.sqrt(np.mean(signal**2))+1e-6; snr_db=rng.uniform(-5*severity,10/(severity+1e-6))
    emg_rms=signal_rms/(10**(snr_db/20)); emg_noise=emg_core*env*emg_rms; result = x+emg_noise
    if np.any(~np.isfinite(result)): return signal
    return result

def add_step_recovery_transients(x, fs, rng, signal_std, severity=1.0):
    n=len(x); y=x.astype(np.float32).copy(); rate=rng.uniform(0.1,1.0)*severity; n_events=rng.poisson(rate*(n/fs))
    if n_events == 0: return y
    for _ in range(n_events):
        t0=rng.integers(0,n);
        if t0 >= n: continue
        amp_scale=rng.uniform(0.5,4.0)*signal_std*severity; A=rng.laplace(0.0,amp_scale)*rng.choice([-1,1]); tau=rng.uniform(100,600)/1000.0
        if fs*tau < 1e-9: tau = 100/1000.0
        exponent = -np.clip(np.arange(n-t0)/(fs*tau), -50, 50); decay = np.exp(exponent); y[t0:] += A*decay
    if np.any(~np.isfinite(y)): return x
    return y

def add_clipping(x, rng, severity=1.0):
    if rng.random()<(0.1*severity):
        signal_std = np.std(x);
        if signal_std < 1e-6: return x
        threshold=rng.uniform(1.5,2.5)*signal_std; return np.clip(x,-threshold,threshold)
    return x

def apply_gain_envelope(x, fs, rng, depth=(0.05,0.30), fc_hz=0.2):
    n=len(x); alpha=np.exp(-2*np.pi*fc_hz/fs); g=np.empty(n,np.float32); d=rng.uniform(*depth); prev=0.0
    for i in range(n): prev=alpha*prev+(1-alpha)*rng.standard_normal(); g[i]=1.0+d*prev
    result = (g*x).astype(np.float32);
    if np.any(~np.isfinite(result)): return x
    return result

def add_pli(x, fs, rng):
    mains = 50.0 if rng.random() < 0.5 else 60.0
    n = len(x)
    if n == 0: return x
    t = np.arange(n, dtype=np.float32) / float(fs)
    phi = np.cumsum(rng.normal(0.0, 0.01, n)).astype(np.float32)
    x_np = np.asarray(x, dtype=np.float32)
    med = np.median(x_np)
    mad = np.median(np.abs(x_np - med)).astype(np.float32)
    sigma_robust = np.float32(1.4826) * mad
    x_rms = float(sigma_robust) if sigma_robust > 1e-8 else float(np.sqrt(np.mean((x_np - med) ** 2) + 1e-12))
    r = float(rng.uniform(0.03, 0.15))
    target_rms = r * x_rms
    m = 0.20
    rms_factor = (0.5 * (1.0 + (m ** 2) / 2.0)) ** 0.5
    A0 = target_rms / rms_factor if rms_factor > 1e-6 else 0.0
    A = (A0 * (1.0 + m * np.sin(2.0 * np.pi * 0.1 * t))).astype(np.float32)
    result = x_np + (A * np.sin(2.0 * np.pi * mains * t + phi)).astype(np.float32)
    return result if np.all(np.isfinite(result)) else x

def add_gaussian_noise(signal, rng, scale=config.GAUSSIAN_NOISE_SCALE):
    std_dev = np.std(signal)
    if std_dev < 1e-6: return signal, "gauss:zero_std"
    scales_ = rng.uniform(0.09, scale) 
    noise_std = rng.uniform(0.4, 1.3) * scales_ * std_dev
    noise = rng.normal(0, noise_std, size=signal.shape)
    result = signal + noise
    if np.any(~np.isfinite(result)): return signal, "gauss:nan_out"
    return result.astype(np.float32), f"gauss:{noise_std/std_dev:.2f}xSD"

def suppress_p_t_waves_wavelet(signal, fs, wavelet=config.WAVELET_NAME, level=config.WAVELET_LEVELS, compress_base=config.WAVELET_PT_SUPPRESS_FACTOR, rng=None):
    sig_len = len(signal)
    try:
        max_level = pywt.dwt_max_level(sig_len, wavelet)
        current_level = min(level, max_level)
        if current_level <= 1: return signal, f"wavelet_pt:low"
        coeffs = pywt.wavedec(signal, wavelet, level=current_level)
        supp_factor = rng.uniform(compress_base, 0.8) if rng else np.random.default_rng().uniform(compress_base, 0.8) 
        coeffs[0] *= supp_factor 
        for i in range(1, current_level // 2 + 1):
            coeffs[i] *= supp_factor 
        reconstructed_signal = pywt.waverec(coeffs, wavelet)
        reconstructed_signal = reconstructed_signal[:sig_len]
        if np.any(~np.isfinite(reconstructed_signal)): return signal, "wavelet_pt:nan_out"
        return reconstructed_signal.astype(np.float32), f"wavelet_pt{current_level}:supp{supp_factor:.1f}"
    except Exception as wav_err:
        return signal, "wavelet_pt:error"

def make_augmented_training_pair(clean_win, rng, fb: FilterBank, local_lpc_kde_models):
    fs = fb.fs; aug_log = []; original_signal_std = np.std(clean_win) + 1e-6;
    x = clean_win.copy().astype(np.float32)
    if not np.all(np.isfinite(x)): return clean_win, "augment_error:nan_input"

    try:
        x, scale_log1 = simulate_chest_to_arm_scaling(x, rng); aug_log.append(scale_log1)
        x, scale_log2 = random_rescale(x, rng); aug_log.append(scale_log2)

        tsn_intensity = 'intense' if rng.random() < config.INTENSE_TSN_PROBABILITY else 'standard'
        x, tsn_log = add_lpc_kde_noise(x, fs, rng, local_lpc_kde_models, intensity=tsn_intensity); aug_log.append(tsn_log)
        
        x, gauss_log = add_gaussian_noise(x, rng)
        aug_log.append(gauss_log)
        
        if rng.random() < 0.95: 
            x = add_pli(x, fs, rng); aug_log.append("pli")
        if rng.random() < 0.85: 
             x, pt_log = suppress_p_t_waves_wavelet(x, fs, rng=rng); aug_log.append(pt_log)

        if rng.random() < 0.5: x, log = random_morphology_transform(x, fs, rng); aug_log.append(log)
        if rng.random() < 0.4: x, log = convolve_with_random_kernel(x, rng); aug_log.append(log)

        difficulty_choice=rng.choice(['easy','medium','hard','extreme'],p=[0.1,0.3,0.4,0.2])
        severity={'easy':1.0,'medium':1.5,'hard':2.0,'extreme':3.0}[difficulty_choice]
        aug_log.append(f"diff:{difficulty_choice}")

        if rng.random() < 0.5: x = add_complex_baseline_wander(x, fs, rng, severity); aug_log.append(f"bw:{severity:.1f}")
        if rng.random() < 0.5: x = add_structured_emg_noise(x, fs, rng, fb, severity); aug_log.append(f"emg:{severity:.1f}")
        if rng.random() < 0.7: x = apply_gain_envelope(x, fs, rng); aug_log.append("gain")
        if rng.random() < 0.7: x = add_step_recovery_transients(x, fs, rng, original_signal_std, severity); aug_log.append(f"step:{severity:.1f}")
        if difficulty_choice in ['hard', 'extreme']:
            if rng.random() < 0.5: x = add_clipping(x, rng, severity); aug_log.append(f"clip:{severity:.1f}")
        if rng.random() < 0.3: x = -x; aug_log.append("flip")
        if rng.random() < 0.1:
            if not x.flags.writeable: x = x.copy()
            x = np.flip(x); aug_log.append("rev")

        return x.astype(np.float32), ";".join(filter(None, aug_log))

    except Exception as aug_err:
        return clean_win.astype(np.float32), f"augment_error_fallback_basic"

# %%
# --- 5. Support Models ---
def build_lpc_kde_models(nstdb_path, lead_map, target_fs, fb):
    print("\n--- Building Generative Noise Models from NSTDB ---")
    if os.path.exists(config.NOISE_MODELS_PATH):
        print(f"✅ Found existing models: {config.NOISE_MODELS_PATH}")
        try:
            with open(config.NOISE_MODELS_PATH, 'rb') as f: return pickle.load(f)
        except Exception: pass
    if not os.path.isdir(nstdb_path): return None
    noise_record_names = ['em', 'ma']; noise_signals = {}
    for rec_name in noise_record_names:
        try:
            rec_path_base = os.path.join(nstdb_path, rec_name)
            rec = wfdb.rdrecord(rec_path_base)
            signal_raw = rec.p_signal[:, 0]
            signal_resampled = resampy.resample(signal_raw.astype(np.float64), rec.fs, target_fs)
            noise_signals[rec_name] = signal_resampled
        except Exception as e: pass
    lpc_kde_models = {}; lpc_order = 3
    for noise_type, full_noise_signal in noise_signals.items():
        try:
            full_noise = full_noise_signal.astype(np.float64)
            r = np.correlate(full_noise, full_noise, mode='full'); r = r[len(full_noise)-1:]
            R_matrix = toeplitz(r[:lpc_order]); r_vec = r[1:lpc_order+1]
            lpc_coeffs = np.linalg.solve(R_matrix, r_vec)
            pred = scipy.signal.lfilter(np.concatenate(([0], -lpc_coeffs)), [1.0], full_noise)
            if len(pred) != len(full_noise): pred = np.pad(pred, (0, len(full_noise) - len(pred)), 'constant')
            residual = full_noise - pred
            kde_model = gaussian_kde(residual)
            lpc_kde_models[noise_type] = (lpc_coeffs.astype(np.float32), kde_model)
        except Exception: pass
    try:
        with open(config.NOISE_MODELS_PATH,'wb') as f: pickle.dump(lpc_kde_models, f)
    except Exception: pass
    return lpc_kde_models

def find_consensus_r_peaks(signal, fs):
    peak_lists = []
    try:
        xqrs_peaks = wfdb.processing.xqrs_detect(sig=signal.astype(np.float64), fs=fs)
        if xqrs_peaks is not None and len(xqrs_peaks) > 0: peak_lists.append(xqrs_peaks)
    except Exception: pass

    try:
        _, nk_info = nk.ecg_peaks(signal, sampling_rate=fs, method='promac')
        promac_peaks = nk_info['ECG_R_Peaks']
        if promac_peaks is not None and len(promac_peaks) > 0: peak_lists.append(promac_peaks)
    except Exception:
         try:
             _, nk_info_def = nk.ecg_peaks(signal, sampling_rate=fs, method='neurokit')
             nk_peaks = nk_info_def['ECG_R_Peaks']
             if nk_peaks is not None and len(nk_peaks) > 0: peak_lists.append(nk_peaks)
         except Exception: pass

    if not peak_lists: return np.array([], dtype=int)
    combined = np.unique(np.concatenate(peak_lists))
    if len(combined) == 0: return np.array([], dtype=int)
    final_peaks = np.sort(combined.astype(int))
    min_dist = int(0.2 * fs)
    if len(final_peaks) <= 1: return final_peaks
    
    accepted_peaks = [final_peaks[0]]
    for i in range(1, len(final_peaks)):
        if final_peaks[i] - accepted_peaks[-1] > min_dist:
            accepted_peaks.append(final_peaks[i])
    return np.array(accepted_peaks, dtype=int)

# %%
# --- 6. Worker Function DIRECT TO WEBDATASET ---
def process_record_to_tar(args):
    """
    Reads a record, slices raw 500-sample sequences (incorporating massive random jitter),
    applies complex augmentation pipelines, and writes outputs securely to a
    local WebDataset tar instance managed uniquely by this worker.
    """
    (db_name, rec_path, out_dir, worker_id, seed, beat_sub, local_lpc_kde_models) = args
    rng = np.random.default_rng(seed)
    fb = FilterBank(fs=config.TARGET_FS)
    p_id = f"{db_name}_{os.path.basename(rec_path)}"
    
    # We write directly to WDS: 1 Tar Archive per Worker/Record combination.
    tar_path = os.path.join(out_dir, f"arm_ecg_rpnet_{db_name}_{worker_id:06d}.tar")
    tar_writer = None
    samples_written = 0

    try:
        rec = wfdb.rdrecord(rec_path)
        lead = config.LEAD_MAP.get(db_name, rec.sig_name[0])
        idx = rec.sig_name.index(lead) if lead in rec.sig_name else 0
        x_raw = rec.p_signal[:, idx].astype(np.float32)
        fs0 = float(rec.fs)
        
        clip_low, clip_high = np.percentile(x_raw[np.isfinite(x_raw)], [0.5, 99.5]) if np.any(np.isfinite(x_raw)) else (-1e6, 1e6)
        x_raw_clipped = np.clip(x_raw, clip_low, clip_high)
        x_resampled = resampy.resample(x_raw_clipped.astype(np.float64), fs0, config.TARGET_FS)
        x_clean_filtered = sosfiltfilt(fb.s_clean, x_resampled)
        if np.any(~np.isfinite(x_clean_filtered)) or np.std(x_clean_filtered) < 1e-6:
            return None

        # --- Peak Detection ---
        ann_exts = ['atr', 'qrs', 'ecg']; found_ann_file = False
        rpeaks_abs = np.array([], dtype=int); src='unknown'
        for e in ann_exts:
             ann_path = f"{rec_path}.{e}"
             if os.path.exists(ann_path):
                 try:
                     ann = wfdb.rdann(rec_path, e)
                     if hasattr(ann, 'sample') and len(ann.sample) > 0:
                         rpeaks_abs = np.round(np.asarray(ann.sample)*(config.TARGET_FS/fs0)).astype(int)
                         src = f'ground_truth_{e}'
                         found_ann_file = True; break
                 except Exception: pass
        if not found_ann_file or len(rpeaks_abs) == 0:
             rpeaks_abs = find_consensus_r_peaks(x_clean_filtered, config.TARGET_FS)
             src = 'consensus_robust'

        if len(rpeaks_abs) == 0: return None
        rpeaks_abs = rpeaks_abs[(rpeaks_abs >= 0) & (rpeaks_abs < len(x_clean_filtered))]

        L = config.WINDOW_SIZE_SAMPLES; windows = []
        for r_abs in rpeaks_abs:
            # HUGE RANDOM JITTER (±125 from User Request configuration)
            center = r_abs + rng.integers(-config.MAX_JITTER_SAMPLES, config.MAX_JITTER_SAMPLES + 1)
            s, e = center - L // 2, center + L // 2
            if s < 0 or e > len(x_clean_filtered): continue
            
            # This mathematically captures ANY AND ALL peak hits inside this precise Jitter Window!
            # If the jitter pulled an adjacent beat into view, it IS naturally included here!
            peaks_in_window_rel = list(rpeaks_abs[(rpeaks_abs >= s) & (rpeaks_abs < e)] - s)
            if not peaks_in_window_rel: continue
            
            # Identify which original target peak is in here (to verify jitter extent)
            target_peak_rel = r_abs - s
            
            windows.append({
                "start": s, "end": e, 
                "peaks_rel": peaks_in_window_rel,
                "target_peak_rel": target_peak_rel
            })

        if 0 < beat_sub < 1.0 and windows:
            if beat_sub < 1.0: rng.shuffle(windows)
            windows = windows[:max(1, int(np.ceil(len(windows)*beat_sub)))]
        if not windows: return None

        # Initialize WDS Tar Writer Only After Verification
        tar_writer = wds.TarWriter(tar_path)

        for win_idx, win in enumerate(windows):
            clean_w_slice = x_clean_filtered[win["start"]:win["end"]]
            if len(clean_w_slice) != L: continue
            clean_w = clean_w_slice.astype(np.float32)
            
            # The Distance label uses `peaks_rel` which holds EVERY PEAK in bounds (0 to 499)
            dist_label = generate_multipeak_distance_transform(L, win["peaks_rel"])
            if np.any(~np.isfinite(dist_label)) or np.any(~np.isfinite(clean_w)): continue

            # Augmentation Engine
            noisy_w, aug_log = make_augmented_training_pair(clean_w, rng, fb, local_lpc_kde_models) 

            if config.VERIFY_PLOTS and win_idx < 5:
                # Save first 5 windows as diagnostic plot
                plt.figure(figsize=(12, 4))
                plt.plot(noisy_w, label="Noisy Input", color='red', alpha=0.8, linewidth=1)
                plt.plot(clean_w, label="Clean Ground Truth", color='black', alpha=0.5, linewidth=1.5)
                
                # Plot DT and actual peak markers
                plt.plot(dist_label, label="Target DT", color='blue', linestyle='dashed', linewidth=2)
                
                # Scatter mark exactly where the algorithm thinks peaks are:
                for p in win["peaks_rel"]:
                     plt.scatter(p, dist_label[p] if p < len(dist_label) else 0, color='magenta', s=100, zorder=5, marker='x')

                # Show how far the original center shifted
                actual_shift = win["target_peak_rel"] - (L // 2)

                plt.title(f"[{db_name}] {p_id} {win_idx} | Jitter Shift: {actual_shift}\nAuth Augs: {aug_log}\nMagenta 'X' = Active Labeling Peaks")
                plt.legend()
                plt.tight_layout()
                os.makedirs(os.path.join(config.PROCESSED_DATA_ROOT, 'verify_plots'), exist_ok=True)
                plt.savefig(os.path.join(config.PROCESSED_DATA_ROOT, 'verify_plots', f"verify_{p_id}_{win_idx}.png"))
                plt.close('all')

            if np.all(np.isfinite(noisy_w)):
                key = f"{p_id}_{samples_written:06d}"
                sample = {
                    "__key__": key,
                    "x_noisy.npy": noisy_w.astype(np.float32),
                    "x_clean.npy": clean_w.astype(np.float32),
                    "y_dt.npy": dist_label.astype(np.float32),
                    "participant_id.txt": p_id,
                    "r_peak_source.txt": src,
                    "dataset.txt": db_name,
                    # Capturing FULL SQI and Artifact Pipeline Text
                    "augmentation.txt": aug_log
                }
                tar_writer.write(sample)
                samples_written += 1

    except Exception as e:
        print(f"❌ {p_id} Failure: {e}", file=sys.stderr)
        return None
    finally:
        if tar_writer: tar_writer.close()
        # Clean up empty tarballs if everything dropped
        if samples_written == 0 and os.path.exists(tar_path):
            os.remove(tar_path)

    if samples_written > 0:
        return {'tar_path': os.path.basename(tar_path), 'num_rows': samples_written, 'participant_id': p_id}
    else:
        return None


# %%

# %%
# --- 7. Main Execution Loop ---
def main():
    import argparse
    
    # We test for sys.argv being invoked, but handle Notebook gracefully too:
    if "ipykernel" not in sys.modules and len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="ArmECG WebDataset Generator")
        parser.add_argument("--verify", action="store_true", help="Generate Visual Matplotlib plots of the first few samples to manually verify augmentations and jitter shift.")
        args, _ = parser.parse_known_args()
        
        if args.verify:
            config.VERIFY_PLOTS = True
    
    # Optional explicitly enable Verify for notebook runs:
    # config.VERIFY_PLOTS = True
            
    if config.VERIFY_PLOTS:
        print("⚠️ VERIFICATION MODE ENABLED: Will export diagnostic plots to verify_plots/!")
        # Drastically reduce scope for quick visual test
        config.DATASETS_TO_PROCESS = config.DATASETS_TO_PROCESS[:1]
        config.RECORD_SUBSET_FRACTION = 0.05
        
    lpc_models = build_lpc_kde_models(os.path.join(config.RAW_DATA_ROOT,config.NSTDB_NAME),config.LEAD_MAP, config.TARGET_FS, FB)
    manifest, jobs = [], []
    
    print("\n--- Discovering records ---")
    for i, db in enumerate(config.DATASETS_TO_PROCESS):
        db_path = os.path.join(config.RAW_DATA_ROOT, db)
        if not os.path.isdir(db_path): continue
        recs = sorted([os.path.splitext(f)[0] for f in os.listdir(db_path) if f.endswith('.hea')])
        if db == 'apnea-ecg': recs = [r for r in recs if not(r.endswith('r') or r.endswith('er'))]
        
        if config.RECORD_SUBSET_FRACTION is not None:
             recs = recs[:max(1, int(len(recs) * config.RECORD_SUBSET_FRACTION))]
             
        for j, rec_name in enumerate(recs):
            worker_id = i * 10000 + j
            jobs.append((db, os.path.join(db_path, rec_name), config.PROCESSED_DATA_ROOT, worker_id, 42+worker_id, config.BEAT_SUBSET_FRACTION, lpc_models))

    print(f"\nCreated {len(jobs)} jobs. Writing directly to native WebDataset (.tar).")
    num_workers = min(50, (os.cpu_count() or 1)) if not config.VERIFY_PLOTS else 1

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        future_to_job = {ex.submit(process_record_to_tar, job): job for job in jobs}
        for future in tqdm(as_completed(future_to_job), total=len(jobs), desc="Encoding Direct WDS"):
            try:
                res = future.result()
                if res: manifest.append(res)
            except Exception as exc:
                pass

    if manifest:
        df = pd.DataFrame(manifest)
        path = os.path.join(config.PROCESSED_DATA_ROOT, 'manifest.parquet')
        df.to_parquet(path, index=False)
        print(f"\n✅ Total Rows Extracted: {df['num_rows'].sum()} across {len(manifest)} native TAR archives.")

if __name__ == "__main__":
    main()


# %%


# %%



