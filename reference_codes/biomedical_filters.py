import numpy as np
import scipy.signal as signal
import scipy.ndimage as ndimage

# =========================================================
#                 PPG PROCESSING (MAX30101)
# =========================================================

def filter_ppg_robust(signal_data: np.ndarray, fs: float, 
                      fs_target: float = None,
                      lowcut: float = 0.8, # Increased from 0.5 to remove wander
                      highcut: float = 5.0, 
                      filter_order: int = 6, # Increased from 4 for sharper cutoff
                      method: str = 'linear',
                      wavelet_name: str = 'morlet',
                      use_despiking: bool = True,
                      use_detrending: bool = True) -> np.ndarray:
    """
    Applies a robust cascading filter pipeline optimized for MAX30101 PPG.
    Allows toggling of specific preprocessing steps.
    
    Args:
        signal_data (np.ndarray): The raw PPG signal.
        fs (float): Input sampling frequency.
        fs_target (float): Ignored (Downsampling disabled).
        method (str): 'linear'/'bandpass' or 'wavelet'.
        use_despiking (bool): If True, applies median filter to remove hardware spikes.
        use_detrending (bool): If True, applies linear detrending to remove drift.
    """
    
    # 1. Handle NaNs/Invalid segments
    clean_signal = _handle_invalid_segments(signal_data)

    # 2. Despiking (Hardware specific correction)
    if use_despiking:
        # Uses median filter to squash single-sample spikes common in optical sensors
        # Increased size to 9 to handle "sharp peaks" / motion artifacts better
        ppg_processed = ndimage.median_filter(clean_signal, size=9)
    else:
        ppg_processed = clean_signal

    # 3. Downsampling (Disabled)
    current_fs = fs

    # 4. Linear Detrending
    if use_detrending:
        ppg_detrended = signal.detrend(ppg_processed, type='linear')
    else:
        ppg_detrended = ppg_processed

    # 5. Robust Filtering
    # Map 'bandpass' to 'linear' for compatibility
    if method in ['linear', 'bandpass']:
        ppg_filtered = _filter_linear_ppg(ppg_detrended, current_fs, lowcut, highcut, filter_order)
    elif method == 'wavelet':
        ppg_filtered = _filter_wavelet(ppg_detrended, current_fs, lowcut, highcut, wavelet_name)
    else:
        # Fallback to linear if unknown
        ppg_filtered = _filter_linear_ppg(ppg_detrended, current_fs, lowcut, highcut, filter_order)

    # 6. Amplitude Normalization (Z-score)
    if np.std(ppg_filtered) > 0:
        ppg_normalized = (ppg_filtered - np.mean(ppg_filtered)) / np.std(ppg_filtered)
    else:
        ppg_normalized = ppg_filtered

    return ppg_normalized

# =========================================================
#                 ECG PROCESSING (Upper Arm)
# =========================================================

def filter_ecg_robust(signal_data: np.ndarray, fs: float, 
                      lowcut: float = 0.5, 
                      highcut: float = 40.0, 
                      powerline_freq: float = 50.0,
                      filter_order: int = 2,
                      method: str = 'linear',
                      wavelet_name: str = 'morlet',
                      use_notch: bool = True) -> np.ndarray:
    """
    Applies a robust cascading filter pipeline optimized for Upper Arm ECG.
    Features optional Harmonic Notch filtering for mains noise.
    
    Args:
        use_notch (bool): If True, applies notch filters at powerline_freq and its harmonics.
    """
    
    # 1. Handle NaNs
    clean_signal = _handle_invalid_segments(signal_data)

    if method in ['linear', 'bandpass']:
        return _filter_linear_ecg(clean_signal, fs, lowcut, highcut, powerline_freq, filter_order, use_notch)
    elif method == 'wavelet':
        return _filter_wavelet(clean_signal, fs, lowcut, highcut, wavelet_name)
    else:
        return _filter_linear_ecg(clean_signal, fs, lowcut, highcut, powerline_freq, filter_order, use_notch)
# =========================================================
#                 GRANULAR FILTERS (CHAINABLE)
# =========================================================

def apply_notch(data: np.ndarray, fs: float, freq: float = 50.0) -> np.ndarray:
    """Applies IIR Notch filter at specified frequency."""
    nyquist = fs / 2.0
    if freq >= nyquist: return data
    b_notch, a_notch = signal.iirnotch(w0=freq, Q=30.0, fs=fs)
    return signal.filtfilt(b_notch, a_notch, data)

def apply_bandpass(data: np.ndarray, fs: float, lowcut: float = 0.5, highcut: float = 40.0, order: int = 2) -> np.ndarray:
    """Applies Butterworth Bandpass filter."""
    if np.isnan(data).any(): data = _handle_invalid_segments(data)
    
    # Use SOS for stability
    sos = signal.butter(order, [lowcut, highcut], btype='bandpass', fs=fs, output='sos')
    return signal.sosfiltfilt(sos, data)

def apply_detrend(data: np.ndarray) -> np.ndarray:
    """Applies linear detrending."""
    if np.isnan(data).any(): data = _handle_invalid_segments(data)
    return signal.detrend(data, type='linear')

def apply_wavelet(data: np.ndarray, fs: float, wavelet: str = 'morlet') -> np.ndarray:
    """Applies Wavelet denoising."""
    if np.isnan(data).any(): data = _handle_invalid_segments(data)
    return _filter_wavelet(data, fs, 0.5, 40.0, wavelet)

# =========================================================
#                 SHARED HELPERS & UTILS
# =========================================================

def _handle_invalid_segments(data: np.ndarray) -> np.ndarray:
    """Internal helper to interpolate over NaN or -99 values."""
    signal_copy = np.array(data, dtype=float).copy()
    invalid_mask = np.isnan(signal_copy) | (signal_copy == -99.0)
    
    if np.any(invalid_mask):
        valid_indices = np.where(~invalid_mask)[0]
        invalid_indices = np.where(invalid_mask)[0]
        
        if len(valid_indices) > 0:
            signal_copy[invalid_indices] = np.interp(invalid_indices, valid_indices, signal_copy[valid_indices])
        else:
            return np.zeros_like(signal_copy)
    return signal_copy

def _filter_linear_ppg(data: np.ndarray, fs: float, lowcut: float, highcut: float, filter_order: int) -> np.ndarray:
    """PPG Zero-Phase Butterworth filtering."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Safety check
    if high >= 1.0: high = 0.99
    
    sos = signal.butter(filter_order, [low, high], btype='band', output='sos')
    return signal.sosfiltfilt(sos, data)

def _filter_linear_ecg(signal_data: np.ndarray, fs: float, lowcut: float, highcut: float, 
                       powerline_freq: float, filter_order: int, use_notch: bool = True) -> np.ndarray:
    """ECG standard IIR cascade filtering (Notch + Bandpass)."""
    # 1. High-Pass
    sos_hp = signal.butter(filter_order, lowcut, btype='highpass', fs=fs, output='sos')
    filtered = signal.sosfiltfilt(sos_hp, signal_data)

    # 2. Harmonic Notch (Powerline) - Removes 50Hz, 100Hz, 150Hz...
    if use_notch:
        nyquist = fs / 2.0
        current_freq = powerline_freq
        while current_freq < nyquist:
            b_notch, a_notch = signal.iirnotch(w0=current_freq, Q=30.0, fs=fs)
            filtered = signal.filtfilt(b_notch, a_notch, filtered)
            current_freq += powerline_freq

    # 3. Low-Pass
    sos_lp = signal.butter(filter_order, highcut, btype='lowpass', fs=fs, output='sos')
    filtered = signal.sosfiltfilt(sos_lp, filtered)
    return filtered

def _filter_wavelet(data: np.ndarray, fs: float, lowcut: float, highcut: float, wavelet_name: str) -> np.ndarray:
    """
    Wavelet-based denoising. 
    Tries ssqueezepy (CWT) first, falls back to PyWavelets (DWT).
    """
    # 1. Try CWT (ssqueezepy) - Superior for specific frequency bands
    try:
        import ssqueezepy as sq
        
        Wx, scales = sq.cwt(data, wavelet=wavelet_name, fs=fs)
        wavelet_obj = sq.Wavelet(wavelet_name)
        peak_freq = 1.0
        if hasattr(wavelet_obj, 'config'):
            config = wavelet_obj.config
            if 'mu' in config: 
                peak_freq = config['mu'] / (2 * np.pi)
                
        freqs = (peak_freq * fs) / scales
        valid_freq_mask = (freqs >= lowcut) & (freqs <= highcut)
        valid_freq_mask = valid_freq_mask[:, np.newaxis]
        Wx_filtered = Wx * valid_freq_mask
        
        sigma = np.median(np.abs(Wx[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(data)))
        
        magnitude = np.abs(Wx_filtered)
        Wx_thresholded = np.sign(Wx_filtered) * np.maximum(magnitude - threshold, 0)
        
        reconstructed = sq.icwt(Wx_thresholded, wavelet=wavelet_name, scales=scales)
        return np.real(reconstructed)
        
    except ImportError:
        pass

    # 2. Try DWT (PyWavelets) - Fallback
    try:
        import pywt
        # Use a discrete wavelet suitable for biomedical signals
        dwt_wavelet = 'sym4' 
        
        coeffs = pywt.wavedec(data, dwt_wavelet, level=4)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = 0.6 * sigma * np.sqrt(2 * np.log(len(data)))
        
        new_coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
        return pywt.waverec(new_coeffs, dwt_wavelet)
        
    except ImportError:
        print("Warning: Neither 'ssqueezepy' nor 'pywt' found. Returning raw data.")
        return data