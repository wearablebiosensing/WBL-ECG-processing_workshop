"""
Configurable ECG preprocessing pipeline.
Supports: notch, bandpass, baseline removal, wavelet denoising,
SSQ-CWT band reconstruction, auto-inversion.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import scipy.signal as sig


@dataclass
class PreprocessConfig:
    """Configuration for the ECG preprocessing pipeline."""
    # Notch filter
    apply_notch: bool = True
    notch_freq: float = 60.0        # Hz (50 for EU, 60 for US)
    notch_quality: float = 30.0

    # Bandpass filter
    apply_bandpass: bool = True
    bp_lowcut: float = 0.5          # Hz
    bp_highcut: float = 40.0        # Hz
    bp_order: int = 4

    # Baseline removal (double median)
    apply_baseline: bool = True
    baseline_win1_ms: float = 200.0  # first pass window
    baseline_win2_ms: float = 600.0  # second pass window

    # Wavelet denoising (optional)
    apply_wavelet: bool = False
    wavelet_method: str = "ssq"     # "ssq" | "pywt" | "none"
    wavelet_fmin: float = 5.0       # Hz (QRS band lower)
    wavelet_fmax: float = 40.0      # Hz (QRS band upper)

    # Auto-inversion (skewness-based)
    auto_invert: bool = True


def notch_filter(signal_data, fs, freq=60.0, quality=30.0):
    """IIR notch filter. Skips if freq >= 0.9 * Nyquist."""
    nyquist = fs / 2.0
    if freq >= 0.9 * nyquist:
        return signal_data
    b, a = sig.iirnotch(w0=freq, Q=quality, fs=fs)
    return sig.filtfilt(b, a, signal_data)


def bandpass_filter(signal_data, fs, lowcut=0.5, highcut=40.0, order=4):
    """Butterworth SOS bandpass (zero-phase)."""
    nyquist = fs / 2.0
    lo = max(lowcut / nyquist, 0.001)
    hi = min(highcut / nyquist, 0.999)
    if lo >= hi:
        return signal_data
    sos = sig.butter(order, [lo, hi], btype="band", output="sos")
    return sig.sosfiltfilt(sos, signal_data)


def remove_baseline(signal_data, fs, win1_ms=200.0, win2_ms=600.0):
    """Double-pass median filter baseline removal."""
    from scipy.ndimage import median_filter

    win1 = int(win1_ms * fs / 1000)
    win2 = int(win2_ms * fs / 1000)
    # Ensure odd window sizes
    win1 = win1 | 1
    win2 = win2 | 1

    baseline1 = median_filter(signal_data, size=win1)
    baseline2 = median_filter(baseline1, size=win2)
    return signal_data - baseline2


def get_wavelet_heatmap(signal_data, fs, fmin=None, fmax=None):
    """Compute Wavelet transform matrix and frequencies for visualization."""
    try:
        import ssqueezepy as sq
    except ImportError:
        return None, None

    Wx, scales = sq.cwt(signal_data, wavelet="morlet", fs=fs)
    # Estimate frequencies from scales
    wavelet_obj = sq.Wavelet("morlet")
    peak_freq = 1.0
    if hasattr(wavelet_obj, "config"):
        config = wavelet_obj.config
        if "mu" in config:
            peak_freq = config["mu"] / (2 * np.pi)
    freqs = (peak_freq * fs) / scales

    if fmin is not None and fmax is not None:
        mask = (freqs >= fmin) & (freqs <= fmax)
        return Wx[mask, :], freqs[mask]
    return Wx, freqs


def wavelet_denoise_ssq(signal_data, fs, fmin=5.0, fmax=40.0):
    """SSQ-CWT frequency-band reconstruction (QRS band)."""
    try:
        import ssqueezepy as sq
    except ImportError:
        print("ssqueezepy not available, skipping SSQ-CWT denoising.")
        return signal_data

    Wx, scales = sq.cwt(signal_data, wavelet="morlet", fs=fs)
    # Estimate frequencies from scales
    wavelet_obj = sq.Wavelet("morlet")
    peak_freq = 1.0
    if hasattr(wavelet_obj, "config"):
        config = wavelet_obj.config
        if "mu" in config:
            peak_freq = config["mu"] / (2 * np.pi)
    freqs = (peak_freq * fs) / scales

    # Frequency band mask
    mask = (freqs >= fmin) & (freqs <= fmax)
    mask = mask[:, np.newaxis]
    Wx_filtered = Wx * mask

    reconstructed = sq.icwt(Wx_filtered, wavelet="morlet", scales=scales)
    return np.real(reconstructed)


def wavelet_denoise_pywt(signal_data, fs, wavelet="sym4", level=4):
    """PyWavelets DWT soft-threshold denoising."""
    try:
        import pywt
    except ImportError:
        print("pywt not available, skipping wavelet denoising.")
        return signal_data

    coeffs = pywt.wavedec(signal_data, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = 0.6 * sigma * np.sqrt(2 * np.log(len(signal_data)))
    new_coeffs = [pywt.threshold(c, threshold, mode="soft") for c in coeffs]
    rec = pywt.waverec(new_coeffs, wavelet)
    # waverec may produce an extra sample
    return rec[:len(signal_data)]


def auto_invert_ecg(ecg, label="signal"):
    """Invert ECG if skewness suggests inverted polarity.

    QRS complexes should produce negative skewness (sharp downward R-peaks
    are rarer than the standard R-up morphology).
    """
    from scipy.stats import skew
    s = skew(ecg)
    if s < -0.1:
        return -ecg, True
    return ecg, False


def preprocess_ecg(ecg, fs, config=None):
    """Apply the full configurable preprocessing pipeline.

    Parameters
    ----------
    ecg : np.ndarray
        Raw ECG signal.
    fs : float
        Sampling rate in Hz.
    config : PreprocessConfig or None
        Pipeline configuration. If None, uses defaults.

    Returns
    -------
    ecg_clean : np.ndarray
        Preprocessed ECG.
    info : dict
        Processing metadata (inverted, steps applied).
    """
    if config is None:
        config = PreprocessConfig()

    ecg_clean = ecg.astype(np.float64).copy()
    info = {"steps": [], "inverted": False}

    # Handle NaN/Inf
    bad = np.isnan(ecg_clean) | np.isinf(ecg_clean)
    if np.any(bad):
        good = np.where(~bad)[0]
        if len(good) > 0:
            ecg_clean[bad] = np.interp(np.where(bad)[0], good, ecg_clean[good])
        else:
            ecg_clean[:] = 0
        info["steps"].append("interpolate_invalid")

    # 1. Notch
    if config.apply_notch:
        ecg_clean = notch_filter(ecg_clean, fs, config.notch_freq, config.notch_quality)
        info["steps"].append(f"notch_{config.notch_freq}Hz")

    # 2. Bandpass
    if config.apply_bandpass:
        ecg_clean = bandpass_filter(ecg_clean, fs,
                                    config.bp_lowcut, config.bp_highcut, config.bp_order)
        info["steps"].append(f"bandpass_{config.bp_lowcut}-{config.bp_highcut}Hz")

    # 3. Baseline removal
    if config.apply_baseline:
        ecg_clean = remove_baseline(ecg_clean, fs,
                                    config.baseline_win1_ms, config.baseline_win2_ms)
        info["steps"].append("baseline_removal")

    # 4. Wavelet denoising (optional)
    if config.apply_wavelet:
        if config.wavelet_method == "ssq":
            ecg_clean = wavelet_denoise_ssq(ecg_clean, fs,
                                            config.wavelet_fmin, config.wavelet_fmax)
            info["steps"].append(f"ssq_cwt_{config.wavelet_fmin}-{config.wavelet_fmax}Hz")
        elif config.wavelet_method == "pywt":
            ecg_clean = wavelet_denoise_pywt(ecg_clean, fs)
            info["steps"].append("pywt_dwt_denoise")

    # 5. Auto-inversion
    if config.auto_invert:
        ecg_clean, was_inverted = auto_invert_ecg(ecg_clean)
        info["inverted"] = was_inverted
        if was_inverted:
            info["steps"].append("auto_inverted")

    return ecg_clean, info
