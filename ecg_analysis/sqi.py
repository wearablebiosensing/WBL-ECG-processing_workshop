"""
Signal Quality Index (SQI) assessment for ECG signals.

Two complementary SQI metrics:
1. QRS Band Energy Ratio — spectral power in 5-15 Hz / total power
2. NeuroKit2 ECG Quality — template-matching correlation score

Both computed per 10-second window with configurable stride.
"""

import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis, skew
from scipy.integrate import trapezoid


def qrs_band_energy_sqi(ecg_window, fs, qrs_low=5.0, qrs_high=20.0, total_high=40.0):
    """Compute QRS band energy ratio SQI for a single window.

    Parameters
    ----------
    ecg_window : np.ndarray
        ECG segment (e.g. 10 seconds).
    fs : float
        Sampling rate.
    qrs_low, qrs_high : float
        QRS frequency band (5-15 Hz default).
    total_high : float
        Upper bound for total power integration.

    Returns
    -------
    dict with:
        qrs_energy_ratio : float in [0, 1]
        snr_db           : float  QRS-band SNR in dB
        kurtosis_val     : float
        skewness_val     : float
    """
    if len(ecg_window) < int(fs):
        return {"qrs_energy_ratio": 0.0, "snr_db": -np.inf,
                "kurtosis_val": 0.0, "skewness_val": 0.0}

    nperseg = min(len(ecg_window), int(2 * fs))  # 2-second Welch segments
    f, Pxx = welch(ecg_window, fs, nperseg=nperseg)

    # Total power up to total_high Hz
    total_mask = (f >= 0.5) & (f <= total_high)
    total_power = trapezoid(Pxx[total_mask], f[total_mask])

    # QRS band power
    qrs_mask = (f >= qrs_low) & (f <= qrs_high)
    qrs_power = trapezoid(Pxx[qrs_mask], f[qrs_mask])

    # Noise power (outside QRS band but within total)
    noise_mask = total_mask & ~qrs_mask
    noise_power = trapezoid(Pxx[noise_mask], f[noise_mask])

    ratio = qrs_power / total_power if total_power > 0 else 0.0
    snr = 10 * np.log10(qrs_power / noise_power) if noise_power > 0 else 0.0

    return {
        "qrs_energy_ratio": float(ratio),
        "snr_db": float(snr),
        "kurtosis_val": float(kurtosis(ecg_window, fisher=True)),
        "skewness_val": float(skew(ecg_window)),
    }


def neurokit_ecg_sqi(ecg_window, fs):
    """Compute NeuroKit2 ECG quality score for a window.

    Uses nk.ecg_quality (averageQRS method) — template matching
    correlation. Falls back to a kurtosis-based heuristic if NK2
    is not available or fails.

    Returns
    -------
    float : quality score in [0, 1]
    """
    try:
        import neurokit2 as nk
        # Process to get R-peaks first
        signals, info = nk.ecg_process(ecg_window, sampling_rate=int(fs))
        quality = signals["ECG_Quality"].values
        return float(np.mean(quality[quality > 0])) if np.any(quality > 0) else 0.0
    except Exception:
        # Fallback: kurtosis-based heuristic
        k = kurtosis(ecg_window, fisher=True)
        if k > 8:
            return 0.1  # extreme artifacts
        elif k > 5:
            return 0.4
        elif k > 3:
            return 0.7
        else:
            return 0.9


def compute_window_sqi(ecg, fs, window_s=10.0, stride_s=5.0,
                       ts_ms=None, use_nk_sqi=True):
    """Compute SQI metrics for all 10-second windows over a signal.

    Parameters
    ----------
    ecg : np.ndarray
        Full ECG signal (preprocessed recommended).
    fs : float
        Sampling rate.
    window_s : float
        Window size in seconds (default 10).
    stride_s : float
        Stride in seconds (default 5 = 50% overlap).
    ts_ms : np.ndarray or None
        Unix timestamps per sample (int64 ms). If None, computed from index.
    use_nk_sqi : bool
        Whether to also compute NeuroKit2 SQI (slower).

    Returns
    -------
    list of dict, each containing:
        window_idx, start_sample, end_sample,
        start_ts_ms, end_ts_ms, center_ts_ms,
        qrs_energy_ratio, snr_db, kurtosis_val, skewness_val,
        nk_sqi (if use_nk_sqi), quality_label, is_usable
    """
    win_samples = int(window_s * fs)
    stride_samples = int(stride_s * fs)
    n = len(ecg)

    if ts_ms is None:
        ts_ms = (np.arange(n) * (1000.0 / fs)).astype(np.int64)

    results = []
    idx = 0
    for start in range(0, n - win_samples + 1, stride_samples):
        end = start + win_samples
        window = ecg[start:end]

        # QRS energy SQI
        qrs_metrics = qrs_band_energy_sqi(window, fs)

        # NK2 SQI (optional)
        nk_score = 0.0
        if use_nk_sqi:
            nk_score = neurokit_ecg_sqi(window, fs)

        # Composite quality assessment
        quality_label, is_usable = assess_quality(
            qrs_metrics["qrs_energy_ratio"],
            qrs_metrics["snr_db"],
            qrs_metrics["kurtosis_val"],
            nk_score,
        )

        results.append({
            "window_idx": idx,
            "start_sample": start,
            "end_sample": end,
            "start_ts_ms": int(ts_ms[start]),
            "end_ts_ms": int(ts_ms[min(end - 1, n - 1)]),
            "center_ts_ms": int(ts_ms[min(start + win_samples // 2, n - 1)]),
            "qrs_energy_ratio": qrs_metrics["qrs_energy_ratio"],
            "snr_db": qrs_metrics["snr_db"],
            "kurtosis_val": qrs_metrics["kurtosis_val"],
            "skewness_val": qrs_metrics["skewness_val"],
            "nk_sqi": nk_score,
            "quality_label": quality_label,
            "is_usable": is_usable,
        })
        idx += 1

    return results


def assess_quality(qrs_ratio, snr_db, kurtosis_val, nk_sqi=0.0):
    """Classify signal quality as good / mediocre / poor.

    Returns
    -------
    quality_label : str  ('good', 'mediocre', 'poor')
    is_usable     : bool (good or mediocre = True)
    """
    score = 0.0

    # QRS energy ratio (strongest indicator)
    if qrs_ratio >= 0.25:
        score += 0.40
    elif qrs_ratio >= 0.15:
        score += 0.25
    elif qrs_ratio > 0.10:
        score += 0.10

    # SNR (note: typically negative for bandpass-filtered ECG since
    # baseline/muscle power outside 5-15 Hz dominates; thresholds
    # reflect this — SNR > -3 dB is already a decent QRS signal)
    if snr_db > 0:
        score += 0.20
    elif snr_db > -5:
        score += 0.15
    elif snr_db > -10:
        score += 0.05

    # Kurtosis (clean ECG: ~3-8; artifacts: >10; flat: <2)
    if 2.5 < kurtosis_val < 8:
        score += 0.20
    elif 1.5 < kurtosis_val < 12:
        score += 0.10

    # NK2 SQI (bonus when available)
    if nk_sqi > 0.7:
        score += 0.20
    elif nk_sqi > 0.4:
        score += 0.10
    elif nk_sqi > 0.2:
        score += 0.05

    if score >= 0.65:
        return "good", True
    elif score >= 0.35:
        return "mediocre", True
    else:
        return "poor", False
