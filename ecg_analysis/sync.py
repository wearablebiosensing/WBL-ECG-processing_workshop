"""
Polarity-invariant signal synchronization.

Uses squared + Hilbert envelope transformations for cross-correlation
so that alignment works even when belt ECG has inverted polarity
relative to BIOPAC (2-electrode vs 3-electrode configuration).

Supports:
- Global lag correction (full-signal cross-correlation)
- Windowed lag refinement (adaptive per-segment correction)
"""

import numpy as np
from scipy.signal import correlate, hilbert, resample_poly
from math import gcd


def _envelope(signal):
    """Compute amplitude envelope via Hilbert transform."""
    analytic = hilbert(signal)
    return np.abs(analytic)


def _squared_envelope(signal):
    """Squared signal + smoothed envelope — polarity-invariant."""
    sq = signal ** 2
    env = _envelope(sq)
    return env


def polarity_invariant_xcorr(sig_a, sig_b, fs, max_lag_ms=2000,
                              segment_s=30.0, method="squared_envelope"):
    """Compute cross-correlation lag using polarity-invariant transforms.

    Both signals are transformed (squared + envelope) before correlation
    so that inverted polarity does not affect alignment.

    Parameters
    ----------
    sig_a, sig_b : np.ndarray
        Two ECG signals (should be roughly same sampling rate).
    fs : float
        Sampling rate (both signals assumed same rate after resampling).
    max_lag_ms : float
        Maximum allowed lag in milliseconds.
    segment_s : float
        Duration of signal segment to use for correlation (from start).
    method : str
        'squared_envelope' — square + Hilbert envelope (default, best)
        'squared' — just squaring (simpler, still polarity-invariant)
        'envelope' — Hilbert envelope only
        'raw' — raw cross-correlation (NOT polarity-invariant)

    Returns
    -------
    dict with:
        lag_samples : int  (positive = sig_b leads sig_a)
        lag_ms      : float
        xcorr_peak  : float (normalized peak correlation)
        method      : str
    """
    max_lag = int(max_lag_ms * fs / 1000)
    seg_len = min(int(segment_s * fs), len(sig_a), len(sig_b))

    seg_a = sig_a[:seg_len].astype(np.float64)
    seg_b = sig_b[:seg_len].astype(np.float64)

    # Remove DC
    seg_a -= np.mean(seg_a)
    seg_b -= np.mean(seg_b)

    # Apply polarity-invariant transform
    if method == "squared_envelope":
        seg_a = _squared_envelope(seg_a)
        seg_b = _squared_envelope(seg_b)
    elif method == "squared":
        seg_a = seg_a ** 2
        seg_b = seg_b ** 2
    elif method == "envelope":
        seg_a = _envelope(seg_a)
        seg_b = _envelope(seg_b)
    # else: raw — no transform

    # Remove DC again after transform
    seg_a -= np.mean(seg_a)
    seg_b -= np.mean(seg_b)

    # Cross-correlation
    corr = correlate(seg_a, seg_b, mode="full")
    mid = len(corr) // 2

    # Search within max_lag
    lo = max(0, mid - max_lag)
    hi = min(len(corr), mid + max_lag + 1)
    search = corr[lo:hi]

    peak_idx = np.argmax(search)
    lag_samples = peak_idx - (mid - lo)
    lag_ms = lag_samples / fs * 1000

    # Normalized correlation
    norm = np.sqrt(np.sum(seg_a ** 2) * np.sum(seg_b ** 2))
    xcorr_peak = search[peak_idx] / (norm + 1e-12)

    return {
        "lag_samples": int(lag_samples),
        "lag_ms": float(lag_ms),
        "xcorr_peak": float(xcorr_peak),
        "method": method,
    }


def _resample_to_common(sig, fs_orig, fs_target):
    """Resample signal to a common rate using polyphase."""
    if abs(fs_orig - fs_target) < 0.5:
        return sig, fs_orig
    g = gcd(int(round(fs_orig)), int(round(fs_target)))
    up = int(round(fs_target)) // g
    down = int(round(fs_orig)) // g
    resampled = resample_poly(sig, up, down)
    return resampled, fs_target


def sync_signals(biopac_data, belt_data, max_lag_ms=2000,
                 segment_s=30.0, method="squared_envelope"):
    """Synchronize belt ECG to BIOPAC ECG using polarity-invariant xcorr.

    Resamples belt to BIOPAC rate, finds lag, shifts belt time axis,
    and clips to overlapping region.

    Parameters
    ----------
    biopac_data : dict
        Output from parsers (must have 'ecg', 'fs', 'time_s', 'ts_ms').
    belt_data : dict
        Output from parsers (same keys).
    max_lag_ms : float
        Maximum lag to search.
    segment_s : float
        Segment duration for correlation.
    method : str
        Polarity-invariant transform method.

    Returns
    -------
    sync_info : dict with:
        lag_samples, lag_ms, xcorr_peak, method,
        belt_ecg_aligned    : np.ndarray (resampled to BIOPAC rate, shifted)
        belt_time_aligned   : np.ndarray
        belt_ts_ms_aligned  : np.ndarray (int64)
        biopac_ecg_clipped  : np.ndarray
        biopac_time_clipped : np.ndarray
        biopac_ts_ms_clipped: np.ndarray
        common_fs           : float
        overlap_duration_s  : float
    """
    bp_ecg = biopac_data["ecg"]
    bp_fs = biopac_data["fs"]
    bp_time = biopac_data["time_s"]
    bp_ts = biopac_data["ts_ms"]

    bl_ecg = belt_data["ecg"]
    bl_fs = belt_data["fs"]

    # Resample belt to BIOPAC rate
    bl_ecg_resampled, common_fs = _resample_to_common(bl_ecg, bl_fs, bp_fs)
    n_belt = len(bl_ecg_resampled)
    bl_time = np.arange(n_belt) / common_fs

    # Find lag
    xcorr_result = polarity_invariant_xcorr(
        bp_ecg, bl_ecg_resampled, common_fs,
        max_lag_ms=max_lag_ms, segment_s=segment_s, method=method
    )
    lag = xcorr_result["lag_samples"]

    # Apply lag: shift belt time axis
    # Positive lag means belt leads BIOPAC → shift belt backwards
    bl_time_shifted = bl_time - lag / common_fs

    # Find overlap region
    t_start = max(0, bl_time_shifted[0], bp_time[0])
    t_end = min(bl_time_shifted[-1], bp_time[-1])

    if t_end <= t_start:
        raise ValueError(
            f"No overlap after sync. Lag={lag} samples ({xcorr_result['lag_ms']:.1f} ms). "
            f"Belt range [{bl_time_shifted[0]:.2f}, {bl_time_shifted[-1]:.2f}], "
            f"BIOPAC range [{bp_time[0]:.2f}, {bp_time[-1]:.2f}]."
        )

    # Clip both to overlap
    bp_mask = (bp_time >= t_start) & (bp_time <= t_end)
    bl_mask = (bl_time_shifted >= t_start) & (bl_time_shifted <= t_end)

    bp_ecg_clip = bp_ecg[bp_mask]
    bp_time_clip = bp_time[bp_mask]

    bl_ecg_clip = bl_ecg_resampled[bl_mask]
    bl_time_clip = bl_time_shifted[bl_mask]

    # Interpolate belt onto BIOPAC time grid for sample-aligned comparison
    bl_ecg_aligned = np.interp(bp_time_clip, bl_time_clip, bl_ecg_clip)

    # Compute aligned timestamps
    bp_ts_clip = bp_ts[bp_mask]
    # Belt timestamps: use BIOPAC timestamp base (they're now aligned)
    bl_ts_aligned = bp_ts_clip.copy()

    result = {
        **xcorr_result,
        "belt_ecg_aligned": bl_ecg_aligned,
        "belt_time_aligned": bp_time_clip,
        "belt_ts_ms_aligned": bl_ts_aligned,
        "biopac_ecg_clipped": bp_ecg_clip,
        "biopac_time_clipped": bp_time_clip,
        "biopac_ts_ms_clipped": bp_ts_clip,
        "common_fs": common_fs,
        "overlap_duration_s": float(t_end - t_start),
    }
    return result


def windowed_sync_refinement(bp_ecg, bl_ecg, fs, window_s=30.0,
                              stride_s=15.0, max_lag_ms=500,
                              method="squared_envelope"):
    """Per-window lag refinement for signals that drift over time.

    Returns array of (window_center_s, local_lag_samples, local_lag_ms, xcorr)
    that can be used to apply adaptive time warping.
    """
    win = int(window_s * fs)
    stride = int(stride_s * fs)
    n = min(len(bp_ecg), len(bl_ecg))

    results = []
    for start in range(0, n - win + 1, stride):
        end = start + win
        seg_bp = bp_ecg[start:end]
        seg_bl = bl_ecg[start:end]
        center_s = (start + win / 2) / fs

        try:
            r = polarity_invariant_xcorr(
                seg_bp, seg_bl, fs,
                max_lag_ms=max_lag_ms,
                segment_s=window_s,
                method=method
            )
            results.append({
                "center_s": center_s,
                "lag_samples": r["lag_samples"],
                "lag_ms": r["lag_ms"],
                "xcorr_peak": r["xcorr_peak"],
            })
        except Exception:
            results.append({
                "center_s": center_s,
                "lag_samples": 0,
                "lag_ms": 0.0,
                "xcorr_peak": 0.0,
            })

    return results
