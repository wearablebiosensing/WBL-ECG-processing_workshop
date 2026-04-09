"""
R-peak detection module.

Detectors: neurokit (default), xqrs, promac, rpnet (CUDA-batched).
All return peak indices into the original (preprocessed) signal.
"""

import numpy as np
import time as _time
from dataclasses import dataclass, field
from typing import List, Optional
from scipy.signal import find_peaks, resample_poly
from math import gcd


@dataclass
class DetectorConfig:
    """Configuration for R-peak detection."""
    methods: List[str] = field(default_factory=lambda: ["neurokit"])
    # RPNet settings
    rpnet_model_dir: str = "reference_codes/RPNet"
    rpnet_weights: str = "model.pt"
    rpnet_window_s: float = 10.0    # 10-second windows
    rpnet_stride_s: float = 5.0     # 50% overlap
    rpnet_target_fs: int = 500      # RPNet expects 500 Hz
    rpnet_batch_size: int = 16      # CUDA batch size
    # PROMAC
    promac_methods: List[str] = field(default_factory=lambda: [
        "neurokit", "pantompkins1985", "hamilton2002",
        "christov2004", "engzeemod2012"
    ])


def detect_rpeaks(ecg, fs, method="neurokit", config=None):
    """Detect R-peaks using the specified method.

    Parameters
    ----------
    ecg : np.ndarray
        Preprocessed ECG signal.
    fs : float
        Sampling rate.
    method : str
        One of: 'neurokit', 'xqrs', 'promac', 'rpnet'.
    config : DetectorConfig or None

    Returns
    -------
    dict with:
        peaks       : np.ndarray of int (sample indices)
        method      : str
        time_ms     : float (detection runtime in ms)
        extra       : dict (method-specific metadata)
    """
    if config is None:
        config = DetectorConfig()

    t0 = _time.perf_counter()

    if method == "neurokit":
        peaks, extra = _detect_neurokit(ecg, fs)
    elif method == "xqrs":
        peaks, extra = _detect_xqrs(ecg, fs)
    elif method == "promac":
        peaks, extra = _detect_promac(ecg, fs, config.promac_methods)
    elif method == "rpnet":
        peaks, extra = _detect_rpnet(ecg, fs, config)
    else:
        raise ValueError(f"Unknown detector: {method}. "
                         f"Choose from: neurokit, xqrs, promac, rpnet")

    elapsed = (_time.perf_counter() - t0) * 1000

    return {
        "peaks": np.array(peaks, dtype=int),
        "method": method,
        "time_ms": elapsed,
        "extra": extra,
    }


# ---------------------------------------------------------------------------
# NeuroKit2 default
# ---------------------------------------------------------------------------

def _detect_neurokit(ecg, fs):
    import neurokit2 as nk
    _, info = nk.ecg_peaks(ecg, sampling_rate=int(fs),
                           method="neurokit", correct_artifacts=True)
    peaks = np.array(info["ECG_R_Peaks"], dtype=int)
    return peaks, {"nk_method": "neurokit"}


# ---------------------------------------------------------------------------
# XQRS (WFDB)
# ---------------------------------------------------------------------------

def _detect_xqrs(ecg, fs):
    import wfdb.processing
    xqrs = wfdb.processing.XQRS(sig=ecg, fs=int(fs))
    xqrs.detect(verbose=False)
    peaks = np.array(xqrs.qrs_inds, dtype=int)
    return peaks, {"wfdb_method": "xqrs"}


# ---------------------------------------------------------------------------
# PROMAC (multi-detector consensus via NeuroKit2)
# ---------------------------------------------------------------------------

def _detect_promac(ecg, fs, methods=None):
    import neurokit2 as nk
    if methods is None:
        methods = ["neurokit", "pantompkins1985", "hamilton2002",
                   "christov2004", "engzeemod2012"]

    _, info = nk.ecg_peaks(ecg, sampling_rate=int(fs),
                           method="promac", correct_artifacts=True)
    peaks = np.array(info["ECG_R_Peaks"], dtype=int)
    return peaks, {"promac_methods": methods}


# ---------------------------------------------------------------------------
# RPNet — CUDA-batched IncRes-UNet distance transform
# ---------------------------------------------------------------------------

def _load_rpnet_model(config):
    """Load RPNet IncUNet model, auto-detect CUDA."""
    import torch
    import sys
    import os

    # Add RPNet directory to path for imports
    rpnet_dir = os.path.abspath(config.rpnet_model_dir)
    if rpnet_dir not in sys.path:
        sys.path.insert(0, rpnet_dir)

    from network import IncUNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = IncUNet(in_shape=(1, 1, 5000))
    weights_path = os.path.join(rpnet_dir, config.rpnet_weights)
    state_dict = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, device


def _rpnet_batch_inference(signal_500hz, model, device, window=5000,
                           stride=2500, batch_size=16):
    """Run RPNet inference with CUDA batching.

    Processes 10-sec windows (5000 samples @ 500 Hz) in batches on GPU.
    Overlapping windows are averaged for the distance transform.

    Returns
    -------
    distance_map : np.ndarray (same length as signal_500hz)
    """
    import torch

    N = len(signal_500hz)
    distance_map = np.zeros(N, dtype=np.float64)
    weight_map = np.zeros(N, dtype=np.float64)

    # Build window start indices
    starts = list(range(0, max(1, N - window + 1), stride))
    if len(starts) > 0 and starts[-1] + window < N:
        starts.append(N - window)
    # Handle signals shorter than one window
    if N < window:
        starts = [0]

    # Batch processing
    total_windows = len(starts)
    for batch_start in range(0, total_windows, batch_size):
        batch_end = min(batch_start + batch_size, total_windows)
        batch_indices = starts[batch_start:batch_end]

        # Build batch tensor
        batch_windows = []
        for s in batch_indices:
            end = min(s + window, N)
            seg = signal_500hz[s:end].copy()
            # Pad if short
            if len(seg) < window:
                seg = np.pad(seg, (0, window - len(seg)), mode="edge")
            # Per-window z-score normalization
            std = np.std(seg)
            if std > 1e-6:
                seg = (seg - np.mean(seg)) / std
            batch_windows.append(seg)

        batch_np = np.array(batch_windows, dtype=np.float32)
        # Shape: [batch, 1, 5000]
        batch_tensor = torch.from_numpy(batch_np[:, np.newaxis, :]).to(device)

        with torch.no_grad():
            output = model(batch_tensor)  # [batch, 1, 5000]

        output_np = output.cpu().numpy()[:, 0, :]  # [batch, 5000]

        # Accumulate with overlap averaging
        for i, s in enumerate(batch_indices):
            end = min(s + window, N)
            seg_len = end - s
            dt_seg = output_np[i, :seg_len]
            distance_map[s:end] += dt_seg
            weight_map[s:end] += 1.0

    # Average overlapping regions
    valid = weight_map > 0
    distance_map[valid] /= weight_map[valid]

    return distance_map


def _peaks_from_distance_transform(distance_map, fs=500):
    """Extract R-peak indices from RPNet distance transform.

    R-peaks correspond to DT minima (valleys). We negate the DT
    and find peaks with adaptive prominence and refractory period.
    """
    inv_dt = -distance_map
    dt_range = np.ptp(distance_map)
    prominence = max(0.1, 0.15 * dt_range)
    min_distance = int(0.3 * fs)  # 300 ms refractory

    peaks, properties = find_peaks(inv_dt, prominence=prominence,
                                   distance=min_distance)
    return peaks


def _detect_rpnet(ecg, fs, config):
    """Full RPNet pipeline: resample → batch inference → peak extraction.

    Returns peaks in original signal coordinates.
    """
    import torch

    target_fs = config.rpnet_target_fs  # 500 Hz

    # Resample to 500 Hz
    fs_int = int(round(fs))
    g = gcd(fs_int, target_fs)
    up = target_fs // g
    down = fs_int // g
    ecg_500 = resample_poly(ecg, up, down)

    # Load model
    model, device = _load_rpnet_model(config)
    device_name = str(device)
    using_cuda = torch.cuda.is_available()

    # Batch inference
    window = int(config.rpnet_window_s * target_fs)  # 5000
    stride_samples = int(config.rpnet_stride_s * target_fs)  # 2500

    distance_map = _rpnet_batch_inference(
        ecg_500, model, device,
        window=window, stride=stride_samples,
        batch_size=config.rpnet_batch_size
    )

    # Extract peaks at 500 Hz
    peaks_500 = _peaks_from_distance_transform(distance_map, target_fs)

    # Map peaks back to original sampling rate
    peaks_orig = np.round(peaks_500 * fs / target_fs).astype(int)
    # Clamp to valid range
    peaks_orig = peaks_orig[(peaks_orig >= 0) & (peaks_orig < len(ecg))]

    extra = {
        "device": device_name,
        "using_cuda": using_cuda,
        "distance_map_500hz": distance_map,
        "peaks_500hz": peaks_500,
        "n_windows": len(range(0, len(ecg_500), stride_samples)),
    }
    return peaks_orig, extra


# ---------------------------------------------------------------------------
# Multi-detector runner
# ---------------------------------------------------------------------------

def run_all_detectors(ecg, fs, methods=None, config=None):
    """Run multiple detectors and return results dict.

    Parameters
    ----------
    ecg : np.ndarray
    fs : float
    methods : list of str or None
        If None, runs ['neurokit', 'xqrs', 'promac'].
    config : DetectorConfig or None

    Returns
    -------
    dict : {method_name: detect_rpeaks result dict}
    """
    if methods is None:
        methods = ["neurokit", "xqrs", "promac"]
    if config is None:
        config = DetectorConfig()

    results = {}
    for m in methods:
        try:
            results[m] = detect_rpeaks(ecg, fs, method=m, config=config)
        except Exception as e:
            print(f"  [{m}] failed: {e}")
            results[m] = {
                "peaks": np.array([], dtype=int),
                "method": m,
                "time_ms": 0.0,
                "extra": {"error": str(e)},
            }
    return results
