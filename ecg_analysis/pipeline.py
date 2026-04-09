"""
End-to-end ECG analysis pipeline.

Ties together: parsing -> preprocessing -> SQI -> sync -> detection -> evaluation -> export.
All timestamps in unix milliseconds.
"""

import numpy as np
import pandas as pd
import time as _time
from pathlib import Path
from typing import Optional, List

from .preprocessing import preprocess_ecg, PreprocessConfig
from .sqi import compute_window_sqi, assess_quality
from .detectors import detect_rpeaks, run_all_detectors, DetectorConfig
from .evaluation import (
    peaks_to_instantaneous_hr, median_smooth_hr,
    window_averaged_hr, evaluate_beats, evaluate_hr, compute_hr_metrics
)


def analyze_window(ecg_window, fs, ts_ms_start, config=None,
                   detector="neurokit", det_config=None, use_nk_sqi=True):
    """Analyze a single 10-second window.

    Parameters
    ----------
    ecg_window : np.ndarray
        Raw or preprocessed ECG segment.
    fs : float
    ts_ms_start : int
        Unix timestamp (ms) of window start.
    config : PreprocessConfig or None
    detector : str
        Detection method.
    det_config : DetectorConfig or None
    use_nk_sqi : bool

    Returns
    -------
    dict with SQI metrics, peaks, HR, timestamps
    """
    if config is None:
        config = PreprocessConfig()
    if det_config is None:
        det_config = DetectorConfig()

    # Preprocess
    ecg_clean, prep_info = preprocess_ecg(ecg_window, fs, config)

    # SQI
    from .sqi import qrs_band_energy_sqi, neurokit_ecg_sqi
    qrs_sqi = qrs_band_energy_sqi(ecg_clean, fs)
    nk_sqi = neurokit_ecg_sqi(ecg_clean, fs) if use_nk_sqi else 0.0
    quality_label, is_usable = assess_quality(
        qrs_sqi["qrs_energy_ratio"], qrs_sqi["snr_db"],
        qrs_sqi["kurtosis_val"], nk_sqi
    )

    # Detection (only if usable)
    peaks = np.array([], dtype=int)
    hr_bpm = 0.0
    det_time_ms = 0.0

    if is_usable:
        det_result = detect_rpeaks(ecg_clean, fs, method=detector, config=det_config)
        peaks = det_result["peaks"]
        det_time_ms = det_result["time_ms"]

        # Instantaneous HR
        hr_times, hr_vals = peaks_to_instantaneous_hr(peaks, fs)
        hr_bpm = float(np.median(hr_vals)) if len(hr_vals) > 0 else 0.0

    # Timestamps for peaks
    peak_ts_ms = (ts_ms_start + (peaks / fs * 1000)).astype(np.int64) if len(peaks) > 0 else np.array([], dtype=np.int64)

    return {
        "ts_ms_start": int(ts_ms_start),
        "ts_ms_end": int(ts_ms_start + len(ecg_window) / fs * 1000),
        "quality_label": quality_label,
        "is_usable": is_usable,
        "qrs_energy_ratio": qrs_sqi["qrs_energy_ratio"],
        "snr_db": qrs_sqi["snr_db"],
        "kurtosis": qrs_sqi["kurtosis_val"],
        "nk_sqi": nk_sqi,
        "peaks": peaks,
        "peak_ts_ms": peak_ts_ms,
        "hr_bpm_median": hr_bpm,
        "n_peaks": len(peaks),
        "detection_time_ms": det_time_ms,
        "preprocessing": prep_info,
        "detector": detector,
    }


def analyze_file(data, preprocess_config=None, detector="neurokit",
                 det_config=None, window_s=10.0, stride_s=5.0,
                 use_nk_sqi=True, analyze_full=True):
    """Analyze a full ECG file window-by-window plus optionally full-signal.

    Parameters
    ----------
    data : dict
        Output from parsers (must have 'ecg', 'fs', 'ts_ms').
    preprocess_config : PreprocessConfig or None
    detector : str
    det_config : DetectorConfig or None
    window_s, stride_s : float
    use_nk_sqi : bool
    analyze_full : bool
        Also run detection on the full preprocessed signal.

    Returns
    -------
    dict with:
        windows : list of window analysis dicts
        full_signal : dict (if analyze_full) with peaks, HR, etc.
        summary : dict with overall statistics
    """
    ecg = data["ecg"]
    fs = data["fs"]
    ts_ms = data["ts_ms"]
    n = len(ecg)

    if preprocess_config is None:
        preprocess_config = PreprocessConfig()
    if det_config is None:
        det_config = DetectorConfig()

    # Preprocess full signal once
    ecg_clean, prep_info = preprocess_ecg(ecg, fs, preprocess_config)

    # ── Window analysis ──
    win_samples = int(window_s * fs)
    stride_samples = int(stride_s * fs)
    windows = []

    print(f"Analyzing {n/fs:.1f}s signal ({n} samples @ {fs:.0f} Hz)")
    print(f"  Windows: {window_s}s, stride {stride_s}s, detector={detector}")

    for start in range(0, n - win_samples + 1, stride_samples):
        end = start + win_samples
        window_ecg = ecg_clean[start:end]
        ts_start = int(ts_ms[start])

        # SQI on this window
        from .sqi import qrs_band_energy_sqi, neurokit_ecg_sqi
        qrs_sqi = qrs_band_energy_sqi(window_ecg, fs)
        nk_score = neurokit_ecg_sqi(window_ecg, fs) if use_nk_sqi else 0.0
        quality_label, is_usable = assess_quality(
            qrs_sqi["qrs_energy_ratio"], qrs_sqi["snr_db"],
            qrs_sqi["kurtosis_val"], nk_score
        )

        # Detect peaks in window
        w_peaks = np.array([], dtype=int)
        w_hr = 0.0
        if is_usable:
            try:
                det_r = detect_rpeaks(window_ecg, fs, method=detector, config=det_config)
                w_peaks = det_r["peaks"]
                hr_t, hr_v = peaks_to_instantaneous_hr(w_peaks, fs)
                w_hr = float(np.median(hr_v)) if len(hr_v) > 0 else 0.0
            except Exception as e:
                print(f"  Window {start/fs:.1f}s: detection failed ({e})")

        # Map window peaks to global indices
        global_peaks = w_peaks + start
        peak_ts = (ts_ms[start] + (w_peaks / fs * 1000)).astype(np.int64) if len(w_peaks) > 0 else np.array([], dtype=np.int64)

        windows.append({
            "window_idx": len(windows),
            "start_sample": start,
            "end_sample": end,
            "ts_ms_start": int(ts_ms[start]),
            "ts_ms_end": int(ts_ms[min(end - 1, n - 1)]),
            "quality_label": quality_label,
            "is_usable": is_usable,
            "qrs_energy_ratio": qrs_sqi["qrs_energy_ratio"],
            "snr_db": qrs_sqi["snr_db"],
            "nk_sqi": nk_score,
            "n_peaks": len(w_peaks),
            "hr_bpm": w_hr,
            "global_peaks": global_peaks,
            "peak_ts_ms": peak_ts,
        })

    n_usable = sum(1 for w in windows if w["is_usable"])
    n_good = sum(1 for w in windows if w["quality_label"] == "good")
    print(f"  Windows: {len(windows)} total, {n_usable} usable, {n_good} good quality")

    # ── Full-signal analysis ──
    full_result = None
    if analyze_full:
        print(f"  Running {detector} on full signal...")
        try:
            full_det = detect_rpeaks(ecg_clean, fs, method=detector, config=det_config)
            full_peaks = full_det["peaks"]

            # HR metrics
            ihr_t, ihr_v = peaks_to_instantaneous_hr(full_peaks, fs)
            ihr_t_sm, ihr_v_sm = median_smooth_hr(ihr_t, ihr_v)

            # Window-averaged HR
            win_hr = window_averaged_hr(full_peaks, fs, window_s, stride_s,
                                        total_duration_s=n / fs)

            # Peak timestamps
            peak_ts_full = ts_ms[np.clip(full_peaks, 0, n - 1)]

            full_result = {
                "peaks": full_peaks,
                "peak_ts_ms": peak_ts_full,
                "detection_time_ms": full_det["time_ms"],
                "n_peaks": len(full_peaks),
                "instantaneous_hr_times": ihr_t,
                "instantaneous_hr_bpm": ihr_v,
                "smoothed_hr_times": ihr_t_sm,
                "smoothed_hr_bpm": ihr_v_sm,
                "window_hr": win_hr,
                "mean_hr_bpm": float(np.mean(ihr_v)) if len(ihr_v) > 0 else 0.0,
                "extra": full_det["extra"],
            }
            print(f"  Full signal: {len(full_peaks)} peaks, "
                  f"mean HR={full_result['mean_hr_bpm']:.1f} BPM, "
                  f"time={full_det['time_ms']:.1f}ms")
        except Exception as e:
            print(f"  Full signal detection failed: {e}")

    # ── Summary ──
    window_hrs = [w["hr_bpm"] for w in windows if w["hr_bpm"] > 0]
    summary = {
        "total_duration_s": n / fs,
        "n_windows": len(windows),
        "n_usable_windows": n_usable,
        "n_good_windows": n_good,
        "pct_usable": n_usable / max(len(windows), 1) * 100,
        "mean_window_hr_bpm": float(np.mean(window_hrs)) if window_hrs else 0.0,
        "std_window_hr_bpm": float(np.std(window_hrs)) if window_hrs else 0.0,
        "detector": detector,
        "preprocessing": prep_info,
    }

    return {
        "windows": windows,
        "full_signal": full_result,
        "summary": summary,
        "ecg_clean": ecg_clean,
        "fs": fs,
        "ts_ms": ts_ms,
    }


def compare_against_biopac(belt_analysis, biopac_analysis,
                            tolerance_ms=100, window_s=10.0, stride_s=5.0):
    """Compare belt detection results against BIOPAC reference.

    Parameters
    ----------
    belt_analysis : dict
        Output from analyze_file (belt).
    biopac_analysis : dict
        Output from analyze_file (BIOPAC).
    tolerance_ms : float
        Beat matching tolerance.

    Returns
    -------
    dict with beat-level and HR-based comparison metrics.
    """
    if belt_analysis["full_signal"] is None or biopac_analysis["full_signal"] is None:
        return {"error": "Full signal analysis required for comparison."}

    ref_peaks = biopac_analysis["full_signal"]["peaks"]
    test_peaks = belt_analysis["full_signal"]["peaks"]
    fs = biopac_analysis["fs"]

    # Comprehensive metrics
    metrics = compute_hr_metrics(
        ref_peaks, test_peaks, fs,
        tolerance_ms=tolerance_ms,
        window_s=window_s, stride_s=stride_s
    )

    return metrics


def export_hr_csv(analysis_result, output_path, label="belt"):
    """Export heart rate values with unix timestamps to CSV.

    Columns: timestamp_ms, hr_bpm_instantaneous, hr_bpm_smoothed,
             hr_bpm_window_avg, window_quality, detector

    Parameters
    ----------
    analysis_result : dict
        Output from analyze_file.
    output_path : str or Path
        Output CSV path.
    label : str
        Signal label for the CSV.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    # From full-signal instantaneous HR
    if analysis_result["full_signal"] is not None:
        full = analysis_result["full_signal"]
        fs = analysis_result["fs"]
        ts_ms = analysis_result["ts_ms"]

        # Peak-level rows
        for i, peak_idx in enumerate(full["peaks"]):
            peak_ts = int(ts_ms[min(peak_idx, len(ts_ms) - 1)])

            # Instantaneous HR: from RR interval
            ihr = np.nan
            if i < len(full["instantaneous_hr_bpm"]):
                ihr = full["instantaneous_hr_bpm"][i]
            elif i > 0 and i - 1 < len(full["instantaneous_hr_bpm"]):
                ihr = full["instantaneous_hr_bpm"][i - 1]

            # Smoothed HR
            shr = np.nan
            if i < len(full["smoothed_hr_bpm"]):
                shr = full["smoothed_hr_bpm"][i]
            elif i > 0 and i - 1 < len(full["smoothed_hr_bpm"]):
                shr = full["smoothed_hr_bpm"][i - 1]

            rows.append({
                "timestamp_ms": peak_ts,
                "peak_sample_idx": int(peak_idx),
                "hr_bpm_instantaneous": round(float(ihr), 2) if not np.isnan(ihr) else "",
                "hr_bpm_smoothed": round(float(shr), 2) if not np.isnan(shr) else "",
                "signal": label,
                "detector": analysis_result["summary"]["detector"],
            })

    # Window-level HR
    window_rows = []
    for w in analysis_result["windows"]:
        window_rows.append({
            "timestamp_ms": w["ts_ms_start"],
            "ts_ms_end": w["ts_ms_end"],
            "window_idx": w["window_idx"],
            "hr_bpm_window_avg": round(w["hr_bpm"], 2),
            "quality": w["quality_label"],
            "is_usable": w["is_usable"],
            "qrs_energy_ratio": round(w["qrs_energy_ratio"], 4),
            "snr_db": round(w["snr_db"], 2),
            "nk_sqi": round(w["nk_sqi"], 3) if w.get("nk_sqi") else "",
            "n_peaks": w["n_peaks"],
            "signal": label,
            "detector": analysis_result["summary"]["detector"],
        })

    # Save beat-level CSV
    if rows:
        df_beats = pd.DataFrame(rows)
        beat_path = str(output_path).replace(".csv", "_beats.csv")
        df_beats.to_csv(beat_path, index=False)
        print(f"  Exported {len(rows)} beat records -> {beat_path}")

    # Save window-level CSV
    if window_rows:
        df_windows = pd.DataFrame(window_rows)
        win_path = str(output_path).replace(".csv", "_windows.csv")
        df_windows.to_csv(win_path, index=False)
        print(f"  Exported {len(window_rows)} window records -> {win_path}")

    return str(output_path)


def export_comparison_csv(comparison_result, output_path):
    """Export BIOPAC vs Belt comparison metrics to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Window-level comparison
    if "window_details" in comparison_result:
        df = pd.DataFrame(comparison_result["window_details"])
        df.to_csv(output_path, index=False)
        print(f"  Exported comparison -> {output_path}")

    # Summary
    summary = comparison_result.get("hr_comparison", {})
    beat = comparison_result.get("beat_level", {})

    summary_rows = [{**beat, **summary}]
    df_summary = pd.DataFrame(summary_rows)
    summary_path = str(output_path).replace(".csv", "_summary.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"  Exported summary -> {summary_path}")

    return str(output_path)
