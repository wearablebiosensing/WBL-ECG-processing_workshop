"""
Data parsers for BIOPAC AcqKnowledge and Belt ECG formats.
All parsers return unix timestamps in milliseconds.
"""

import numpy as np
import pandas as pd
import time as _time
import re
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# MAX30003 ADC → mV conversion
# ---------------------------------------------------------------------------

def _max30003_convert(raw_int32):
    """Convert MAX30003 raw ADC integers to millivolts.

    Steps: cast int64 → mask 24-bit → ETAG extract → sign-extend →
    arithmetic right-shift 6 → scale (LSB ≈ 0.000381 mV/count) →
    interpolate invalid ETAG samples → remove DC.
    """
    raw = np.array(raw_int32, dtype=np.int64)
    raw_u24 = raw & 0xFFFFFF
    etag = (raw_u24 >> 3) & 0x07
    valid = (etag == 0) | (etag == 1) | (etag == 2)

    signed = raw_u24.copy()
    neg_mask = signed >= 0x800000
    signed[neg_mask] -= 0x1000000

    ecg_counts = signed >> 6  # 18-bit ECG

    VREF = 1.0   # V
    GAIN = 20.0   # V/V
    N_BITS = 18
    lsb_mV = (VREF * 1000.0) / (2**(N_BITS - 1) * GAIN)
    ecg_mV = ecg_counts * lsb_mV

    # Interpolate invalid samples
    if not np.all(valid):
        good_idx = np.where(valid)[0]
        bad_idx = np.where(~valid)[0]
        if len(good_idx) > 1:
            ecg_mV[bad_idx] = np.interp(bad_idx, good_idx, ecg_mV[good_idx])

    # Remove DC
    ecg_mV -= np.mean(ecg_mV)
    return ecg_mV.astype(np.float64)


# ---------------------------------------------------------------------------
# BIOPAC (Baby Belt pairing) — 27-line fixed header
# ---------------------------------------------------------------------------

def load_biopac(filepath, ecg_channel="CH40"):
    """Parse AcqKnowledge .txt export (Baby Belt format, 27-line header).

    Returns
    -------
    dict with keys:
        ecg        : np.ndarray   ECG signal
        time_s     : np.ndarray   time in seconds from start
        ts_ms      : np.ndarray   unix timestamp per sample (int64, ms)
        fs         : float        sampling rate Hz
        start_ms   : int          unix timestamp of recording start (ms)
        metadata   : dict         header fields
        df         : pd.DataFrame raw dataframe
    """
    filepath = str(filepath)
    meta = {}
    header_lines = 27

    with open(filepath, "r", encoding="utf-8-sig") as f:
        raw_header = [f.readline() for _ in range(header_lines + 5)]

    # Extract sampling rate
    for line in raw_header:
        if "msec/sample" in line.lower():
            meta["sample_interval_ms"] = float(line.split()[0])
            break
    if "sample_interval_ms" not in meta:
        raise ValueError("Could not find 'msec/sample' in header.")
    fs = 1000.0 / meta["sample_interval_ms"]

    # Extract recording date
    for line in raw_header:
        if "recording on:" in line.lower():
            tstr = line.split(":", 1)[1].strip()
            for fmt in ["%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S",
                        "%m/%d/%Y %H:%M:%S.%f", "%m/%d/%Y %H:%M:%S"]:
                try:
                    meta["start_datetime"] = datetime.strptime(tstr, fmt)
                    break
                except ValueError:
                    continue
            break

    start_dt = meta.get("start_datetime", datetime.now())
    start_ms = int(start_dt.timestamp() * 1000)

    # Find column header line
    data_start = header_lines
    col_line = ""
    for i, line in enumerate(raw_header):
        low = line.strip().lower()
        has_delim = "\t" in line or "," in line
        if not has_delim:
            continue
        if low.startswith("sec") or low.startswith("min") or low.startswith("millisec"):
            data_start = i
            col_line = line.strip()
            break

    # Skip metadata lines after header (e.g. "N samples")
    actual_start = data_start + 1
    for i in range(data_start + 1, min(len(raw_header), data_start + 10)):
        line = raw_header[i].strip()
        if not line or "samples" in line.lower():
            actual_start = i + 1
            continue
        if line[0].isdigit() or line[0] == "-" or line[0] == ".":
            actual_start = i
            break

    df = pd.read_csv(filepath, sep="\t", skiprows=actual_start,
                     header=None, on_bad_lines="skip", engine="python")

    # Parse column names from header
    col_names = [c.strip() for c in re.split(r"[\t,]", col_line)]
    n_data_cols = df.shape[1]
    n_header_cols = len(col_names)
    if n_header_cols >= n_data_cols:
        df.columns = col_names[:n_data_cols]
    else:
        df.columns = col_names + [f"extra_{i}" for i in range(n_data_cols - n_header_cols)]

    # Coerce numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(how="all", inplace=True)

    # Extract ECG
    ecg_col = ecg_channel.upper()
    matched_col = None
    for c in df.columns:
        if ecg_col in str(c).upper():
            matched_col = c
            break
    if matched_col is None:
        # Fallback: last numeric column
        num_cols = df.select_dtypes(include=[np.number]).columns
        matched_col = num_cols[-1] if len(num_cols) > 0 else df.columns[-1]

    ecg = df[matched_col].to_numpy(dtype=np.float64)
    n = len(ecg)
    time_s = np.arange(n) / fs
    ts_ms = (start_ms + (np.arange(n) * meta["sample_interval_ms"])).astype(np.int64)

    return {
        "ecg": ecg,
        "time_s": time_s,
        "ts_ms": ts_ms,
        "fs": fs,
        "start_ms": start_ms,
        "metadata": meta,
        "df": df,
    }


# ---------------------------------------------------------------------------
# Baby Belt CSV
# ---------------------------------------------------------------------------

def load_belt(filepath):
    """Parse Baby Belt CSV (BLE timing columns, ~100 Hz irregular).

    Returns dict with same keys as load_biopac.
    """
    filepath = str(filepath)
    df = pd.read_csv(filepath)

    # ECG column
    ecg_col = None
    for c in df.columns:
        if "ecg" in c.lower():
            ecg_col = c
            break
    if ecg_col is None:
        raise ValueError(f"No 'ECG' column found in {filepath}")

    ecg_raw = pd.to_numeric(df[ecg_col], errors="coerce").to_numpy(dtype=np.float64)
    valid = ~np.isnan(ecg_raw)
    ecg_raw = np.interp(np.arange(len(ecg_raw)),
                        np.where(valid)[0], ecg_raw[valid])

    # Estimate fs from InterArrival or Seq columns
    fs = 100.0
    if "InterArrival" in df.columns:
        ia = pd.to_numeric(df["InterArrival"], errors="coerce").dropna()
        if len(ia) > 10:
            median_ia = ia.median()
            if median_ia > 0:
                fs = 1000.0 / median_ia

    # Resample to uniform grid
    n_raw = len(ecg_raw)
    duration_s = n_raw / fs
    n_uniform = int(duration_s * fs)
    time_raw = np.arange(n_raw) / fs
    time_uniform = np.arange(n_uniform) / fs
    ecg = np.interp(time_uniform, time_raw, ecg_raw[:n_raw])

    # Unix timestamps (use current time as base — no absolute time in belt)
    start_ms = int(_time.time() * 1000)
    ts_ms = (start_ms + (time_uniform * 1000)).astype(np.int64)

    return {
        "ecg": ecg,
        "time_s": time_uniform,
        "ts_ms": ts_ms,
        "fs": fs,
        "start_ms": start_ms,
        "metadata": {"source": "baby_belt", "original_length": n_raw},
        "df": df,
    }


# ---------------------------------------------------------------------------
# CareWear BIOPAC — dynamic header
# ---------------------------------------------------------------------------

def load_carewear_biopac(filepath, ecg_col=None):
    """Parse CareWear BIOPAC .txt with dynamic header detection.

    Parameters
    ----------
    ecg_col : str or None
        Column name like 'CH9' (raw) or 'CH40' (AHA-filtered).
        If None, auto-detects CH9 first, then CH40.

    Returns dict with same keys as load_biopac.
    """
    filepath = str(filepath)
    meta = {}

    with open(filepath, "r", encoding="utf-8-sig") as f:
        raw_header = [f.readline() for _ in range(100)]

    # Sampling rate
    for line in raw_header:
        if "msec/sample" in line.lower():
            meta["sample_interval_ms"] = float(line.split()[0])
            break
    if "sample_interval_ms" not in meta:
        raise ValueError("Could not find 'msec/sample' in header.")
    fs = 1000.0 / meta["sample_interval_ms"]

    # Recording date
    for line in raw_header:
        if "recording on:" in line.lower():
            tstr = line.split(":", 1)[1].strip()
            for fmt in ["%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S",
                        "%m/%d/%Y %H:%M:%S.%f", "%m/%d/%Y %H:%M:%S"]:
                try:
                    meta["start_datetime"] = datetime.strptime(tstr, fmt)
                    break
                except ValueError:
                    continue
            break
    start_dt = meta.get("start_datetime", datetime.now())
    start_ms = int(start_dt.timestamp() * 1000)

    # Find column header — must have multiple tab/comma-separated fields
    data_start = None
    col_line = ""
    for i, line in enumerate(raw_header):
        low = line.strip().lower()
        has_delim = "\t" in line or "," in line
        # Column header must have delimiters (multiple fields)
        if not has_delim:
            continue
        if low.startswith("sec") or low.startswith("min") or low.startswith("millisec"):
            data_start = i
            col_line = line.strip()
            break
        if "ch" in low:
            data_start = i
            col_line = line.strip()
            break

    if data_start is None:
        raise ValueError("Could not locate column header in file.")

    # Skip post-header metadata
    actual_start = data_start + 1
    for i in range(data_start + 1, min(len(raw_header), data_start + 15)):
        line = raw_header[i].strip()
        if not line or "samples" in line.lower():
            actual_start = i + 1
            continue
        ch = line.lstrip(",.- ")
        if ch and (ch[0].isdigit() or ch[0] == "-"):
            actual_start = i
            break

    # Detect delimiter
    sep = "\t" if "\t" in col_line else ","
    col_names = [c.strip() for c in col_line.split(sep)]

    df = pd.read_csv(filepath, sep=sep, skiprows=actual_start,
                     header=None, on_bad_lines="skip", engine="python")

    # Rename columns from header — handle count mismatch
    n_data_cols = df.shape[1]
    n_header_cols = len(col_names)
    if n_header_cols >= n_data_cols:
        df.columns = col_names[:n_data_cols]
    else:
        df.columns = col_names + [f"extra_{i}" for i in range(n_data_cols - n_header_cols)]

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(how="all", inplace=True)

    # Auto-detect ECG column
    if ecg_col is None:
        for candidate in ["CH9", "CH40", "CH2"]:
            for c in df.columns:
                if candidate in str(c).upper():
                    ecg_col = c
                    break
            if ecg_col:
                break
    if ecg_col is None:
        num_cols = df.select_dtypes(include=[np.number]).columns
        ecg_col = num_cols[-1] if len(num_cols) > 0 else df.columns[-1]

    # Find matching column
    matched = None
    for c in df.columns:
        if str(ecg_col).upper() in str(c).upper():
            matched = c
            break
    if matched is None:
        matched = ecg_col

    ecg = df[matched].to_numpy(dtype=np.float64)
    ecg = np.nan_to_num(ecg, nan=0.0)
    n = len(ecg)
    time_s = np.arange(n) / fs
    ts_ms = (start_ms + (np.arange(n) * meta["sample_interval_ms"])).astype(np.int64)

    meta["ecg_column"] = str(matched)
    return {
        "ecg": ecg,
        "time_s": time_s,
        "ts_ms": ts_ms,
        "fs": fs,
        "start_ms": start_ms,
        "metadata": meta,
        "df": df,
    }


# ---------------------------------------------------------------------------
# CareWear Belt — extensionless CSV, Channel 4
# ---------------------------------------------------------------------------

def load_carewear_belt(filepath, ecg_mode="MAX30003", ecg_scale_fn=None):
    """Parse CareWear belt CSV (extensionless, Channel 4 ECG).

    Parameters
    ----------
    ecg_mode : str
        'MAX30003' — decode ADC to mV via _max30003_convert.
        'NORMALIZE' — float cast + zero-mean.
    ecg_scale_fn : callable or None
        Custom scaling function. If None, auto-set from ecg_mode.

    Returns dict with same keys as load_biopac.
    """
    filepath = str(filepath)
    df = pd.read_csv(filepath, header=None, on_bad_lines="skip", engine="python")

    # Detect and skip string header row (e.g. "timestamp, Channel 1, ...")
    first_val = str(df.iloc[0, 0]).strip().lower()
    if not first_val.replace(".", "").replace("-", "").replace("+", "").isdigit():
        df = df.iloc[1:].reset_index(drop=True)

    # CareWear belt format: col 0 = timestamp, cols 1-4 = channels
    # Channel 4 (last column) is ECG
    ecg_col_idx = df.shape[1] - 1  # default: last column

    # Coerce all columns to numeric
    for i in range(df.shape[1]):
        df[i] = pd.to_numeric(df[i], errors="coerce")

    raw = df[ecg_col_idx].to_numpy()
    valid_mask = ~np.isnan(raw)
    if not np.all(valid_mask):
        raw = np.interp(np.arange(len(raw)),
                        np.where(valid_mask)[0], raw[valid_mask])

    # Apply scaling
    if ecg_scale_fn is not None:
        ecg = ecg_scale_fn(raw)
    elif ecg_mode.upper() == "MAX30003":
        ecg = _max30003_convert(raw.astype(np.int64))
    else:
        ecg = raw.astype(np.float64)
        median = np.median(ecg)
        mad = np.median(np.abs(ecg - median))
        if mad > 1e-9:
            ecg = (ecg - median) / (1.4826 * mad)
        else:
            ecg = ecg - np.mean(ecg)

    # Estimate fs from timestamp column (col 0 has ms-resolution unix timestamps)
    fs = 125.0  # default CareWear
    belt_ts_raw = None
    if df.shape[1] > 1:
        ts_col = df[0].dropna()
        if len(ts_col) > 10:
            diffs = np.diff(ts_col.values)
            diffs = diffs[(diffs > 0) & (diffs < 100)]
            if len(diffs) > 5:
                median_diff_ms = np.median(diffs)
                if median_diff_ms > 0:
                    fs = 1000.0 / median_diff_ms
            belt_ts_raw = ts_col.values

    n = len(ecg)
    time_s = np.arange(n) / fs
    # Use actual belt timestamps if available
    if belt_ts_raw is not None and len(belt_ts_raw) >= n:
        start_ms = int(belt_ts_raw[0])
        ts_ms = belt_ts_raw[:n].astype(np.int64)
    else:
        start_ms = int(_time.time() * 1000)
        ts_ms = (start_ms + (time_s * 1000)).astype(np.int64)

    meta = {
        "source": "carewear_belt",
        "ecg_mode": ecg_mode,
        "estimated_fs": fs,
    }
    return {
        "ecg": ecg,
        "time_s": time_s,
        "ts_ms": ts_ms,
        "fs": fs,
        "start_ms": start_ms,
        "metadata": meta,
        "df": df,
    }
