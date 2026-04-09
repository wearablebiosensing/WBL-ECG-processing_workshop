import numpy as np
import pandas as pd
from scipy import interpolate, signal
from typing import Dict, List, Tuple

class FeatureExtractor:
    def __init__(self, fs: int = 250):
        self.fs = fs

    def _get_intersecting_tangent(self, signal_arr: np.ndarray, onset_idx: int, peak_idx: int) -> float:
        """
        Calculates the 'Intersecting Tangents' point for robust PAT.
        Intersection of:
        1. Tangent at maximum gradient (steepest upslope).
        2. Horizontal line at the onset value (baseline).
        """
        if peak_idx <= onset_idx: return float(onset_idx)
        
        # Extract segment
        segment = signal_arr[onset_idx : peak_idx + 1]
        
        # Calculate 1st derivative (slope)
        # precise gradient
        grad = np.gradient(segment)
        
        # Find max slope point
        # We expect a positive slope for the systolic upstroke
        max_slope_idx_local = np.argmax(grad)
        max_slope = grad[max_slope_idx_local]
        
        if max_slope <= 0:
            # Fallback: invalid upslope
            return float(onset_idx)
            
        # Point of max slope (relative to onset)
        t_max = max_slope_idx_local
        y_max = segment[t_max]
        
        # Onset Value (y_base)
        y_base = segment[0]
        
        # Equation of Tangent Line: y - y_max = m * (x - t_max)
        # We want to find x where y = y_base
        # y_base - y_max = m * (x - t_max)
        # (y_base - y_max) / m = x - t_max
        # x = t_max + (y_base - y_max) / m
        
        t_intersect_local = t_max + (y_base - y_max) / max_slope
        
        # Convert to absolute index
        t_intersect_abs = onset_idx + t_intersect_local
        
        return t_intersect_abs

    def _calculate_pw50(self, signal_arr: np.ndarray, onset_idx: int, peak_idx: int, next_onset_idx: int) -> float:
        """
        Calculates Pulse Width at 50% Amplitude (PW50).
        Returns width in samples.
        """
        try:
            y_base = signal_arr[onset_idx]
            y_peak = signal_arr[peak_idx]
            amp = y_peak - y_base
            
            if amp <= 0: return None
            
            thresh_y = y_base + 0.5 * amp
            
            # Find Rising Edge (Onset -> Peak)
            # Find first index where signal > thresh
            rise_seg = signal_arr[onset_idx:peak_idx+1]
            # np.where returns tuple
            rise_candidates = np.where(rise_seg >= thresh_y)[0]
            if len(rise_candidates) == 0: return None
            
            # Exact interpolation for sub-sample accuracy? 
            # For robustness, let's use linear interpolation between idx-1 and idx
            # rise_idx relative to onset
            r_idx_rel = rise_candidates[0]
            if r_idx_rel == 0: 
                 # already above? unusual but handle
                 t_rise = float(onset_idx)
            else:
                 # Interpolate between r_idx_rel-1 and r_idx_rel
                 y1 = rise_seg[r_idx_rel - 1]
                 y2 = rise_seg[r_idx_rel]
                 # y = y1 + slope*(t - t1) -> thresh = y1 + (y2-y1)*frac
                 # frac = (thresh - y1) / (y2 - y1)
                 if (y2 - y1) == 0: frac = 0
                 else: frac = (thresh_y - y1) / (y2 - y1)
                 t_rise = onset_idx + (r_idx_rel - 1) + frac
            
            # Find Falling Edge (Peak -> Next Onset)
            fall_seg = signal_arr[peak_idx:next_onset_idx+1]
            fall_candidates = np.where(fall_seg <= thresh_y)[0]
            if len(fall_candidates) == 0: return None
            
            f_idx_rel = fall_candidates[0]
            if f_idx_rel == 0:
                 t_fall = float(peak_idx)
            else:
                 y1_f = fall_seg[f_idx_rel - 1]
                 y2_f = fall_seg[f_idx_rel]
                 # crossing down
                 if (y2_f - y1_f) == 0: frac_f = 0
                 else: frac_f = (thresh_y - y1_f) / (y2_f - y1_f)
                 t_fall = peak_idx + (f_idx_rel - 1) + frac_f
            
            width = t_fall - t_rise
            return width if width > 0 else None
            
        except Exception:
            return None
    def extract_metrics(self, 
                        rpeaks: np.ndarray, 
                        ppg_peaks: np.ndarray, 
                        ppg_onsets: np.ndarray, 
                        ppg_signal: np.ndarray,
                        timestamps: np.ndarray,
                        use_morphology_filter: bool = False) -> pd.DataFrame:
        """
        Extracts robust PAT metrics suitable for Buprenorphine study.
        Phase 3: Cross-Verification (R-peak -> PPG Foot in 100-400ms window).
        Phase 4: Anatomical Constraints (100 < PAT < 400, MA +/- 20ms).
        """
        metrics = []
        
        # Sort
        rpeaks = np.sort(rpeaks)
        ppg_onsets = np.sort(ppg_onsets)
        
        # Parameters
        win_min_samples = int(0.100 * self.fs) # 100ms
        win_max_samples = int(0.400 * self.fs) # 400ms
        
        valid_pats = [] # For Moving Average calculation
        
        for r_idx in rpeaks:
            # Phase 3: Beat-Matching Logic
            # Window = [tR + 100ms, tR + 400ms]
            search_start = r_idx + win_min_samples
            search_end = r_idx + win_max_samples
            
            # Attempt to find PPG foot
            pat_ms = np.nan
            pat_p_ms = np.nan
            rise_time_ms = np.nan
            rise_amp = np.nan
            ppg_pw50_ms = np.nan
            onset_idx = -1
            
            candidates = ppg_onsets[(ppg_onsets >= search_start) & (ppg_onsets <= search_end)]
            
            if len(candidates) > 0:
                onset_idx = candidates[0]
                
                # --- Advanced PAT: Intersecting Tangents (PMC6308183) ---
                # We need the next peak to define the upstroke
                future_peaks = ppg_peaks[ppg_peaks > onset_idx]
                
                # Default "Foot" approach
                algo_pat_idx = float(onset_idx) 
                
                if len(future_peaks) > 0:
                    peak_idx = future_peaks[0]
                    
                    # Safety check: Peak must be reasonably close (e.g. within 300ms of onset)
                    if (peak_idx - onset_idx) < (0.3 * self.fs):
                        
                        apply_refinement = True
                        
                        # --- 7-Step Filter Logic (Optional) ---
                        if use_morphology_filter:
                            # S4: Amplitude > 0
                            amp = ppg_signal[peak_idx] - ppg_signal[onset_idx]
                            if amp <= 0:
                                apply_refinement = False
                        
                        if apply_refinement:
                             # Calculate refined foot
                            refined_idx = self._get_intersecting_tangent(ppg_signal, onset_idx, peak_idx)
                            algo_pat_idx = refined_idx
                            
                            # Metrics
                            rise_time_ms = ((peak_idx - refined_idx) / self.fs) * 1000.0
                            rise_amp = ppg_signal[peak_idx] - ppg_signal[onset_idx]
                            pat_p_ms = ((peak_idx - r_idx) / self.fs) * 1000.0

                            # --- PW50 Calculation (New) ---
                            # Need next onset to define the full pulse width
                            future_onsets = ppg_onsets[ppg_onsets > peak_idx]
                            if len(future_onsets) > 0:
                                next_onset_idx = future_onsets[0]
                                # Ensure next onset is not too far (e.g. < 2s)
                                if (next_onset_idx - onset_idx) < (2.0 * self.fs):
                                    pw50 = self._calculate_pw50(ppg_signal, onset_idx, peak_idx, next_onset_idx)
                                    if pw50 is not None:
                                        # pw50 is in samples, convert to ms
                                        ppg_pw50_ms = (pw50 / self.fs) * 1000.0

                # Calculate PAT using refined or raw index
                pat_samples = algo_pat_idx - r_idx
                raw_pat_ms = (pat_samples / self.fs) * 1000.0
                
                # Check constraints
                is_valid = True
                
                # 1. Absolute Range
                if not (100 <= raw_pat_ms <= 400): is_valid = False
                
                # 2. Moving Average Outlier
                if is_valid and len(valid_pats) >= 3:
                    mu = np.mean(valid_pats[-10:])
                    if abs(raw_pat_ms - mu) > 20.0: is_valid = False
                
                if is_valid:
                    pat_ms = raw_pat_ms
                    valid_pats.append(pat_ms)

            metrics.append({
                'timestamp_ms': timestamps[r_idx],
                'pat_f_ms': pat_ms,
                'pat_p_ms': pat_p_ms,
                'ppg_rise_time_ms': rise_time_ms,
                'ppg_rise_amp': rise_amp,
                'ppg_pw50_ms': ppg_pw50_ms,
                'rpeak_idx': r_idx,
                'ppg_onset_idx': onset_idx
            })
            
        df_metrics = pd.DataFrame(metrics)
        if df_metrics.empty: return df_metrics
        
        # --- Advanced Metrics (Vectorized) ---
        
        # 1. RR Interval (ms)
        df_metrics['rr_interval_ms'] = df_metrics['timestamp_ms'].diff()
        # Gap Detection (> 2000ms)
        df_metrics.loc[df_metrics['rr_interval_ms'] > 2000, 'rr_interval_ms'] = np.nan
        
        # --- RR Outlier Filtering (Pivot to Fix SDNN Spikes) ---
        # 1. Hard lower bound
        df_metrics.loc[df_metrics['rr_interval_ms'] < 300, 'rr_interval_ms'] = np.nan
        
        # 2. Rolling Median Filter
        # Window: 11 beats (5 before, 5 after)
        rr_median = df_metrics['rr_interval_ms'].rolling(window=11, center=True, min_periods=3).median()
        
        # Threshold: 35% deviation from local median (User requested 35%)
        threshold = rr_median * 0.35 
        
        is_outlier = np.abs(df_metrics['rr_interval_ms'] - rr_median) > threshold
        if is_outlier.any():
            # Set outliers to NaN so they are ignored by subsequent rolling std/mean
            df_metrics.loc[is_outlier, 'rr_interval_ms'] = np.nan
        # -------------------------------------------------------

        # --- PPG/PAT Outlier Filtering (35% Local Median) ---
        for col in ['pat_f_ms', 'pat_p_ms', 'ppg_rise_time_ms', 'ppg_pw50_ms']:
            if col in df_metrics.columns:
                # Calculate Median
                local_med = df_metrics[col].rolling(window=11, center=True, min_periods=3).median()
                
                # Calc Threshold
                thresh = local_med * 0.35
                
                # Detect
                # Note: PAT is typically ~250ms. 35% is ~87ms.
                is_out_col = np.abs(df_metrics[col] - local_med) > thresh
                
                if is_out_col.any():
                    df_metrics.loc[is_out_col, col] = np.nan
        # ----------------------------------------------------
        
        # 2. HRV SDNN (Standard Deviation of NN intervals)
        df_metrics['hrv_sdnn'] = df_metrics['rr_interval_ms'].rolling(window=50, min_periods=10, center=True).std()
        
        # 3. HRV RMSSD (Root Mean Square of Successive Differences)
        # diff(RR) -> square -> rolling mean -> sqrt
        rr_diff_sq = df_metrics['rr_interval_ms'].diff() ** 2
        df_metrics['hrv_rmssd'] = np.sqrt(rr_diff_sq.rolling(window=50, min_periods=10, center=True).mean())
        
        # 4. PAT Variability
        df_metrics['pat_f_sdnn'] = df_metrics['pat_f_ms'].rolling(window=50, min_periods=10, center=True).std()
        df_metrics['pat_p_sdnn'] = df_metrics['pat_p_ms'].rolling(window=50, min_periods=10, center=True).std()
        
        # 5. HRV Frequency Domain (Lomb-Scargle)
        # Window: 300s (5 min), Step: 30s
        df_metrics['hrv_lfhf'] = np.nan
        df_metrics['hrv_tp'] = np.nan
        
        # Check if we have enough data (at least 300 seconds worth, roughly)
        duration_sec = (timestamps[-1] - timestamps[0]) / 1000.0
        
        if duration_sec > 300:
             # Use Custom SciPy Implementation
             self._run_scipy_calc(df_metrics, timestamps, win_sec=300.0, step_sec=30.0)
        else:
             print(f"DEBUG: Not enough data for Freq Domain (Duration: {duration_sec:.1f}s)")

        
        return df_metrics

    def _run_scipy_calc(self, df_metrics, timestamps, win_sec=300.0, step_sec=30.0):
        """
        Primary: Custom SciPy Lomb-Scargle Implementation
        "Do it ourselves" approach to bypass library issues.
        """
        try:
            from scipy.signal import lombscargle
            
            t_sig = df_metrics['timestamp_ms'].values / 1000.0
            rr_sig = df_metrics['rr_interval_ms'].values
            
            t_start = t_sig[0]
            t_end = t_sig[-1]
            curr_t = t_start
            
            # print("DEBUG: Running HRV Frequency Analysis (Custom SciPy)...")
            
            windows_processed = 0
            windows_valid = 0
            
            # Pre-define Frequency Grid (0.001 to 0.4 Hz)
            freqs = np.linspace(0.001, 0.4, 1000)
            angular_freqs = 2 * np.pi * freqs
            
            while curr_t + win_sec <= t_end:
                windows_processed += 1
                mask = (t_sig >= curr_t) & (t_sig < curr_t + win_sec)
                rr_win = rr_sig[mask]
                
                # Filter NaNs
                valid_mask = ~np.isnan(rr_win)
                n_valid = np.sum(valid_mask)
                
                if n_valid > 100:
                    windows_valid += 1
                    nni = rr_win[valid_mask] # ms
                    
                    # Prepare Data for Lomb-Scargle
                    # Time in seconds, relative to window start for numerical stability
                    t_win = t_sig[mask][valid_mask]
                    t_win = t_win - t_win[0] 
                    y_ms = nni
                    
                    try:
                        # Compute Periodogram
                        # Power vs Angular Freq
                        # precenter=True removes the mean (detrending constant)
                        pgram = lombscargle(t_win, y_ms, angular_freqs, precenter=True)
                        
                        # Integrate Bands (Trapezoidal Rule)
                        # VLF: 0.0033-0.04, LF: 0.04-0.15, HF: 0.15-0.4
                        vlf_mask = (freqs >= 0.0033) & (freqs < 0.04)
                        lf_mask = (freqs >= 0.04) & (freqs < 0.15)
                        hf_mask = (freqs >= 0.15) & (freqs < 0.40)
                        
                        # Power Integration
                        # Note: lombscargle returns unnormalized power. 
                        # For Ratio, scaling cancels out. For Total Power, it's relative but consistent.
                        p_vlf = np.trapz(pgram[vlf_mask], freqs[vlf_mask])
                        p_lf = np.trapz(pgram[lf_mask], freqs[lf_mask])
                        p_hf = np.trapz(pgram[hf_mask], freqs[hf_mask])
                        
                        tp = p_vlf + p_lf + p_hf
                        ratio = p_lf / p_hf if p_hf > 0 else np.nan
                        
                        # Assign to DataFrame
                        mid_t = curr_t + win_sec/2
                        idx = np.searchsorted(t_sig, mid_t)
                        
                        if idx < len(df_metrics):
                            df_metrics.at[idx, 'hrv_lfhf'] = ratio
                            df_metrics.at[idx, 'hrv_tp'] = tp
                            
                    except Exception as e:
                        print(f"DEBUG: SciPy error at {curr_t:.1f}s: {e}")
                        pass
                
                curr_t += step_sec
                
            # print(f"DEBUG: SciPy Processed {windows_processed} windows. Valid: {windows_valid}.")
            
            # Interpolate
            df_metrics['hrv_lfhf'] = df_metrics['hrv_lfhf'].interpolate(method='linear', limit_direction='both')
            df_metrics['hrv_tp'] = df_metrics['hrv_tp'].interpolate(method='linear', limit_direction='both')
            
            # print("DEBUG: Final HRV Metrics Stats (SciPy):")
            # print(df_metrics[['hrv_lfhf', 'hrv_tp']].describe())
            
        except ImportError:
            print("Error: scipy not installed. Cannot calculate HRV.")
        except Exception as e:
            print(f"Error in SciPy calculation: {e}")

    def _run_nk2_calc(self, df_metrics, timestamps, win_sec=300.0, step_sec=30.0):
        """Secondary: neurokit2"""
        try:
            import neurokit2 as nk
            # ... existing nk2 logic ...
            t_sig = df_metrics['timestamp_ms'].values / 1000.0
            rr_sig = df_metrics['rr_interval_ms'].values
            
            t_start = t_sig[0]
            t_end = t_sig[-1]
            curr_t = t_start
            
            while curr_t + win_sec <= t_end:
                mask = (t_sig >= curr_t) & (t_sig < curr_t + win_sec)
                rr_win = rr_sig[mask]
                valid_mask = ~np.isnan(rr_win)
                n_valid = np.sum(valid_mask)
                
                if n_valid > 100:
                     peaks = {"RRI": rr_win[valid_mask], "RRI_Time": t_sig[mask][valid_mask]}
                     try:
                         res = nk.hrv_frequency(peaks, sampling_rate=4, psd_method='lomb', show=False)
                         mid_t = curr_t + win_sec/2
                         idx = np.searchsorted(t_sig, mid_t)
                         if idx < len(df_metrics):
                             if 'HRV_LFHF' in res: df_metrics.at[idx, 'hrv_lfhf'] = float(res['HRV_LFHF'].iloc[0])
                             tp = 0.0
                             if 'HRV_VLF' in res: tp += float(res['HRV_VLF'].iloc[0])
                             if 'HRV_LF' in res: tp += float(res['HRV_LF'].iloc[0])
                             if 'HRV_HF' in res: tp += float(res['HRV_HF'].iloc[0])
                             df_metrics.at[idx, 'hrv_tp'] = tp
                     except: pass
                curr_t += step_sec
            
            df_metrics['hrv_lfhf'] = df_metrics['hrv_lfhf'].interpolate(method='linear', limit_direction='both')
            df_metrics['hrv_tp'] = df_metrics['hrv_tp'].interpolate(method='linear', limit_direction='both')
        except:
            pass

