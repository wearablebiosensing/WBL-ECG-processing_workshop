import numpy as np
import scipy.signal as signal
import scipy.ndimage as ndimage
import logging

class Preprocessor:
    def __init__(self, target_fs: int = 250):
        self.target_fs = float(target_fs)
        self.logger = logging.getLogger(__name__)

    def _handle_invalid_segments(self, data: np.ndarray) -> np.ndarray:
        """Interpolate over NaN or -99 values."""
        signal_copy = data.copy()
        # Treat -99 or NaN as invalid
        # Note: If -99 is a valid signal value (unlikely for bio signals), this needs adjustment.
        invalid_mask = np.isnan(signal_copy) | np.isclose(signal_copy, -99.0)
        
        if np.any(invalid_mask):
            valid_indices = np.where(~invalid_mask)[0]
            invalid_indices = np.where(invalid_mask)[0]
            
            if len(valid_indices) > 0:
                signal_copy[invalid_indices] = np.interp(invalid_indices, valid_indices, signal_copy[valid_indices])
            else:
                return np.zeros_like(signal_copy) # All invalid
        return signal_copy

    def filter_ppg(self, signal_data: np.ndarray, fs: float, 
                   lowcut: float = 0.5, highcut: float = 8.0, 
                   filter_order: int = 4) -> np.ndarray:
        """
        Robust PPG filtering: Despike -> Detrend -> Bandpass -> Z-score.
        """
        # 1. Handle NaNs
        clean_sig = self._handle_invalid_segments(signal_data)

        # 2. Despiking (Median Filter)
        # Good for removing sudden artifacts from sensor movement/light
        ppg_despiked = ndimage.median_filter(clean_sig, size=5)

        # 3. Linear Detrending
        ppg_detrended = signal.detrend(ppg_despiked, type='linear')

        # 4. Bandpass Filtering (Butterworth SOS)
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        sos = signal.butter(filter_order, [low, high], btype='band', output='sos')
        ppg_filtered = signal.sosfiltfilt(sos, ppg_detrended)

        # 5. Z-score Normalization
        # Essential for consistent thresholding in detectors
        if np.std(ppg_filtered) > 1e-6:
             ppg_normalized = (ppg_filtered - np.mean(ppg_filtered)) / np.std(ppg_filtered)
        else:
             ppg_normalized = ppg_filtered

        return ppg_normalized

    def filter_ecg(self, signal_data: np.ndarray, fs: float,
                   lowcut: float = 0.5, highcut: float = 40.0,
                   powerline_freq: float = 50.0,
                   filter_order: int = 2) -> np.ndarray:
        """
        Robust ECG filtering: Baseline Removal -> Notch (Mains) -> Lowpass.
        """
        # 1. Handle NaNs
        clean_sig = self._handle_invalid_segments(signal_data)
        
        # 2. Baseline Wander Removal (High-pass)
        sos_hp = signal.butter(filter_order, lowcut, btype='highpass', fs=fs, output='sos')
        filtered = signal.sosfiltfilt(sos_hp, clean_sig)
        
        # 3. Powerline Removal (Notch Comb)
        # Remove mains and harmonics
        nyquist = fs / 2.0
        current_freq = powerline_freq
        while current_freq < nyquist:
            b_notch, a_notch = signal.iirnotch(w0=current_freq, Q=30.0, fs=fs)
            filtered = signal.filtfilt(b_notch, a_notch, filtered)
            current_freq += powerline_freq
            
        # 4. High Frequency Noise Suppression (Low-pass)
        sos_lp = signal.butter(filter_order, highcut, btype='lowpass', fs=fs, output='sos')
        filtered = signal.sosfiltfilt(sos_lp, filtered)
        
        return filtered

    def process_dataframe(self, df: 'pd.DataFrame', orig_fs: float, invert_ecg: bool = False) -> tuple:
        """
        Resamples first (if needed), then applies robust filtering.
        invert_ecg: If True, flips the polarity of the ECG signal.
        """
        import pandas as pd
        
        # 1. Resample if necessary
        # We prefer to resample first to reduce computational load for filtering, 
        # UNLESS the user wants to filter at high res.
        # Reference implementation suggests handling resampling carefully.
        # We will use the target_fs for output.
        
        if orig_fs != self.target_fs:
            # Calculate new length
            ratio = self.target_fs / orig_fs
            new_len = int(len(df) * ratio)
            
            # Resample Timestamp
            start_ts = df['timestamp_ms'].iloc[0]
            new_timestamps = start_ts + (np.arange(new_len) * (1000.0 / self.target_fs))
            
            resampled_data = {'timestamp_ms': new_timestamps}
            
            for col in ['ECG', 'PPG', 'SKT', 'EDA']:
                if col in df.columns:
                    # Enforce numeric type to avoid "ufunc isnan not supported"
                    # Coerce errors to NaN
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Polyphase resampling (includes anti-aliasing)
                    sig = df[col].to_numpy()
                    # Handle NaNs before resampling? resample_poly doesn't like NaNs.
                    sig = self._handle_invalid_segments(sig)
                    
                    
                    # Compute resampling
                    # Note: resample_poly is better than signal.resample (fft based) for edge effects usually
                    # But we need integer ratio up/down.
                    import math
                    gcd = math.gcd(int(orig_fs), int(self.target_fs))
                    up = int(self.target_fs // gcd)
                    down = int(orig_fs // gcd)
                    
                    resampled_sig = signal.resample_poly(sig, up, down)
                    
                    # Fix length
                    if len(resampled_sig) != new_len:
                         if len(resampled_sig) > new_len:
                             resampled_sig = resampled_sig[:new_len]
                         else:
                             resampled_sig = np.pad(resampled_sig, (0, new_len - len(resampled_sig)), 'edge')
                    
                    resampled_data[col] = resampled_sig
            
                    resampled_data[col] = resampled_sig
            
                    resampled_data[col] = resampled_sig
            
            # --- Preserve Label Column (Phase 30 Fix) ---
            if 'label' in df.columns:
                # User Request: "I only want it in the first instance matching closest timestamp and empty in others"
                # "merge_asof" with direction='nearest' fills EVERY row. We must stop this.
                # We need sparse assignment.
                
                # 1. Initialize empty
                label_col = np.full(len(new_timestamps), '', dtype=object)
                
                # 2. Extract labels
                df_lbl = df[['timestamp_ms', 'label']].dropna()
                
                if not df_lbl.empty:
                    # 3. For each label, find single nearest index in new_timestamps
                    t_new = new_timestamps  # float64
                    
                    for _, row in df_lbl.iterrows():
                        t_lbl = float(row['timestamp_ms'])
                        lbl_txt = str(row['label'])
                        
                        # Find nearest index
                        # np.searchsorted is fast O(log N)
                        idx = np.searchsorted(t_new, t_lbl)
                        
                        nearest_idx = -1
                        if idx == 0: nearest_idx = 0
                        elif idx == len(t_new): nearest_idx = len(t_new) - 1
                        else:
                            d1 = abs(t_new[idx-1] - t_lbl)
                            d2 = abs(t_new[idx] - t_lbl)
                            nearest_idx = idx-1 if d1 < d2 else idx
                            
                        # Assign
                        if 0 <= nearest_idx < len(label_col):
                            current = label_col[nearest_idx]
                            if current:
                                label_col[nearest_idx] = f"{current}; {lbl_txt}"
                            else:
                                label_col[nearest_idx] = lbl_txt
                                
                resampled_data['label'] = label_col
            
            processed_df = pd.DataFrame(resampled_data)
            current_fs = self.target_fs
        else:
            processed_df = df.copy()
            current_fs = orig_fs

        # 2. Filter
        if 'ECG' in processed_df.columns:
            processed_df['ECG'] = pd.to_numeric(processed_df['ECG'], errors='coerce')
            ecg_sig = processed_df['ECG'].to_numpy()
            
            if invert_ecg:
                ecg_sig = -ecg_sig
                
            processed_df['ECG'] = self.filter_ecg(ecg_sig, current_fs)
            
        if 'PPG' in processed_df.columns:
            # Use 'robust' parameters: 0.5-8Hz
            processed_df['PPG'] = pd.to_numeric(processed_df['PPG'], errors='coerce')
            processed_df['PPG'] = self.filter_ppg(processed_df['PPG'].to_numpy(), current_fs)
            
        return processed_df, current_fs
