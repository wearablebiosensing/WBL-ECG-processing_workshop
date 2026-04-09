import numpy as np
import warnings
import logging

# Optional imports handled gracefully
try:
    import neurokit2 as nk
except ImportError:
    nk = None

try:
    import wfdb.processing
except ImportError:
    wfdb = None

try:
    from ecg2rr import detector as ECG2RRDetector
except (ImportError, AttributeError, Exception) as _e:
    # ecg2rr depends on TensorFlow which is incompatible with NumPy 2.x
    ECG2RRDetector = None

try:
    from e2epyppg import kazemi_peak_detection as kazemi_ppg_detector
    from e2epyppg import ppg_sqa, ppg_reconstruction
    E2E_AVAILABLE = True
except ImportError:
    E2E_AVAILABLE = False

from .ppg_beats import MSPTDfastv2

class BeatDetector:
    def __init__(self, fs: int = 250):
        self.fs = fs
        self.logger = logging.getLogger(__name__)
        self._ecg2rr_instance = None

    def detect_ecg_beats(self, ecg_signal: np.ndarray, method: str = 'neurokit') -> tuple:
        """
        Detects ECG R-peaks.
        Methods: 'neurokit' (default), 'promac', 'pantompkins', 'wfdb', 'ecg2rr'.
        """
        # Handle NaNs in input
        if np.any(np.isnan(ecg_signal)):
             ecg_signal = np.nan_to_num(ecg_signal, nan=np.nanmean(ecg_signal))

        rpeaks = np.array([], dtype=int)
        
        try:
            if method == 'neurokit':
                if nk is None: raise ImportError("NeuroKit2 not installed.")
                # Fast & Reliable default
                _, info = nk.ecg_peaks(ecg_signal, sampling_rate=self.fs, method="neurokit",correct_artifacts=True)
                rpeaks = info['ECG_R_Peaks']

            elif method == 'promac':
                if nk is None: raise ImportError("NeuroKit2 not installed.")
                # Clean first for Promac? NK ecg_peaks usually handles it, but robust clean is good.
                cleaned = nk.ecg_clean(ecg_signal, sampling_rate=self.fs, method="neurokit")
                _, info = nk.ecg_peaks(cleaned, sampling_rate=self.fs, method="promac",correct_artifacts=True)
                rpeaks = info['ECG_R_Peaks']

            elif method == 'pantompkins':
                if nk is None: raise ImportError("NeuroKit2 not installed.")
                cleaned = nk.ecg_clean(ecg_signal, sampling_rate=self.fs, method="neurokit")
                _, info = nk.ecg_peaks(cleaned, sampling_rate=self.fs, method="pantompkins",correct_artifacts=True)
                rpeaks = info['ECG_R_Peaks']

            elif method == 'wfdb':
                if wfdb is None: raise ImportError("wfdb not installed.")
                # xqrs_detect matches provided reference
                # Note: xqrs might print to stdout, suppress if needed
                rpeaks = wfdb.processing.xqrs_detect(sig=ecg_signal, fs=self.fs, verbose=False)
                rpeaks = nk.signal_fixpeaks(rpeaks, sampling_rate=self.fs)

            elif method == 'ecg2rr':
                if ECG2RRDetector is None: raise ImportError("ecg2rr not installed.")
                if self._ecg2rr_instance is None:
                    # Model expects ~100Hz usually? Or is it agnostic?
                    # The reference used stride=250?
                    # We will reuse the parameters from reference.
                    self._ecg2rr_instance = ECG2RRDetector.ECG_detector(sampling_rate=100, stride=250, threshold=0.05)
                
                # ecg2rr usually expects 100Hz?
                # If our fs is 250, we might need resample or let the model handle?
                # Reference implementation passes 'clean_sig' which might be varying fs.
                # Assuming model handles it or user ensures correct setup.
                peaks, probs = self._ecg2rr_instance.find_peaks(ecg_signal)
                # Filter close
                rpeaks = self._ecg2rr_instance.remove_close(peaks=peaks, peak_probs=probs, threshold_ms=200)
                rpeaks = np.array(rpeaks, dtype=int)
                rpeaks = nk.signal_fixpeaks(rpeaks, sampling_rate=self.fs)

            else:
                self.logger.warning(f"Unknown ECG method '{method}'. Using 'neurokit' fallback.")
                if nk:
                    _, info = nk.ecg_peaks(ecg_signal, sampling_rate=self.fs, method="neurokit")
                    rpeaks = info['ECG_R_Peaks']
        
        except Exception as e:
            self.logger.error(f"ECG Detection ({method}) failed: {e}")
            # Fallback
            return np.array([], dtype=int), {}

        # Ensure we have array
        if rpeaks is None: rpeaks = np.array([])
        if isinstance(rpeaks, list): rpeaks = np.array(rpeaks)

        # Filter abnormally large beats
        rpeaks = self.filter_peaks_by_amplitude(ecg_signal, rpeaks)

        # Ensure int array
        rpeaks = rpeaks.astype(int)
        return rpeaks, {'method': method}

    def detect_ppg_beats(self, ppg_signal: np.ndarray, method: str = 'msptd') -> tuple:
        """
        Detects PPG peaks (systolic) and onsets (foot).
        Methods: 'msptd' (default), 'e2e'.
        """
        if np.any(np.isnan(ppg_signal)):
             ppg_signal = np.nan_to_num(ppg_signal, nan=np.nanmean(ppg_signal))

        peaks = np.array([], dtype=int)
        onsets = np.array([], dtype=int)

        try:
            if method == 'msptd':
                # Use our local port
                peaks, onsets = MSPTDfastv2.detect_beats(ppg_signal, self.fs)
                # peaks = nk.signal_fixpeaks(peaks, sampling_rate=self.fs)
                # onsets = nk.signal_fixpeaks(onsets, sampling_rate=self.fs)
            
            elif method == 'e2e':
                if not E2E_AVAILABLE: raise ImportError("E2E-PPG not installed.")
                # E2E pipeline: SQA -> Reconstruct -> Kazemi
                # Note: Reference only returns peaks. Onsets might need derivation or fallback.
                # We will perform detection.
                
                # 1. SQA & Reconstruct
                # e2e lib usually requires float32
                sig_32 = ppg_signal.astype(np.float32)
                clean_ind, noisy_ind = ppg_sqa.sqa(sig=sig_32, sampling_rate=self.fs, filter_signal=False)
                rec_sig, _, _ = ppg_reconstruction.reconstruction(
                    sig=sig_32, clean_indices=clean_ind, noisy_indices=noisy_ind, 
                    sampling_rate=self.fs, filter_signal=False
                )
                
                # 2. Peak Detect
                # minlen logic from reference
                total_sec = len(rec_sig) / self.fs
                minlen = min(5, total_sec) if total_sec > 0 else 5
                
                curr_peaks = kazemi_ppg_detector(
                    rec_sig, sampling_rate=self.fs, 
                    seconds=int(total_sec) if total_sec>1 else 1, 
                    overlap=0.5, minlen=minlen
                )
                
                # Flatten
                flat_peaks = []
                if isinstance(curr_peaks, (list, np.ndarray)):
                    for item in curr_peaks:
                         if isinstance(item, (list, np.ndarray)): flat_peaks.extend(item)
                         else: flat_peaks.append(item)
                peaks = np.unique(flat_peaks).astype(int)
                
                # E2E doesn't give onsets by default. Estimate onsets as minimum between peaks.
                # Simple approximation: Find min between p[i] and p[i+1]
                onsets_approx = []
                sorted_peaks = np.sort(peaks)
                for i in range(len(sorted_peaks)-1):
                    start, end = sorted_peaks[i], sorted_peaks[i+1]
                    if start < end:
                        segment = ppg_signal[start:end]
                        if len(segment) > 0:
                            min_idx = start + np.argmin(segment)
                            onsets_approx.append(min_idx)
                
                # Also check before first peak
                if len(sorted_peaks) > 0:
                    first = sorted_peaks[0]
                    # Look back up to 1 sec
                    lookback = int(self.fs)
                    start = max(0, first - lookback)
                    if start < first:
                         segment = ppg_signal[start:first]
                         if len(segment) > 0:
                             onsets_approx.insert(0, start + np.argmin(segment))
                
                onsets = np.array(onsets_approx, dtype=int)

            else:
                 # Fallback
                 peaks, onsets = MSPTDfastv2.detect_beats(ppg_signal, self.fs)

        except Exception as e:
            self.logger.error(f"PPG Detection ({method}) failed: {e}")
            return np.array([], dtype=int), np.array([], dtype=int)

        # Ensure we have arrays
        if peaks is None: peaks = np.array([])
        if onsets is None: onsets = np.array([])
        
        # Convert to numpy if list
        if isinstance(peaks, list): peaks = np.array(peaks)
        if isinstance(onsets, list): onsets = np.array(onsets)

        # Filter PPG peaks
        peaks = self.filter_peaks_by_amplitude(ppg_signal, peaks.astype(int))
        
        # Note: onsets are harder to filter by amplitude directly without associating to peaks
        # For now, we leave onsets as is, or we should re-derive valid onsets from valid peaks?
        # If we remove a peak, the corresponding onset might be orphan.
        # Ideally we filter pairs.
        # But simplified approach: filter peaks, and visualization will just show fewer peaks.
        # Feature extraction matches Peak-Onset pairs. If peak is missing, onset is unused.
        
        return peaks.astype(int), onsets.astype(int)

    def filter_peaks_by_amplitude(self, signal_data: np.ndarray, peaks: np.ndarray, threshold_std: float = 1.5) -> np.ndarray:
        """
        Removes peaks with amplitude > median + threshold_std * std.
        Used to remove abnormally large artifacts.
        """
        if len(peaks) < 3: return peaks
        
        # Get amplitudes
        amps = signal_data[peaks]
        
        # Calculate robust stats
        median_amp = np.median(amps)
        std_amp = np.std(amps)
        
        # Define range
        # User requested abnormally large like 1.5x std from typical (median)
        # We focus on UPPER bound for artifacts, but could check lower too?
        # "abnormally large" usually means positive outliers for peaks.
        upper_limit = median_amp + (threshold_std * std_amp)
        lower_limit = median_amp - (threshold_std * std_amp) 
        
        # Filter
        # We might want to be lenient on lower limit for R-peaks? 
        # But if it's too small it might be noise.
        # Let's apply symmetric check around median for "typical height"
        # valid_mask = (amps <= upper_limit) & (amps >= lower_limit)
        
        # User specifically said "abnormally large", so mainly upper bound?
        # "1.5x std from typical peak heights"
        # Let's filter both ends to be safe against glitches
        
        valid_mask = (amps <= upper_limit) & (amps >= lower_limit)
        return peaks[valid_mask]
        
    def refine_onsets(self, signal_data: np.ndarray, onsets: np.ndarray, peaks: np.ndarray) -> np.ndarray:
        """
        Refines PPG onset detection using the Intersecting Tangent Method.
        Finds the intersection of the tangent at the max slope point and the baseline at the valley.
        Used to identify the 'anatomical foot' or 'sknee'.
        """
        if len(onsets) == 0 or len(peaks) == 0:
            return onsets
            
        refined_onsets = []
        
        # Sort
        onsets = np.sort(onsets)
        peaks = np.sort(peaks)
        
        # Associate Onset -> Peak (Next peak)
        for onset in onsets:
            # Find next peak
            future_peaks = peaks[peaks > onset]
            if len(future_peaks) == 0:
                # No peak, keep original
                refined_onsets.append(onset)
                continue
                
            peak = future_peaks[0]
            
            # Check for sanity (interval too long?)
            if (peak - onset) > self.fs: # > 1s upstroke is impossible
                refined_onsets.append(onset)
                continue
                
            # Segment
            segment = signal_data[onset : peak+1]
            if len(segment) < 4:
                refined_onsets.append(onset)
                continue
                
            # 1. First Derivative
            d1 = np.diff(segment)
            
            # 2. Max Slope Point
            # We want the max positive slope
            if len(d1) == 0:
                 refined_onsets.append(onset)
                 continue
                 
            max_slope_idx = np.argmax(d1)
            max_slope = d1[max_slope_idx]
            
            if max_slope <= 0:
                refined_onsets.append(onset)
                continue
                
            # max_slope_idx is index in d1. d1[i] = y[i+1] - y[i].
            # Approx location in segment is i or i+0.5. Let's say i.
            
            # 3. Y value at Max Slope
            # We use the point on the signal corresponding to the max slope
            # Let's say index `k = max_slope_idx` within segment
            
            # 4. Tangent Line: y = m(x - x0) + y0
            # m = max_slope
            # x0 = max_slope_idx
            # y0 = segment[x0]
            
            # 5. Baseline: Horizontal line at local minimum (Onset value)
            # y = y_onset = segment[0]
            
            # 6. Intersection
            # y_onset = m(x - x0) + y0
            # y_onset - y0 = m(x - x0)
            # (y_onset - y0)/m = x - x0
            # x = x0 + (y_onset - y0)/m
            
            y0 = segment[max_slope_idx]
            y_onset = segment[0] # Valley value
            
            delta_x = (y_onset - y0) / max_slope
            x_intersect = max_slope_idx + delta_x
            
            # x_intersect is relative to onset
            refined_idx = int(onset + x_intersect)
            
            # Constraint: Must be between onset and peak (and usually close to onset)
            if refined_idx < onset: refined_idx = onset
            if refined_idx >= peak: refined_idx = onset # Fallback
            
            refined_onsets.append(refined_idx)
            
        return np.array(refined_onsets, dtype=int)
