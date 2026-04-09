import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis, entropy
import logging
from ..utils.config_manager import ConfigManager

class SQIAnalyzer:
    def __init__(self, fs: int = 250):
        self.fs = fs
        self.logger = logging.getLogger(__name__)
        
        # Load Config
        self.config_manager = ConfigManager()
        self.config_manager.load_config()
        self.config = self.config_manager.get('sqi')
        
        # Flatten for internal use if strict mapping needed, 
        # but better to access config directly.

    def _safe_metric(self, func, *args):
        try:
            val = func(*args)
            return val if not np.isnan(val) and not np.isinf(val) else 0.0
        except:
            return 0.0

    def compute_ecg_sqi(self, signal: np.ndarray) -> dict:
        if len(signal) < self.fs: return {'ecg_sqi_score': 0.0}
        
        s = skew(signal)
        k = kurtosis(signal)
        
        f, Pxx = welch(signal, self.fs, nperseg=min(len(signal), 1024))
        total_p = np.trapz(Pxx, f)
        
        qrs_mask = (f >= 5) & (f <= 15)
        qrs_p = np.trapz(Pxx[qrs_mask], f[qrs_mask])
        rel_power = qrs_p / total_p if total_p > 0 else 0
        
        noise_mask = ((f < 5) | (f > 15)) & (f <= 40)
        noise_p = np.trapz(Pxx[noise_mask], f[noise_mask])
        snr_val = 10 * np.log10(qrs_p / noise_p) if noise_p > 0 else 0
        
        sig_cent = signal - np.mean(signal)
        zcr_ratio = np.sum(np.abs(np.diff(np.sign(sig_cent)))) / (2 * len(signal))
        
        # Rule-based Score
        score = 0.0
        # Use config rules? For now keeping heuristic score but applying hard thresholds later
        if rel_power > 0.3: score += 0.4
        if snr_val > 0: score += 0.2
        if k > 3: score += 0.2
        if 0.0 < zcr_ratio < 0.2: score += 0.2
        
        metrics = {
            'ecg_skew': s,
            'ecg_kurtosis': k,
            'ecg_rel_power_qrs': rel_power,
            'ecg_snr': snr_val,
            'ecg_zcr': zcr_ratio,
            'ecg_total_power': total_p, # For clustering
            'ecg_sqi_score': min(score, 1.0)
        }
        return metrics

    def compute_ppg_sqi(self, signal: np.ndarray) -> dict:
        if len(signal) < self.fs: return {'ppg_sqi_score': 0.0}

        ac_amp = np.ptp(signal)
        s = skew(signal)
        k = kurtosis(signal)
        
        hist, _ = np.histogram(signal, bins='auto', density=True)
        ent = entropy(hist)
        
        f, Pxx = welch(signal, self.fs, nperseg=min(len(signal), 1024))
        total_p = np.trapz(Pxx, f)
        
        band_mask = (f >= 0.8) & (f <= 3.0)
        band_p = np.trapz(Pxx[band_mask], f[band_mask])
        rel_power = band_p / total_p if total_p > 0 else 0
        
        sig_cent = signal - np.mean(signal)
        zcr_ratio = np.sum(np.abs(np.diff(np.sign(sig_cent)))) / (2 * len(signal))

        score = 0.0
        if rel_power > 0.5: score += 0.4
        if abs(s) > 0.5: score += 0.2
        if ent < 4.0: score += 0.2
        if k > 0: score += 0.2
        
        metrics = {
            'ppg_skew': s,
            'ppg_kurtosis': k,
            'ppg_entropy': ent,
            'ppg_rel_power': rel_power,
            'ppg_zcr': zcr_ratio,
            'ppg_ac_amp': ac_amp,
            'ppg_total_power': total_p, # For clustering
            'ppg_sqi_score': min(score, 1.0)
        }
        return metrics

    def compute_template_sqi(self, signal: np.ndarray, signal_type: str = 'ecg') -> float:
        """
        Computes SQI based on correlation with the median Beat Template.
        Requires detecting peaks first.
        """
        try:
            # Quick beat detection for SQI purposes
            # We use a simple distance-based finder or neurokit if available
            import neurokit2 as nk
            
            # Normalize
            sig_norm = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)
            
            if signal_type == 'ecg':
                # Fast peak detect
                _, info = nk.ecg_peaks(sig_norm, sampling_rate=self.fs)
                peaks = info['ECG_R_Peaks']
                epoch_start, epoch_end = -0.2, 0.4 # Seconds
            else:
                # PPG
                # Clean slightly first for better peak detect?
                # Simple find_peaks from voltage
                from scipy.signal import find_peaks
                peaks, _ = find_peaks(sig_norm, distance=self.fs*0.5, prominence=0.5)
                epoch_start, epoch_end = -0.3, 0.3
                
            if len(peaks) < 3: return 0.0 # Need beats to compare
            
            # Extract Epochs
            epochs = []
            valid_peaks = []
            
            win_start = int(epoch_start * self.fs)
            win_end = int(epoch_end * self.fs)
            
            for p in peaks:
                start = p + win_start
                end = p + win_end
                
                if start >= 0 and end < len(sig_norm):
                    ep = sig_norm[start:end]
                    epochs.append(ep)
                    valid_peaks.append(p)
            
            if len(epochs) < 3: return 0.0
            
            epochs = np.array(epochs)
            # Median Template
            template = np.median(epochs, axis=0)
            
            # Correlate each beat with template
            corrs = []
            for ep in epochs:
                # Pearson correlation
                if np.std(ep) > 1e-6 and np.std(template) > 1e-6:
                    c = np.corrcoef(ep, template)[0, 1]
                    corrs.append(c)
                else:
                    corrs.append(0.0)
            
            # SQI is mean correlation
            # Heavily penalize negative correlations (artifacts)
            clean_corrs = [max(0, c) for c in corrs]
            sqi_score = np.mean(clean_corrs)
            
            return sqi_score
            
        except Exception as e:
            self.logger.debug(f"Template SQI failed: {e}")
            return 0.0

    def analyze_window(self, ecg_win: np.ndarray, ppg_win: np.ndarray, 
                       method: str = 'rule_based', custom_config: dict = None) -> dict:
        """
        Analyzes window.
        method: 'rule_based' (legacy) or 'template'
        custom_config: Dictionary of thresholds to use instead of loading file (for real-time GUI updates)
        """
        if custom_config:
            self.config = custom_config
        else:
            self.config_manager.load_config()
            self.config = self.config_manager.get('sqi')
        
        # Default scores
        metrics = {}
        ecg_score = 0.0
        ppg_score = 0.0
        
        if method == 'template':
            ecg_score = self.compute_template_sqi(ecg_win, 'ecg')
            ppg_score = self.compute_template_sqi(ppg_win, 'ppg')
            
            # Update metrics dict with standard ones too for reference?
            # Or just minimal? Let's compute standard too so table isn't empty on other cols
            # but override the main score
            ecg_met = self.compute_ecg_sqi(ecg_win)
            ppg_met = self.compute_ppg_sqi(ppg_win)
            metrics.update(ecg_met)
            metrics.update(ppg_met)
            
            metrics['ecg_sqi_score'] = ecg_score
            metrics['ppg_sqi_score'] = ppg_score
            
        else:
            # Rule Based / Standard
            ecg_met = self.compute_ecg_sqi(ecg_win)
            ppg_met = self.compute_ppg_sqi(ppg_win)
            metrics.update(ecg_met)
            metrics.update(ppg_met)
        
        # Pass/Fail Check
        # Use simple threshold on the final score
        ecg_pass = metrics['ecg_sqi_score'] >= self.config['ecg']['sqi_score_min']
        ppg_pass = metrics['ppg_sqi_score'] >= self.config['ppg']['sqi_score_min']
        
        # If using rule_based, we might apply extra logic (Legacy)
        if method == 'rule_based':
            # Detailed Rules Check (PPG)
            ppg_rules = self.config['ppg'].get('rules', {})
            if 'min_skew_abs' in ppg_rules:
                if abs(metrics['ppg_skew']) < ppg_rules['min_skew_abs']: ppg_pass = False
            if 'min_entropy' in ppg_rules:
                if metrics['ppg_entropy'] < ppg_rules['min_entropy']: ppg_pass = False
            
            # Detailed Rules Check (ECG)
            ecg_rules = self.config['ecg'].get('rules', {})
            if 'min_kurtosis' in ecg_rules:
                if metrics['ecg_kurtosis'] < ecg_rules['min_kurtosis']: ecg_pass = False

            # Valid PPG check (Skewness > 0)
            # "Clean PPG beats are asymmetric. If S_sqi > 0 (positive skew), the window likely contains valid beats."
            if metrics.get('ppg_skew', 0) <= 0: 
                ppg_pass = False
        
        # Check BPM Diff - DISABLED as per user request
        # rate_pass = True 
        # hr_diff = 0.0
        # (HR check moved to post-detection validation if needed)
        
        is_good = ecg_pass and ppg_pass
        hr_diff = 0.0
        rate_pass = True
        
        is_good = ecg_pass and ppg_pass and rate_pass
        
        metrics.update({
            'hr_diff': hr_diff,
            'ecg_pass': ecg_pass,
            'ppg_pass': ppg_pass,
            'rate_pass': rate_pass,
            'is_good': is_good
        })
        
        return metrics

    def cluster_windows(self, metrics_list: list, n_clusters: int = 4) -> dict:
        """
        Groups metrics into clusters using K-Means.
        Returns labels and centroids for visualization.
        """
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        from sklearn.impute import SimpleImputer
        
        if not metrics_list: return {}
        
        df = pd.DataFrame(metrics_list)
        
        # Select Features (Dual Signal: ECG + PPG)
        features = [
            'ppg_skew', 'ppg_kurtosis', 'ppg_entropy', 'ppg_rel_power', 'ppg_zcr', 'ppg_total_power',
            'ecg_skew', 'ecg_kurtosis', 'ecg_rel_power_qrs', 'ecg_zcr', 'ecg_total_power'
        ]
        
        # Filter columns that actually exist
        use_cols = [c for c in features if c in df.columns]
        if not use_cols: return {}
             
        X = df[use_cols].values
        
        # Handle NaNs/Infs
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        if len(X) < n_clusters: return {}
            
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        # Calculate centroids (mean scaled values for now)
        centroids = kmeans.cluster_centers_
        
        result = {
            'labels': labels,
            'feature_cols': use_cols,
            'model': kmeans,
            'scaler': scaler,
            'centroids': centroids
        }
        return result
