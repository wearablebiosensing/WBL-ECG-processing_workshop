import numpy as np
from scipy import signal

def tidy_beats(beat_indices):
    """
    Cleans up beat detections.
    Makes the vector of beat indices into a column vector of unique values.
    """
    beat_indices = np.array(beat_indices)
    beat_indices = beat_indices.flatten()
    beat_indices = np.unique(beat_indices)
    return beat_indices

class MSPTDfastv2:
    def __init__(self):
        pass

    @staticmethod
    def detect_beats(sig, fs):
        """
        MSPTDfast (v2.0) PPG beat detector.
        """
        options = {
            'find_trs': True,
            'find_pks': True,
            'do_ds': True,
            'ds_freq': 20,
            'use_reduced_lms_scales': True,
            'win_len': 6,
            'win_overlap': 0.2,
            'optimisation': False,
            'plaus_hr_bpm': [30, 200],
            'lms_calc_method': 1
        }
        
        peaks, onsets = MSPTDfastv2.msptdpcref_beat_detector(sig, fs, options)
        return peaks, onsets

    @staticmethod
    def msptdpcref_beat_detector(sig, fs, options):
        no_samps_in_win = int(options['win_len'] * fs)
        if len(sig) <= no_samps_in_win:
            win_starts = [0]
            win_ends = [len(sig)]
        else:
            win_offset = int(round(no_samps_in_win * (1 - options['win_overlap'])))
            win_starts = list(range(0, len(sig) - no_samps_in_win + 1, win_offset))
            win_ends = [s + no_samps_in_win for s in win_starts]
            if win_ends[-1] < len(sig):
                win_starts.append(len(sig) - no_samps_in_win)
                win_ends.append(len(sig))
        
        ds_factor = 1
        ds_fs = fs
        do_ds = options['do_ds']
        if do_ds:
            min_fs = options['ds_freq']
            if fs > min_fs:
                ds_factor = int(np.floor(fs / min_fs))
                ds_fs = fs / ds_factor
            else:
                do_ds = False
        
        peaks = []
        onsets = []
        
        for win_no in range(len(win_starts)):
            win_sig = sig[win_starts[win_no]:win_ends[win_no]]
            
            if do_ds:
                rel_sig = win_sig[::ds_factor]
                rel_fs = ds_fs
            else:
                rel_sig = win_sig
                rel_fs = fs
            
            p, t = MSPTDfastv2.detect_peaks_and_onsets_using_msptd(rel_sig, rel_fs, options)
            
            if do_ds:
                p = p * ds_factor
                t = t * ds_factor
            
            tol_durn = 0.05
            if rel_fs < 10:
                tol_durn = 0.2
            elif rel_fs < 20:
                tol_durn = 0.1
            
            tol = int(np.ceil(fs * tol_durn))
            
            for pk_no in range(len(p)):
                start_idx = max(0, p[pk_no] - tol)
                end_idx = min(len(win_sig), p[pk_no] + tol + 1)
                
                if start_idx < end_idx:
                    segment = win_sig[start_idx:end_idx]
                    if len(segment) > 0:
                        temp = np.argmax(segment)
                        p[pk_no] = start_idx + temp
            
            for onset_no in range(len(t)):
                start_idx = max(0, t[onset_no] - tol)
                end_idx = min(len(win_sig), t[onset_no] + tol + 1)
                
                if start_idx < end_idx:
                    segment = win_sig[start_idx:end_idx]
                    if len(segment) > 0:
                        temp = np.argmin(segment)
                        t[onset_no] = start_idx + temp
            
            win_peaks = p + win_starts[win_no]
            peaks.extend(win_peaks)
            win_onsets = t + win_starts[win_no]
            onsets.extend(win_onsets)
            
        peaks = tidy_beats(peaks)
        onsets = tidy_beats(onsets)
        
        return peaks, onsets

    @staticmethod
    def detect_peaks_and_onsets_using_msptd(x, fs, options):
        N = len(x)
        L = int(np.ceil(N / 2) - 1)
        
        plaus_hr_hz = np.array(options['plaus_hr_bpm']) / 60.0
        init_scales = np.arange(1, L + 1)
        durn_signal = len(x) / fs
        init_scales_fs = (L / init_scales) / durn_signal
        
        if options['use_reduced_lms_scales']:
            init_scales_inc_log = init_scales_fs >= plaus_hr_hz[0]
        else:
            init_scales_inc_log = np.ones(len(init_scales), dtype=bool)
            
        true_indices = np.where(init_scales_inc_log)[0]
        if len(true_indices) > 0:
            max_scale = true_indices[-1] + 1
        else:
            max_scale = 1
            
        x = signal.detrend(x)
        
        m_max = None
        m_min = None
        
        if options['find_pks']:
            m_max = np.zeros((max_scale, N), dtype=bool)
        if options['find_trs']:
            m_min = np.zeros((max_scale, N), dtype=bool)
            
        m_max, m_min = MSPTDfastv2.find_lms_using_msptd_approach(max_scale, x, options)
        
        lambda_max = 0
        lambda_min = 0
        
        if options['find_pks']:
            gamma_max = np.sum(m_max, axis=1)
            lambda_max = np.argmax(gamma_max)
            
        if options['find_trs']:
            gamma_min = np.sum(m_min, axis=1)
            lambda_min = np.argmax(gamma_min)
            
        first_scale_idx = np.where(init_scales_inc_log)[0][0] if np.any(init_scales_inc_log) else 0
        
        if options['find_pks']:
            if lambda_max >= first_scale_idx:
                m_max_subset = m_max[first_scale_idx : lambda_max + 1, :]
            else:
                m_max_subset = np.zeros((0, N), dtype=bool)
                
        if options['find_trs']:
            if lambda_min >= first_scale_idx:
                m_min_subset = m_min[first_scale_idx : lambda_min + 1, :]
            else:
                m_min_subset = np.zeros((0, N), dtype=bool)

        p = []
        t = []
        
        if options['find_pks']:
            if m_max_subset.shape[0] > 0:
                m_max_sum = np.sum(~m_max_subset, axis=0)
                p = np.where(m_max_sum == 0)[0]
            else:
                p = np.array([])
                
        if options['find_trs']:
            if m_min_subset.shape[0] > 0:
                m_min_sum = np.sum(~m_min_subset, axis=0)
                t = np.where(m_min_sum == 0)[0]
            else:
                t = np.array([])
                
        return p, t

    @staticmethod
    def find_lms_using_msptd_approach(max_scale, x, options):
        N = len(x)
        m_max = None
        m_min = None
        
        if options['find_pks']:
            m_max = np.zeros((max_scale, N), dtype=bool)
            for k in range(1, max_scale + 1):
                j_start = k
                j_end = N - k
                if j_start < j_end:
                    center = x[j_start:j_end]
                    left = x[j_start-k : j_end-k]
                    right = x[j_start+k : j_end+k]
                    is_max = (center > left) & (center > right)
                    m_max[k-1, j_start:j_end] = is_max
                    
        if options['find_trs']:
            m_min = np.zeros((max_scale, N), dtype=bool)
            for k in range(1, max_scale + 1):
                j_start = k
                j_end = N - k
                if j_start < j_end:
                    center = x[j_start:j_end]
                    left = x[j_start-k : j_end-k]
                    right = x[j_start+k : j_end+k]
                    is_min = (center < left) & (center < right)
                    m_min[k-1, j_start:j_end] = is_min
                    
        return m_max, m_min
