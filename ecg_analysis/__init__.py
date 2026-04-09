"""
ECG Analysis Pipeline
=====================
End-to-end ECG heart rate analysis with SQI-gated detection,
polarity-invariant synchronization, and HR-based evaluation.

Supports: BIOPAC AcqKnowledge + Baby Belt / CareWear Belt
Detectors: NeuroKit2 default, XQRS, PROMAC, RPNet (CUDA-batched)
"""

from .parsers import load_biopac, load_belt, load_carewear_biopac, load_carewear_belt
from .preprocessing import preprocess_ecg, PreprocessConfig
from .sqi import compute_window_sqi, assess_quality
from .sync import sync_signals, polarity_invariant_xcorr
from .detectors import detect_rpeaks, DetectorConfig
from .evaluation import evaluate_hr, evaluate_beats, compute_hr_metrics
from .pipeline import (analyze_file, analyze_window, export_hr_csv,
                       compare_against_biopac, export_comparison_csv)
from .evaluation import (peaks_to_instantaneous_hr, median_smooth_hr,
                         window_averaged_hr)

__version__ = "1.0.0"
