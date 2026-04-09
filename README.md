# WBL ECG Processing Workshop & Analysis GUI

A comprehensive **Python-based ECG Signal Processing Toolkit** and **Interactive PyQtGraph GUI** for analyzing, comparing, and validating wearable biometric sensors against clinical-grade ground truth devices (e.g., BIOPAC).

This repository contains both the core mathematical algorithms for signal processing and the graphical interface designed for clinical visualization, dataset augmentation, and statistical evaluation.

---

## 🚀 Quick Start GUI

The easiest way to perform comparative analysis between a wearable sensor and clinical ground truth is via the included PyQT Application.

```bash
# Clone the repository
git clone <this-repo>
cd WBL-ECG-processing_workshop

# Create a conda environment
conda create -n ecgworkshop python=3.10 -y
conda activate ecgworkshop

# Install primary dependencies
pip install numpy pandas scipy matplotlib seaborn neurokit2 wfdb PyQt6 pyqtgraph

# Install optional (but recommended) dependencies for Deep Learning and Wavelets
pip install pywt ssqueezepy torch
```

Launch the interactive GUI:
```bash
python gui.py
```

### GUI Features
- **Raw Signal Ingestion:** Cross-correlates and intelligently aligns asynchronous sensors via polarity-invariant synchronization.
- **Interactive Pipeline & Scalograms:** View raw signals, wavelet scalograms, and processed algorithmic outputs step-by-step with synchronized panning.
- **Adaptive Evaluation:** Uses Dynamic Time Warping (DTW)-inspired moving medians to calculate beat-by-beat F1 scores while ignoring independent hardware clock-drift.
- **Adversarial Augmentation:** Locally simulate motion artifacts, 60Hz PLI interference, and Gaussian white noise interactively to test algorithm robustness.
- **Comparison Metrics:** Full Bland-Altman statistical analysis, True/False Positives scoring, and HR MAE evaluation natively displayed.

---

## 🛠️ Core Processing Pipeline (`ecg_analysis/`)

The backend engine (`ecg_analysis/`) driving the GUI can be directly imported into Jupyter Notebooks or batch processing pipelines. 

### Data Parsers (`parsers.py`)
- **BIOPAC** – Dynamically parses clinical tab-delimited files (.txt) regardless of variable header length.
- **CareWear & MAX30003** – Unpacks Extensionless CSV files recorded from Bluetooth belts. Supports raw ADC conversion matrices natively directly from integer format.

### Intelligent Synchronization (`sync.py`)
Hardware clocks differ. By calculating the cross-correlation of squared Hilbert energy envelopes, the sync module automatically adjusts padding and global starting lag bounds for wearable devices matching clinical data—**even if the ECG patches are inverted due to lead placement**.

### Signal Preprocessing (`preprocessing.py`)
Defines the configurable multi-stage filtering algorithm:
1. Interpolation over NaN packets.
2. IIR Notch filtering (e.g., 50/60 Hz powerline mapping).
3. Zero-phase SOS Butterworth bandpass (0.5Hz – 40Hz).
4. Double-pass Median Filter (Baseline Wander Removal).
5. Continuous Wavelet Transform (CWT) PyWavelets/SSQ bandpass isolation.
6. **Autonomic Physiological Inversion:** Fixes QRS complex deflections by tracking distributional skewness polarity.

### Classical & Deep Evaluators (`detectors.py`, `evaluation.py`)
Supports 12 independent detector frameworks natively, including deep-learning architectures (RPNet) and standard statistical thresholders (NeuroKit2, Pan-Tompkins).

For scoring datasets against Ground Truth annotations, the pipeline utilizes **Time-Warped Sub-RR Alignment**. Instead of a fixed temporal delay index, it analyzes rolling 11-beat neighborhoods via dynamic medians, calculating continuous time-distortions without bleeding False Positives due to standard crystal drift hardware inaccuracies.

---

## 🧬 Repository Layout

```
WBL-ECG-processing_workshop/
├── gui.py                                # Primary Graphical Interface Application
├── ecg_analysis/
│   ├── detectors.py                      # NeuroKit2 + Custom RPNet Integrations
│   ├── evaluation.py                     # Non-linear Time-Warping Scoring & HR MAE
│   ├── parsers.py                        # Hardware Decoders (BIOPAC & Wearables)
│   ├── pipeline.py                       # Orchestrator Node for File-to-Metrics
│   ├── preprocessing.py                  # Filters, Smoothing, & Skew Auto-Inverts
│   └── sync.py                           # Polarity-Invariant XCorr Lead Synchronization
├── reference_codes/                      # Academic & Architecture Reference Modules
│   ├── ECG_augumentor_dataset.py
│   └── RPNet/                            # Deep Learning PyTorch Backbone
├── sample_data/                          # Clinical BIOPAC + Wearable Calibration Runs
└── README.md
```

## Licensing & Attribution
Follow the licenses in the `reference_codes/` root branches (e.g., RPNet subtree MIT license) for redistributed code. The core parsing, alignment, and PyQtGraph toolkits are intended for research, education, and technical evaluation.