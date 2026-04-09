#!/usr/bin/env python3
"""
ECG Analysis GUI  --  PyQt6 frontend + PyQtGraph backend.

Usage:
    python gui.py

Features:
  - PyqtGraph replacing Matplotlib for performance.
  - Custom Augmentations Tab
  - Interactive Scalogram visualization connected via ViewBox
"""

import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import traceback
from pathlib import Path
import numpy as np

# Path bootstrap
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout,
    QTabWidget, QGroupBox, QLabel, QLineEdit, QPushButton,
    QComboBox, QCheckBox, QDoubleSpinBox, QSpinBox,
    QFileDialog, QProgressBar,
    QTableWidget, QTableWidgetItem,
    QSizePolicy, QMessageBox, QHeaderView, QScrollArea,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor

import pyqtgraph as pg

# Configure PyQtGraph global settings
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

try:
    from ecg_analysis import (
        load_biopac, load_belt,
        load_carewear_biopac, load_carewear_belt,
        sync_signals,
        PreprocessConfig, DetectorConfig,
        analyze_file, compare_against_biopac, export_hr_csv,
    )
    from ecg_analysis.preprocessing import get_wavelet_heatmap
    _PKG_OK = True
except ImportError as _ie:
    _PKG_OK  = False
    _PKG_ERR = str(_ie)


# =========================================================================== #
#  Background analysis worker
# =========================================================================== #

def apply_augmentations(cfg, ecg, fs):
    """Applies augmentation based on UI booleans."""
    rng = np.random.default_rng(42)
    try:
        from reference_codes.ECG_augumentor_dataset import add_complex_baseline_wander
        if cfg.get("aug_bw"):
            ecg = add_complex_baseline_wander(ecg, fs, rng, severity=cfg.get("aug_bw_val", 1.0))
    except ImportError:
        pass
        
    try:
        if cfg.get("aug_noise"):
            ecg = ecg + rng.normal(0, cfg.get("aug_noise_val", 0.6), len(ecg))
        if cfg.get("aug_pli"):
            t = np.arange(len(ecg)) / fs
            # 60Hz PLI scaled to 5% of signal standard deviation
            ecg = ecg + (0.05 * np.std(ecg)) * np.sin(2 * np.pi * 60 * t)
    except Exception:
        pass
    return ecg

class AnalysisWorker(QThread):
    progress = pyqtSignal(int)
    status   = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, params: dict, load_only: bool = False, parent=None):
        super().__init__(parent)
        self.params    = params
        self.load_only = load_only

    def _step(self, pct: int, msg: str):
        self.progress.emit(pct)
        self.status.emit(msg)

    def run(self):
        try:
            p   = self.params
            fmt = p["format"]
            res = {}

            # 1. Load BIOPAC
            self._step(5, "Loading BIOPAC file...")
            if fmt == "carewear":
                bp = load_carewear_biopac(p["biopac_path"], ecg_col=p["ecg_col"])
            else:
                bp = load_biopac(p["biopac_path"])
            
            bp["ecg"] = apply_augmentations(p, bp["ecg"], bp["fs"])
            res["biopac_raw"] = bp

            # 2. Load Belt
            belt_path = (p.get("belt_path") or "").strip() or None
            if belt_path:
                self._step(12, "Loading Belt file...")
                if fmt == "carewear":
                    bl = load_carewear_belt(belt_path, ecg_mode=p["belt_mode"])
                else:
                    bl = load_belt(belt_path)
                
                res["belt_raw_unaugmented"] = bl["ecg"].copy()
                bl["ecg"] = apply_augmentations(p, bl["ecg"], bl["fs"])
                res["belt_raw"] = bl
            else:
                bl = None
                res["belt_raw_unaugmented"] = None
                res["belt_raw"] = None

            # 3. Synchronisation
            if bl is not None:
                self._step(20, "Polarity-invariant xcorr sync (squared+Hilbert envelope)...")
                sync = sync_signals(bp, bl)
                res["sync"] = sync
                bp_work = dict(
                    ecg    = sync["biopac_ecg_clipped"],
                    fs     = sync["common_fs"],
                    ts_ms  = sync["biopac_ts_ms_clipped"],
                    time_s = sync["biopac_time_clipped"],
                )
                bl_work = dict(
                    ecg    = sync["belt_ecg_aligned"],
                    fs     = sync["common_fs"],
                    ts_ms  = sync["belt_ts_ms_aligned"],
                    time_s = sync["belt_time_aligned"],
                )
            else:
                res["sync"] = None
                bp_work, bl_work = bp, None

            if self.load_only:
                self._step(100, "Files loaded -- click 'Run Full Analysis' to continue.")
                self.finished.emit(res)
                return

            # 4. Configs
            prep = PreprocessConfig(
                apply_notch    = p["notch"],
                notch_freq     = p["notch_freq"],
                apply_bandpass = p["bandpass"],
                bp_lowcut      = p["bp_low"],
                bp_highcut     = p["bp_high"],
                apply_baseline = p["baseline"],
                apply_wavelet  = p["wavelet"],
                wavelet_method = p["wavelet_meth"],
                wavelet_fmin   = p["wav_fmin"],
                wavelet_fmax   = p["wav_fmax"],
            )
            det = DetectorConfig(
                rpnet_model_dir  = p.get("rpnet_dir", "reference_codes/RPNet"),
                rpnet_batch_size = p.get("batch_size", 16),
            )

            # 5. BIOPAC
            self._step(30, "Analysing BIOPAC (SQI + detect)...")
            bp_res = analyze_file(bp_work, prep, p["detector"], det, p["window_s"], p["stride_s"], p["nk_sqi"], True)
            res["biopac_analysis"] = bp_res

            # Computing Scalograms
            self._step(45, "Computing BIOPAC Scalogram...")
            if p["wavelet"]:
                Wx, freqs = get_wavelet_heatmap(bp_res["ecg_clean"], bp_res["fs"], p["wav_fmin"], p["wav_fmax"])
                if Wx is not None:
                    res["biopac_scalo"] = np.abs(Wx)[::-1, :].T
                    res["scalo_freqs"] = freqs[::-1]

            if bl_work is not None:
                self._step(60, "Analysing Belt (SQI + detect)...")
                bl_res = analyze_file(bl_work, prep, p["detector"], det, p["window_s"], p["stride_s"], p["nk_sqi"], True)
                res["belt_analysis"] = bl_res
                
                self._step(80, "Computing Belt Scalogram...")
                if p["wavelet"]:
                    Wx_bl, _ = get_wavelet_heatmap(bl_res["ecg_clean"], bl_res["fs"], p["wav_fmin"], p["wav_fmax"])
                    if Wx_bl is not None:
                        res["belt_scalo"] = np.abs(Wx_bl)[::-1, :].T

                self._step(90, "Computing Metrics...")
                res["comparison"] = compare_against_biopac(bl_res, bp_res, p.get("tol_ms", 100), p["window_s"], p["stride_s"])
            else:
                res["belt_analysis"] = None
                res["comparison"]    = None

            self._step(100, "Analysis complete.")
            self.finished.emit(res)

        except Exception:
            self.error.emit(traceback.format_exc())

# =========================================================================== #
#  Main GUI Window
# =========================================================================== #

class ECGAnalysisGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ECG Analysis  --  WBL Workshop")
        self.resize(1500, 950)

        self._results: dict = {}
        self._worker: AnalysisWorker | None = None

        if not _PKG_OK:
            QMessageBox.critical(
                self, "Import error",
                f"Could not import ecg_analysis:\n{_PKG_ERR}\n\n"
                "Run gui.py from the repository root directory.")

        self._build_ui()
        self._wire()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        hl = QHBoxLayout(root)
        hl.setContentsMargins(4, 4, 4, 4)
        hl.setSpacing(6)
        hl.addWidget(self._build_left_panel())
        hl.addWidget(self._build_tabs(), stretch=1)

        self._prog = QProgressBar()
        self._prog.setMaximumWidth(180)
        self._prog.setVisible(False)
        self.statusBar().addPermanentWidget(self._prog)
        self.statusBar().showMessage("Ready")

    def _build_left_panel(self) -> QScrollArea:
        container = QWidget()
        container.setFixedWidth(314)
        lay = QVBoxLayout(container)
        lay.addWidget(self._grp_files())
        lay.addWidget(self._grp_augment())
        lay.addWidget(self._grp_preprocess())
        lay.addWidget(self._grp_detector())
        lay.addWidget(self._grp_run())
        lay.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(container)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedWidth(350)
        return scroll

    def _grp_files(self) -> QGroupBox:
        g = QGroupBox("Data Files")
        lay = QVBoxLayout(g)
        r = QHBoxLayout(); r.addWidget(QLabel("Format:")); self._fmt = QComboBox()
        self._fmt.addItems(["CareWear  (MAX30003 belt)", "Baby Belt (100 Hz)"])
        r.addWidget(self._fmt); lay.addLayout(r)

        lay.addWidget(QLabel("BIOPAC file (.txt):"))
        r2 = QHBoxLayout()
        self._bp_edit = QLineEdit(); self._bp_edit.setPlaceholderText("select .txt ...")
        self._bp_btn = QPushButton("..."); self._bp_btn.setFixedWidth(28)
        r2.addWidget(self._bp_edit); r2.addWidget(self._bp_btn); lay.addLayout(r2)

        r3 = QHBoxLayout(); r3.addWidget(QLabel("ECG channel:")); self._ecg_col = QComboBox()
        self._ecg_col.addItems(["CH9  (raw ECG)", "CH40 (AHA filtered)"])
        r3.addWidget(self._ecg_col); lay.addLayout(r3)

        lay.addWidget(QLabel("Belt file (.csv or extensionless):"))
        r4 = QHBoxLayout()
        self._bl_edit = QLineEdit(); self._bl_edit.setPlaceholderText("select belt file ...")
        self._bl_btn = QPushButton("..."); self._bl_btn.setFixedWidth(28)
        r4.addWidget(self._bl_edit); r4.addWidget(self._bl_btn); lay.addLayout(r4)

        r5 = QHBoxLayout(); r5.addWidget(QLabel("Belt mode:")); self._belt_mode = QComboBox()
        self._belt_mode.addItems(["NORMALIZE (z-score)", "MAX30003 (decode ADC -> mV)"])
        r5.addWidget(self._belt_mode); lay.addLayout(r5)

        self._load_btn = QPushButton("Load & Sync Signals")
        self._load_btn.setStyleSheet("QPushButton{background:#1565C0;color:white;font-weight:bold;padding:5px;border-radius:3px}QPushButton:hover{background:#1976D2}")
        lay.addWidget(self._load_btn)
        return g

    def _grp_augment(self) -> QGroupBox:
        g = QGroupBox("Testing Augmentations")
        lay = QVBoxLayout(g)
        
        r1 = QHBoxLayout(); self._aug_bw = QCheckBox("Simulate Baseline Wander")
        self._aug_bw_val = QDoubleSpinBox(); self._aug_bw_val.setRange(0.1, 10.0); self._aug_bw_val.setValue(1.0)
        r1.addWidget(self._aug_bw); r1.addWidget(QLabel("Sev:")); r1.addWidget(self._aug_bw_val); lay.addLayout(r1)
        
        r2 = QHBoxLayout(); self._aug_noise = QCheckBox("Add Gaussian Noise")
        self._aug_noise_val = QDoubleSpinBox(); self._aug_noise_val.setRange(0.01, 5.0); self._aug_noise_val.setValue(0.6)
        r2.addWidget(self._aug_noise); r2.addWidget(QLabel("Scale:")); r2.addWidget(self._aug_noise_val); lay.addLayout(r2)
        
        self._aug_pli = QCheckBox("Include 60Hz Powerline")
        lay.addWidget(self._aug_pli)
        return g

    def _grp_preprocess(self) -> QGroupBox:
        g = QGroupBox("Preprocessing Pipeline")
        lay = QVBoxLayout(g)
        r = QHBoxLayout(); self._notch_cb = QCheckBox("Notch filter"); self._notch_cb.setChecked(True)
        self._notch_hz = QDoubleSpinBox(); self._notch_hz.setRange(50, 60); self._notch_hz.setValue(60); self._notch_hz.setFixedWidth(72)
        r.addWidget(self._notch_cb); r.addWidget(self._notch_hz); lay.addLayout(r)

        self._bp_cb = QCheckBox("Bandpass filter"); self._bp_cb.setChecked(True); lay.addWidget(self._bp_cb)
        r2 = QHBoxLayout(); r2.addWidget(QLabel("Low:")); self._bp_lo = QDoubleSpinBox(); self._bp_lo.setRange(0.1, 5); self._bp_lo.setValue(0.5); r2.addWidget(self._bp_lo)
        r2.addWidget(QLabel("High:")); self._bp_hi = QDoubleSpinBox(); self._bp_hi.setRange(10, 100); self._bp_hi.setValue(40); r2.addWidget(self._bp_hi)
        lay.addLayout(r2)

        self._base_cb = QCheckBox("Baseline removal (double median)"); self._base_cb.setChecked(True); lay.addWidget(self._base_cb)
        
        r3 = QHBoxLayout()
        self._wav_cb = QCheckBox("Wavelet denoising"); self._wav_meth = QComboBox()
        self._wav_meth.addItems(["ssq  (SSQ-CWT)", "pywt  (PyWavelets)"]); self._wav_meth.setEnabled(False)
        r3.addWidget(self._wav_cb); r3.addWidget(self._wav_meth); lay.addLayout(r3)

        r4 = QHBoxLayout()
        r4.addWidget(QLabel("W-Low:")); self._wav_fmin = QDoubleSpinBox(); self._wav_fmin.setRange(0.1, 20); self._wav_fmin.setValue(5.0)
        r4.addWidget(QLabel("W-High:")); self._wav_fmax = QDoubleSpinBox(); self._wav_fmax.setRange(10, 100); self._wav_fmax.setValue(40.0)
        r4.addWidget(self._wav_fmin); r4.addWidget(self._wav_fmax); lay.addLayout(r4)

        return g

    def _grp_detector(self) -> QGroupBox:
        g = QGroupBox("R-Peak Detector")
        lay = QVBoxLayout(g)
        r = QHBoxLayout(); r.addWidget(QLabel("Method:")); self._det = QComboBox()
        self._det.addItems(["neurokit", "promac", "xqrs", "rpnet"]); r.addWidget(self._det); lay.addLayout(r)

        lay.addWidget(QLabel("RPNet model directory:"))
        r2 = QHBoxLayout(); self._rpnet_dir = QLineEdit("reference_codes/RPNet")
        self._rpnet_dir_btn = QPushButton("..."); self._rpnet_dir_btn.setFixedWidth(28)
        r2.addWidget(self._rpnet_dir); r2.addWidget(self._rpnet_dir_btn); lay.addLayout(r2)

        r3 = QHBoxLayout(); r3.addWidget(QLabel("GPU batch:")); self._batch = QSpinBox(); self._batch.setValue(16)
        r3.addWidget(self._batch); r3.addWidget(QLabel("Beat tol:")); self._tol = QSpinBox(); self._tol.setRange(20, 500); self._tol.setValue(100); r3.addWidget(self._tol)
        lay.addLayout(r3)
        return g

    def _grp_run(self) -> QGroupBox:
        g = QGroupBox("Analysis Settings")
        lay = QVBoxLayout(g)
        r = QHBoxLayout(); r.addWidget(QLabel("Window:")); self._win = QDoubleSpinBox(); self._win.setValue(10); r.addWidget(self._win)
        r.addWidget(QLabel("Stride:")); self._stride = QDoubleSpinBox(); self._stride.setValue(5); r.addWidget(self._stride)
        lay.addLayout(r)

        self._nk_sqi = QCheckBox("NeuroKit ECG SQI per window"); lay.addWidget(self._nk_sqi)
        
        self._run_btn = QPushButton("Run Full Analysis")
        self._run_btn.setEnabled(False)
        self._run_btn.setStyleSheet("QPushButton{background:#2E7D32;color:white;font-weight:bold;padding:5px;border-radius:3px}QPushButton:disabled{background:#888}QPushButton:hover:!disabled{background:#388E3C}")
        lay.addWidget(self._run_btn)

        self._export_btn = QPushButton("Export HR CSV")
        self._export_btn.setEnabled(False)
        self._export_btn.setStyleSheet("QPushButton{background:#E65100;color:white;font-weight:bold;padding:5px;border-radius:3px}QPushButton:disabled{background:#888}QPushButton:hover:!disabled{background:#F4511E}")
        lay.addWidget(self._export_btn)
        return g

    def _build_tabs(self) -> QTabWidget:
        tabs = QTabWidget()

        # Tab 0: Raw Signals Sync
        self._gw_sig = pg.GraphicsLayoutWidget()
        tabs.addTab(self._gw_sig, "Raw Signals")
        self._p_sig_bp = self._gw_sig.addPlot(title="BIOPAC")
        self._p_sig_bp.setLabels(bottom='Time (s)', left='Amplitude (mV)')
        self._gw_sig.nextRow()
        self._p_sig_bl = self._gw_sig.addPlot(title="Belt")
        self._p_sig_bl.setLabels(bottom='Time (s)', left='Amplitude (mV)')
        self._p_sig_bl.setXLink(self._p_sig_bp)

        # Tab 1: Interactive Pipeline & Scalograms
        self._gw_interact = pg.GraphicsLayoutWidget()
        tabs.addTab(self._gw_interact, "Interactive Pipeline")

        self._pi_bp_wave = self._gw_interact.addPlot(title="BIOPAC ECG")
        self._pi_bp_wave.setLabels(bottom='Time (s)', left='Amplitude (mV)')
        self._pi_bp_wave.showGrid(x=True, y=True, alpha=0.3)
        self._gw_interact.nextRow()

        self._pi_bp_sc = self._gw_interact.addPlot(title="BIOPAC Scalogram")
        self._pi_bp_sc.setLabels(bottom='Time (s)', left='Freq (Hz)')
        self._pi_bp_sc.setXLink(self._pi_bp_wave)
        self._img_bp_sc = pg.ImageItem()
        self._pi_bp_sc.addItem(self._img_bp_sc)
        self._img_bp_sc.setLookupTable(pg.colormap.get('plasma').getLookupTable())
        self._gw_interact.nextRow()

        self._pi_bl_wave = self._gw_interact.addPlot(title="Belt Denoised ECG")
        self._pi_bl_wave.setLabels(bottom='Time (s)', left='Amplitude (mV)')
        self._pi_bl_wave.setXLink(self._pi_bp_wave)
        self._pi_bl_wave.showGrid(x=True, y=True, alpha=0.3)
        self._gw_interact.nextRow()

        self._pi_bl_sc = self._gw_interact.addPlot(title="Belt Scalogram")
        self._pi_bl_sc.setLabels(bottom='Time (s)', left='Freq (Hz)')
        self._pi_bl_sc.setXLink(self._pi_bp_wave)
        self._img_bl_sc = pg.ImageItem()
        self._pi_bl_sc.addItem(self._img_bl_sc)
        self._img_bl_sc.setLookupTable(pg.colormap.get('plasma').getLookupTable())

        # Scaling hook
        self._pi_bp_wave.sigRangeChanged.connect(self._update_scalogram_levels)
        
        # Tab 2: Augmentation Preview
        self._gw_prev = pg.GraphicsLayoutWidget()
        tabs.addTab(self._gw_prev, "Augment. Preview")
        self._p_prev_orig = self._gw_prev.addPlot(title="Belt Raw (Original)")
        self._p_prev_orig.setLabels(bottom='Time (s)', left='Amplitude (mV)')
        self._gw_prev.nextRow()
        self._p_prev_aug = self._gw_prev.addPlot(title="Belt Augmented")
        self._p_prev_aug.setLabels(bottom='Time (s)', left='Amplitude (mV)')
        self._p_prev_aug.setXLink(self._p_prev_orig)
        self._gw_prev.nextRow()
        self._p_prev_clean = self._gw_prev.addPlot(title="Belt Preprocessed (Pipeline Output)")
        self._p_prev_clean.setLabels(bottom='Time (s)', left='Amplitude (mV)')
        self._p_prev_clean.setXLink(self._p_prev_orig)

        # Tab 3: SQI Windows
        self._gw_sqi = pg.GraphicsLayoutWidget()
        tabs.addTab(self._gw_sqi, "SQI Windows")

        # Tab 4: Heart Rate
        self._gw_hr = pg.GraphicsLayoutWidget()
        tabs.addTab(self._gw_hr, "Heart Rate")
        self._p_ihr = self._gw_hr.addPlot(title="Instantaneous HR")
        self._p_ihr.addLegend()
        self._p_ihr.setLabels(bottom='Time (s)', left='Heart Rate (BPM)')
        self._gw_hr.nextRow()
        self._p_whr = self._gw_hr.addPlot(title="Window-Averaged HR")
        self._p_whr.addLegend()
        self._p_whr.setLabels(bottom='Window Index', left='Avg HR (BPM)')

        # Tab 5: Comparison Metrics
        mw = QWidget(); ml = QVBoxLayout(mw)
        self._mtbl = QTableWidget(0, 2)
        self._mtbl.setHorizontalHeaderLabels(["Metric", "Value"])
        self._mtbl.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._mtbl.maximumHeight = 300
        ml.addWidget(self._mtbl)
        self._gw_ba = pg.GraphicsLayoutWidget()
        ml.addWidget(self._gw_ba)
        self._p_ba = self._gw_ba.addPlot(title="Bland-Altman")
        self._p_ba.setLabels(bottom='Mean (BPM)', left='Difference (BPM)')
        tabs.addTab(mw, "Comparison Metrics")

        self._tabs = tabs
        return tabs

    def _wire(self):
        self._bp_btn.clicked.connect(lambda: _pick_file(self, self._bp_edit, "Select BIOPAC file"))
        self._bl_btn.clicked.connect(lambda: _pick_file(self, self._bl_edit, "Select Belt file"))
        self._rpnet_dir_btn.clicked.connect(self._pick_rpnet_dir)
        self._load_btn.clicked.connect(lambda: self._start(load_only=True))
        self._run_btn.clicked.connect(lambda: self._start(load_only=False))
        self._export_btn.clicked.connect(self._export)
        self._wav_cb.toggled.connect(self._wav_meth.setEnabled)
        self._fmt.currentIndexChanged.connect(self._on_fmt_change)

    def _on_fmt_change(self, idx: int):
        cw = (idx == 0)
        self._ecg_col.setEnabled(cw)
        self._belt_mode.setEnabled(cw)

    def _pick_rpnet_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select RPNet dir", str(_HERE))
        if d: self._rpnet_dir.setText(d)

    def _params(self) -> dict:
        cw = (self._fmt.currentIndex() == 0)
        return dict(
            format       = "carewear" if cw else "baby_belt",
            biopac_path  = self._bp_edit.text().strip(),
            belt_path    = self._bl_edit.text().strip() or None,
            ecg_col      = "CH9" if self._ecg_col.currentIndex() == 0 else "CH40",
            belt_mode    = "NORMALIZE" if self._belt_mode.currentIndex() == 0 else "MAX30003",
            notch        = self._notch_cb.isChecked(),
            notch_freq   = self._notch_hz.value(),
            bandpass     = self._bp_cb.isChecked(),
            bp_low       = self._bp_lo.value(),
            bp_high      = self._bp_hi.value(),
            baseline     = self._base_cb.isChecked(),
            wavelet      = self._wav_cb.isChecked(),
            wavelet_meth = "ssq" if self._wav_meth.currentIndex() == 0 else "pywt",
            wav_fmin     = self._wav_fmin.value(),
            wav_fmax     = self._wav_fmax.value(),
            detector     = self._det.currentText(),
            rpnet_dir    = self._rpnet_dir.text().strip(),
            batch_size   = self._batch.value(),
            tol_ms       = self._tol.value(),
            window_s     = self._win.value(),
            stride_s     = self._stride.value(),
            nk_sqi       = self._nk_sqi.isChecked(),
            aug_bw       = self._aug_bw.isChecked(),
            aug_bw_val   = self._aug_bw_val.value(),
            aug_noise    = self._aug_noise.isChecked(),
            aug_noise_val= self._aug_noise_val.value(),
            aug_pli      = self._aug_pli.isChecked()
        )

    def _start(self, load_only: bool):
        if self._worker and self._worker.isRunning(): return
        p = self._params()
        if not p["biopac_path"]: return QMessageBox.warning(self, "No file", "Select BIOPAC file.")
        
        self._load_btn.setEnabled(False)
        self._run_btn.setEnabled(False)
        self._export_btn.setEnabled(False)
        self._prog.setVisible(True)
        self._prog.setValue(0)

        self._worker = AnalysisWorker(p, load_only, self)
        self._worker.progress.connect(self._prog.setValue)
        self._worker.status.connect(self.statusBar().showMessage)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_err)
        self._worker.start()

    def _on_done(self, res: dict):
        self._results = res
        self._prog.setVisible(False)
        self._load_btn.setEnabled(True)
        self._run_btn.setEnabled(True)

        self._plot_raw_signals()

        if res.get("biopac_analysis"):
            self._plot_interactive_pipeline()
            self._plot_preview()
            self._plot_sqi()
            self._plot_hr()
            self._export_btn.setEnabled(True)

        if res.get("comparison"):
            self._plot_metrics()
            self._tabs.setCurrentIndex(1)

        self.statusBar().showMessage("Analysis Finished.")

    def _on_err(self, tb: str):
        self._prog.setVisible(False)
        self._load_btn.setEnabled(True)
        self._run_btn.setEnabled(bool(self._results.get("biopac_raw")))
        self.statusBar().showMessage("Error")
        QMessageBox.critical(self, "Error", tb[:3500])

    def _export(self):
        import os, re
        bp_path = self._params().get("biopac_path", "")
        fname = os.path.basename(bp_path)
        match = re.split(r'[-_](biopac|belt|carewear)', fname, flags=re.IGNORECASE)
        default_name = match[0] + "_hr" if len(match) > 1 else "hr_analysis"
        
        dest, _ = QFileDialog.getSaveFileName(self, "Export HR CSV", default_name + ".csv", "CSV (*.csv)")
        if not dest: return
        base = dest.replace(".csv", "")
        for key, lbl in [("biopac_analysis", "biopac"), ("belt_analysis", "belt")]:
            r = self._results.get(key)
            if r: export_hr_csv(r, f"{base}_{lbl}.csv", label=lbl)

    # ----------------------------------------------------------------------- #
    # PyqtGraph Plotting Functions
    # ----------------------------------------------------------------------- #

    def _plot_raw_signals(self):
        self._p_sig_bp.clear()
        self._p_sig_bl.clear()
        sync = self._results.get("sync")
        bp_r = self._results.get("biopac_raw")
        bl_r = self._results.get("belt_raw")

        if sync:
            t_bp = sync["biopac_time_clipped"]
            self._p_sig_bp.plot(t_bp, sync["biopac_ecg_clipped"], pen='#1565C0')
            self._p_sig_bl.plot(sync["belt_time_aligned"], sync["belt_ecg_aligned"], pen='#B71C1C')
        elif bp_r:
            self._p_sig_bp.plot(bp_r["time_s"], bp_r["ecg"], pen='#1565C0')
            if bl_r:
                self._p_sig_bl.plot(bl_r["time_s"], bl_r["ecg"], pen='#B71C1C')

    def _plot_preview(self):
        self._p_prev_orig.clear()
        self._p_prev_aug.clear()
        self._p_prev_clean.clear()

        bl_raw_orig = self._results.get("belt_raw_unaugmented")
        bl_raw = self._results.get("belt_raw")
        bl_an = self._results.get("belt_analysis")

        if bl_raw_orig is not None and bl_raw is not None:
            t_raw = np.arange(len(bl_raw_orig)) / bl_raw["fs"]
            self._p_prev_orig.plot(t_raw, bl_raw_orig, pen='#1565C0')
            self._p_prev_aug.plot(t_raw, bl_raw["ecg"], pen='#E65100')
            if bl_an and bl_an.get("ecg_clean") is not None:
                t_clean = np.arange(len(bl_an["ecg_clean"])) / bl_an["fs"]
                self._p_prev_clean.plot(t_clean, bl_an["ecg_clean"], pen='#2E7D32')
        self._p_prev_orig.autoRange()

    def _plot_interactive_pipeline(self):
        self._pi_bp_wave.clear()
        self._pi_bl_wave.clear()

        bp_r = self._results.get("biopac_analysis")
        bl_r = self._results.get("belt_analysis")

        if bp_r and bp_r.get("ecg_clean") is not None:
            ecg = bp_r["ecg_clean"]
            t = np.arange(len(ecg)) / bp_r["fs"]
            self._pi_bp_wave.plot(t, ecg, pen='#1565C0')
            if "full_signal" in bp_r and bp_r["full_signal"]:
                pks = bp_r["full_signal"]["peaks"]
                t_peaks = pks / bp_r["fs"]
                s = pg.ScatterPlotItem(t_peaks, ecg[pks], pen='r', brush='r', symbol='t', size=8)
                self._pi_bp_wave.addItem(s)

        if bl_r and bl_r.get("ecg_clean") is not None:
            ecg = bl_r["ecg_clean"]
            t = np.arange(len(ecg)) / bl_r["fs"]
            self._pi_bl_wave.plot(t, ecg, pen='#B71C1C')
            if "full_signal" in bl_r and bl_r["full_signal"]:
                pks = bl_r["full_signal"]["peaks"]
                t_peaks = pks / bl_r["fs"]
                s = pg.ScatterPlotItem(t_peaks, ecg[pks], pen='m', brush='m', symbol='t', size=8)
                self._pi_bl_wave.addItem(s)

        # Draw Heatmaps
        bp_scalo = self._results.get("biopac_scalo")
        if bp_scalo is not None:
            freqs = self._results.get("scalo_freqs")
            self._img_bp_sc.setImage(bp_scalo)
            dy = (freqs.max() - freqs.min()) / len(freqs)
            self._img_bp_sc.setRect(pg.QtCore.QRectF(0, freqs.min(), len(bp_r["ecg_clean"])/bp_r["fs"], freqs.max()-freqs.min()))
            self._img_bp_sc.setVisible(True)
        else:
            self._img_bp_sc.setVisible(False)

        bl_scalo = self._results.get("belt_scalo")
        if bl_scalo is not None:
            freqs = self._results.get("scalo_freqs")
            self._img_bl_sc.setImage(bl_scalo)
            self._img_bl_sc.setRect(pg.QtCore.QRectF(0, freqs.min(), len(bl_r["ecg_clean"])/bl_r["fs"], freqs.max()-freqs.min()))
            self._img_bl_sc.setVisible(True)
        else:
            self._img_bl_sc.setVisible(False)

        self._pi_bp_wave.autoRange()

    def _update_scalogram_levels(self, window, viewRange):
        """Rescale power of colorscale based on the current window and not based on data outside."""
        x_min, x_max = viewRange[0]
        fs = self._results.get("biopac_analysis", {}).get("fs", 125)

        for img_data, img_item in [(self._results.get("biopac_scalo"), self._img_bp_sc),
                                   (self._results.get("belt_scalo"), self._img_bl_sc)]:
            if img_data is not None:
                idx_min = max(0, int(x_min * fs))
                idx_max = min(img_data.shape[0], int(x_max * fs))
                if idx_max > idx_min:
                    seg = img_data[idx_min:idx_max, :]
                    vmin, vmax = np.min(seg), np.max(seg)
                    if vmax > vmin:
                        img_item.setLevels([vmin, vmax])

    def _plot_sqi(self):
        self._gw_sqi.clear()
        bp_r = self._results.get("biopac_analysis")
        bl_r = self._results.get("belt_analysis")
        if not bp_r: return
        
        p1 = self._gw_sqi.addPlot(title="BIOPAC QRS Energy SQI")
        p1.setLabels(bottom='Window Index', left='SQI Energy Ratio')
        wins = bp_r["windows"]
        x = np.array([w["window_idx"] for w in wins])
        y = np.array([w["qrs_energy_ratio"] for w in wins])
        bg1 = pg.BarGraphItem(x=x, height=y, width=0.8, brush='b')
        p1.addItem(bg1)

        if bl_r:
            self._gw_sqi.nextRow()
            p2 = self._gw_sqi.addPlot(title="Belt QRS Energy SQI")
            p2.setLabels(bottom='Window Index', left='SQI Energy Ratio')
            wins2 = bl_r["windows"]
            x2 = np.array([w["window_idx"] for w in wins2])
            y2 = np.array([w["qrs_energy_ratio"] for w in wins2])
            bg2 = pg.BarGraphItem(x=x2, height=y2, width=0.8, brush='r')
            p2.addItem(bg2)

    def _plot_hr(self):
        self._p_ihr.clear()
        self._p_whr.clear()
        bp_r = self._results.get("biopac_analysis")
        bl_r = self._results.get("belt_analysis")
        
        def plot_ihr(p, r, color, name):
            if not r or "full_signal" not in r: return
            f = r["full_signal"]
            if len(f.get("instantaneous_hr_times", [])) > 0:
                p.plot(f["instantaneous_hr_times"], f["instantaneous_hr_bpm"], pen=pg.mkPen(color, width=1, style=Qt.PenStyle.DashLine), name=name+" raw")
                p.plot(f["smoothed_hr_times"], f["smoothed_hr_bpm"], pen=pg.mkPen(color, width=3), name=name+" smooth")

        plot_ihr(self._p_ihr, bp_r, '#1565C0', 'BIOPAC')
        plot_ihr(self._p_ihr, bl_r, '#B71C1C', 'Belt')

        cmp = self._results.get("comparison")
        if cmp:
            valid = [(w.get("window_center_s", i), w.get("ref_hr_bpm"), w.get("test_hr_bpm")) for i, w in enumerate(cmp.get("window_details", [])) if w.get("ref_hr_bpm") is not None]
            if valid:
                x = np.array([w[0] for w in valid])
                ref = np.array([w[1] for w in valid])
                test = np.array([w[2] for w in valid])
                self._p_whr.plot(x, ref, pen=pg.mkPen('#1565C0', width=2), symbol='o', symbolBrush='#1565C0', symbolPen='#1565C0', name='BIOPAC HR')
                self._p_whr.plot(x, test, pen=pg.mkPen('#B71C1C', width=2), symbol='t', symbolBrush='#B71C1C', symbolPen='#B71C1C', name='Belt HR')

    def _plot_metrics(self):
        cmp = self._results.get("comparison")
        if not cmp: return
        beat = cmp.get("beat_level", {})
        hr   = cmp.get("hr_comparison", {})

        rows = [
            ("True Positives (TP)", str(beat.get("TP", 0))),
            ("False Positives (FP)", str(beat.get("FP", 0))),
            ("False Negatives (FN)", str(beat.get("FN", 0))),
            ("Sensitivity", f"{beat.get('Se', 0):.4f}"),
            ("Precision", f"{beat.get('PPV', 0):.4f}"),
            ("F1 Score", f"{beat.get('F1', 0):.4f}"),
            ("HR MAE", f"{hr.get('mae_bpm', 0):.2f} BPM"),
            ("HR RMSE", f"{hr.get('rmse_bpm', 0):.2f} BPM")
        ]
        self._mtbl.setRowCount(len(rows))
        for i, (m, v) in enumerate(rows):
            self._mtbl.setItem(i, 0, QTableWidgetItem(m))
            self._mtbl.setItem(i, 1, QTableWidgetItem(v))

        # Bland Altman
        self._p_ba.clear()
        valid = [(w["ref_hr_bpm"], w["test_hr_bpm"]) for w in cmp.get("window_details", []) if w.get("ref_hr_bpm", 0) > 0 and w.get("test_hr_bpm", 0) > 0]
        if valid:
            ref = np.array([r for r, _ in valid])
            test = np.array([t for _, t in valid])
            mean = (ref + test) / 2
            diff = test - ref
            s = pg.ScatterPlotItem(mean, diff, size=8, brush='#1565C0')
            self._p_ba.addItem(s)
            bias = float(np.mean(diff))
            loa = 1.96 * float(np.std(diff))
            self._p_ba.addLine(y=bias, pen=pg.mkPen('r', width=2))
            self._p_ba.addLine(y=bias+loa, pen=pg.mkPen('y', width=2, style=Qt.PenStyle.DashLine))
            self._p_ba.addLine(y=bias-loa, pen=pg.mkPen('y', width=2, style=Qt.PenStyle.DashLine))

            # Add Text Labels
            t1 = pg.TextItem(f"Bias: {bias:.2f}", color='r', anchor=(0,1))
            t1.setPos(min(mean), bias)
            self._p_ba.addItem(t1)
            t2 = pg.TextItem(f"+1.96 SD: {bias+loa:.2f}", color='b', anchor=(0,1))
            t2.setPos(min(mean), bias+loa)
            self._p_ba.addItem(t2)
            t3 = pg.TextItem(f"-1.96 SD: {bias-loa:.2f}", color='b', anchor=(0,0))
            t3.setPos(min(mean), bias-loa)
            self._p_ba.addItem(t3)

def _pick_file(parent, edit: QLineEdit, title: str):
    path, _ = QFileDialog.getOpenFileName(parent, title, str(_HERE))
    if path:
        edit.setText(path)

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("ECG Analysis")
    app.setStyle("Fusion")
    win = ECGAnalysisGUI()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
