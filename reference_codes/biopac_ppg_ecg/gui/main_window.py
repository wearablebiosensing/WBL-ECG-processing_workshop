import sys
import os
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QSplitter, 
                             QGroupBox, QFormLayout, QDoubleSpinBox, QMessageBox, 
                             QProgressDialog, QComboBox, QCheckBox, QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
                             QProgressBar, QApplication, QScrollArea, QMenu, QAction, QListWidget, QListWidgetItem, QDialog, QSpinBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import pyqtgraph as pg

from .widgets.plot_widget import PlotWidget
from .widgets.cluster_dialog import ClusterSelectionDialog
from ..pipeline.ingestion import DataIngestion
from ..pipeline.preprocess import Preprocessor
from ..pipeline.sqi import SQIAnalyzer
from ..pipeline.beat_detection import BeatDetector
from ..pipeline.features import FeatureExtractor
from ..utils.config_manager import ConfigManager

class LoadWorker(QThread):
    finished = pyqtSignal(object, object, object, str, object) # df, fs, analysis_dir, error_msg, raw_labels_df
    progress = pyqtSignal(int)
    status_msg = pyqtSignal(str) # Detailed message
    
    def __init__(self, ingestion, preprocessor, biopac_path, label_path, invert_ecg=False):
        super().__init__()
        self.ingestion = ingestion
        self.preprocessor = preprocessor
        self.biopac_path = biopac_path # Can be str or list of str
        self.label_path = label_path
        self.invert_ecg = invert_ecg
        
    def _ingest_callback(self, msg, pct):
        self.status_msg.emit(msg)
        self.progress.emit(int(pct * 50)) # First 50% is ingestion
        
    def run(self):
        try:
            self.status_msg.emit("Initializing...")
            print("DEBUG: LoadWorker Started.")
            
            # 1. Load Biopac (0-50%)
            raw_df = None
            fs_orig = 0.0
            
            # Handle List of Files (Stitching)
            if isinstance(self.biopac_path, list):
                 self.status_msg.emit(f"Stitching {len(self.biopac_path)} files...")
                 print(f"DEBUG: Stitching {len(self.biopac_path)} files: {self.biopac_path}")
                 # Assume they are ACQ because multi-select is usually for ACQ
                 raw_df, fs_orig, _ = self.ingestion.load_and_stitch_acq(self.biopac_path, progress_callback=self._ingest_callback)
                 print("DEBUG: Stitching Complete.")
                 
                 # Set primary name for analysis dir using the first file
                 clean_path = self.biopac_path[0].strip()
            else:
                clean_path = self.biopac_path.strip()
                fname = clean_path.lower()
                print(f"DEBUG: Loading file: '{clean_path}' (Ext detected: {fname.endswith('.acq')})")
                
                if fname.endswith('.acq'):
                    raw_df, fs_orig, _ = self.ingestion.load_acq(clean_path, progress_callback=self._ingest_callback)
                else:
                    raw_df, fs_orig, _ = self.ingestion.load_biopac(clean_path, progress_callback=self._ingest_callback)
            
            # 2. Load Labels (50-60%)
            if self.label_path:
                self.status_msg.emit("Loading Labels...")
                lbl_df = self.ingestion.load_labels_file(self.label_path)
                self.progress.emit(55)
                
                self.status_msg.emit("Merging Tags...")
                raw_df = self.ingestion.merge_biopac_labels(raw_df, lbl_df)
                self.progress.emit(60)
            
            # 3. Preprocess (60-95%)
            self.status_msg.emit(f"Resampling/Filtering... (Original FS: {fs_orig:.1f}Hz)")
            # Preprocessing doesn't have a callback yet, so we just set indeterminate or static
            # But we can split it if we want deeper insight. 
            # For now single step is fine, the user sees "Resampling..."
            processed_df, fs_new = self.preprocessor.process_dataframe(raw_df, fs_orig, invert_ecg=self.invert_ecg)
            self.progress.emit(90)
            
            # Analysis Dir
            self.status_msg.emit("Finalizing...")
            # Analysis Dir
            self.status_msg.emit("Finalizing...")
            # If list, strict parent dir of first file
            if isinstance(self.biopac_path, list):
                main_path = self.biopac_path[0]
            else:
                main_path = self.biopac_path
                
            parent_dir = os.path.dirname(main_path)
            pid = os.path.basename(parent_dir)
            if not pid or len(pid)<2: pid = os.path.splitext(os.path.basename(main_path))[0]
            # if isinstance(self.biopac_path, list): pid += "_Stitched"
            analysis_dir = os.path.join(parent_dir, f"{pid}_Analysis")
            
            self.progress.emit(100)
            self.status_msg.emit("Done.")
            
            # Debug Ranges
            if self.label_path and 'lbl_df' in locals():
                t_bio_start = processed_df['timestamp_ms'].iloc[0]
                t_lbl_start = lbl_df['timestamp_ms'].min()
                print(f"DEBUG: Biopac Start: {t_bio_start}, Label Start: {t_lbl_start}, Diff: {t_lbl_start - t_bio_start}ms")

            # Pass label_df separately for plotting
            lbl_out = lbl_df if (self.label_path and 'lbl_df' in locals()) else None
            self.finished.emit(processed_df, fs_new, analysis_dir, None, lbl_out)
            
        except Exception as e:
            self.finished.emit(None, None, None, str(e), None)

class DetectWorker(QThread):
    finished = pyqtSignal(object, object, object) # rpeaks, ppg_peaks, ppg_onsets
    progress = pyqtSignal(int)
    status_msg = pyqtSignal(str)
    error_msg = pyqtSignal(str)
    
    def __init__(self, detector, df, ecg_method, ppg_method):
        super().__init__()
        self.detector = detector
        self.df = df
        self.ecg_method = ecg_method
        self.ppg_method = ppg_method
        self._is_running = True
        
    def run(self):
        try:
            # 1. ECG
            self.status_msg.emit("Detecting ECG R-Peaks...")
            self.progress.emit(10)
            if not self._is_running: return
            
            # Note: NeuroKit ecg_peaks is not easily interruptible unless we chunk it,
            # but it is usually fast enough per chunk. Global run might take time.
            # Ideally we'd chunk, but for now just offloading to thread is huge win.
            rp, _ = self.detector.detect_ecg_beats(self.df['ECG'].values, self.ecg_method)
            self.progress.emit(50)
            
            # 2. PPG
            self.status_msg.emit("Detecting PPG Beats...")
            if not self._is_running: return
            
            p, o = self.detector.detect_ppg_beats(self.df['PPG'].values, self.ppg_method)
            self.progress.emit(90)
            
            # Safety
            if rp is None: rp = np.array([], dtype=int)
            if p is None: p = np.array([], dtype=int)
            if o is None: o = np.array([], dtype=int)
            
            # 3. Refine Onsets (if needed)
            # The detector.refine_onsets is fast, but we can do it here too to save UI thread.
            self.status_msg.emit("Refining Onsets...")
            o_refined = self.detector.refine_onsets(self.df['PPG'].values, o, p)
            self.progress.emit(100)
            
            self.finished.emit(rp, p, o_refined)
            
        except Exception as e:
            import traceback
            self.error_msg.emit(f"{e}\n{traceback.format_exc()}")
            
    def stop(self):
        self._is_running = False

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ECG/PPG Analyzer (Calibrate & Configure)")
        self.setGeometry(100, 100, 1600, 950)
        
        # Config
        self.config_manager = ConfigManager()
        self.config_manager.load_config()
        
        # Pipeline Objects
        self.ingestion = DataIngestion()
        self.preprocessor = Preprocessor(target_fs=250)
        self.sqi_analyzer = SQIAnalyzer(fs=250)
        self.beat_detector = BeatDetector(fs=250)
        self.feature_extractor = FeatureExtractor(fs=250)
        
        # Data State
        self.processed_data = None
        self.fs = 250
        self.window_size_sec = 10
        self.window_step_sec = 5 
        self.windows_results = []
        self.beat_results = {}
        self.analysis_dir = None
        
        self.biopac_path = None
        self.label_path = None
        
        # UI Mode
        self.laptop_mode = False
        
        self.init_ui()
        self.load_settings_from_config()
        
    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # --- Left Panel ---
        left_panel = QWidget()
        left_panel.setMaximumWidth(400) # Wider for table if needed
        control_layout = QVBoxLayout(left_panel)
        
        # Laptop Mode Toggle
        self.chk_laptop_mode = QCheckBox("💻 Laptop Mode (Compact UI)")
        self.chk_laptop_mode.setToolTip("Enable compact layout for smaller screens")
        self.chk_laptop_mode.setStyleSheet("font-weight: bold; padding: 5px;")
        self.chk_laptop_mode.clicked.connect(self.toggle_laptop_mode)
        control_layout.addWidget(self.chk_laptop_mode)
        
        # Store reference to left panel for width adjustment
        self.left_panel = left_panel
        
        tabs = QTabWidget()
        
        # Tab 1: Workflow
        tab_flow = QWidget()
        tab_flow_layout = QVBoxLayout(tab_flow)
        tab_flow_layout.setContentsMargins(0, 0, 0, 0)
        
        # Scroll Area for Controls
        scroll_controls = QScrollArea()
        scroll_controls.setWidgetResizable(True)
        # Remove border for cleaner look
        scroll_controls.setFrameShape(QScrollArea.NoFrame)
        
        scroll_content = QWidget()
        flow_layout = QVBoxLayout(scroll_content)
        flow_layout.setContentsMargins(5, 5, 5, 5)
        flow_layout.setSpacing(10)
        
        scroll_controls.setWidget(scroll_content)
        tab_flow_layout.addWidget(scroll_controls)
        
        # Import
        grp_import = QGroupBox("1. Ingestion")
        l_imp = QVBoxLayout()
        
        self.btn_biopac = QPushButton("1. Select BIOPAC File (.txt/.acq)")
        self.btn_biopac.clicked.connect(self.select_biopac)
        
        self.btn_label = QPushButton("2. Select Label File (.csv/xlsx)")
        self.btn_label.clicked.connect(self.select_label)
        
        self.btn_process = QPushButton("3. Process & Merge")
        self.btn_process.clicked.connect(self.start_processing)
        self.btn_process.setEnabled(False)
        
        self.lbl_status = QLabel("No files selected")
        self.lbl_status.setWordWrap(True)
        
        self.lbl_progress_details = QLabel("Ready") # TQDM-like status
        self.lbl_progress_details.setStyleSheet("font-weight: bold; color: #333;")
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        
        self.btn_save_merged = QPushButton("Save Merged Data")
        self.btn_save_merged.clicked.connect(self.save_merged_data)
        self.btn_save_merged.setEnabled(False)
        
        self.chk_invert = QCheckBox("Invert ECG Signal")
        self.chk_invert.setToolTip("Check this if your ECG polarity is reversed (R-peaks pointing down)")
        
        l_imp.addWidget(self.btn_biopac)
        l_imp.addWidget(self.btn_label)
        
        self.d_duration = QDoubleSpinBox()
        self.d_duration.setRange(0, 10) # Hours
        self.d_duration.setSingleStep(0.5)
        self.d_duration.setValue(3.0) # Default
        self.d_duration.setSuffix(" hr")
        self.d_duration.setToolTip("Limit analysis to first X hours (0 = All)")
        self.lbl_duration = QLabel("Duration Limit:")
        l_imp.addWidget(self.lbl_duration)
        l_imp.addWidget(self.d_duration)
        
        l_imp.addWidget(self.chk_invert)
        l_imp.addWidget(self.btn_process)
        l_imp.addWidget(self.lbl_progress_details)
        l_imp.addWidget(self.progress_bar)
        l_imp.addWidget(self.lbl_status)
        l_imp.addWidget(self.btn_save_merged)
        
        grp_import.setLayout(l_imp)
        flow_layout.addWidget(grp_import)
        
        # SQI
        grp_sqi = QGroupBox("2. SQI Calibration")
        l_sqi = QFormLayout()
        
        self.combo_sqi_method = QComboBox()
        self.combo_sqi_method.addItems(['Standard (Rule Based)', 'Template Matching', 'Clustering (ECG+PPG)'])
        l_sqi.addRow("Method:", self.combo_sqi_method)
        
        self.d_ecg_sqi = QDoubleSpinBox(); self.d_ecg_sqi.setRange(0, 1); self.d_ecg_sqi.setSingleStep(0.05)
        self.d_ppg_sqi = QDoubleSpinBox(); self.d_ppg_sqi.setRange(0, 1); self.d_ppg_sqi.setSingleStep(0.05)
        
        l_sqi.addRow("ECG Thresh:", self.d_ecg_sqi)
        l_sqi.addRow("PPG Thresh:", self.d_ppg_sqi)

        self.d_sqi_win = QDoubleSpinBox()
        self.d_sqi_win.setRange(1.0, 30.0) # Expanded range
        self.d_sqi_win.setSingleStep(0.5)
        self.d_sqi_win.setValue(3.0) 
        self.d_sqi_win.setSuffix(" s")
        l_sqi.addRow("Window Size:", self.d_sqi_win)

        self.d_n_clusters = QSpinBox()
        self.d_n_clusters.setRange(2, 10)
        self.d_n_clusters.setValue(4)
        self.d_n_clusters.setValue(4)
        l_sqi.addRow("Num Clusters:", self.d_n_clusters)
        
        self.combo_refine = QComboBox()
        self.combo_refine.addItems([
            "No Refinement",
            "Refine: Template Matching",
            "Refine: Rule-Based Thresholds"
        ])
        self.combo_refine.setToolTip("Select a method to filter 'Good' clusters further using the thresholds above.")
        l_sqi.addRow("Post-Cluster Refine:", self.combo_refine)
        
        self.btn_sqi = QPushButton("Run SQI")
        self.btn_sqi.clicked.connect(self.run_sqi)
        self.btn_sqi.setEnabled(False)
        l_sqi.addRow(self.btn_sqi)
        grp_sqi.setLayout(l_sqi)
        flow_layout.addWidget(grp_sqi)
        
        # Table for Metrics Inspection
        self.table_metrics = QTableWidget()
        self.table_metrics.setMinimumHeight(300) # Increased from default/small

        self.table_metrics.setColumnCount(4)
        self.table_metrics.setHorizontalHeaderLabels(["Start(s)", "ECG Score", "PPG Score", "Is Good"])
        self.table_metrics.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_metrics.cellClicked.connect(self.on_table_click)
        self.table_metrics.cellDoubleClicked.connect(self.on_table_double_click)
        self.lbl_window_inspector = QLabel("Window Inspector (Double-click 'Is Good' to toggle):")
        flow_layout.addWidget(self.lbl_window_inspector)
        flow_layout.addWidget(self.table_metrics)
        
        # Beat Detect
        grp_beats = QGroupBox("3. Detect & Export")
        l_beats = QFormLayout()
        self.combo_ecg = QComboBox(); self.combo_ecg.addItems(['neurokit', 'promac', 'wfdb', 'ecg2rr'])
        self.combo_ppg = QComboBox(); self.combo_ppg.addItems(['msptd', 'e2e'])
        l_beats.addRow("ECG:", self.combo_ecg)
        l_beats.addRow("PPG:", self.combo_ppg)
        
        self.btn_beats = QPushButton("Detect Beats")
        self.btn_beats.clicked.connect(self.detect_beats)
        self.btn_beats.setEnabled(False)
        l_beats.addRow(self.btn_beats)
        
        self.btn_exp = QPushButton("Export Analysis to Folder")
        self.btn_exp.clicked.connect(self.export_results)
        self.btn_exp.setEnabled(False)
        l_beats.addRow(self.btn_exp)
        
        grp_beats.setLayout(l_beats)
        flow_layout.addWidget(grp_beats)
        
        # Store references for laptop mode
        self.grp_import = grp_import
        self.grp_sqi = grp_sqi
        self.grp_beats = grp_beats
        
        tabs.addTab(tab_flow, "Analysis")
        
        # Tab 2: Event Navigation
        tab_events = QWidget()
        ev_layout = QVBoxLayout(tab_events)
        ev_layout.addWidget(QLabel("Click to Jump to Event:"))
        self.list_labels = QListWidget()
        self.list_labels.itemClicked.connect(self.on_label_click)
        ev_layout.addWidget(self.list_labels)
        tabs.addTab(tab_events, "Events")
        
        control_layout.addWidget(tabs)
        main_layout.addWidget(left_panel)
        
        # --- Right Panel ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Plot Controls
        plot_ctrl_layout = QHBoxLayout()
        
        self.btn_pan = QPushButton("Enable Pan Mode")
        self.btn_pan.setCheckable(True)
        self.btn_pan.clicked.connect(self.toggle_pan)

        self.combo_downsample = QComboBox()
        self.combo_downsample.addItems(["Full Res", "100 Hz", "50 Hz"])
        self.combo_downsample.currentIndexChanged.connect(self.update_downsampling)

        self.btn_measure = QPushButton("Measure Mode")
        self.btn_measure.setCheckable(True)
        self.btn_measure.clicked.connect(self.toggle_measure)
        
        plot_ctrl_layout.addWidget(QLabel("Plot Mode:"))
        plot_ctrl_layout.addWidget(self.btn_pan)
        plot_ctrl_layout.addWidget(self.btn_measure)
        plot_ctrl_layout.addStretch()
        plot_ctrl_layout.addWidget(QLabel("Visualization Rate:"))
        plot_ctrl_layout.addWidget(self.combo_downsample)
        
        right_layout.addLayout(plot_ctrl_layout)
        
        # Main Visualization Tabs (Signals vs Metrics)
        self.right_tabs = QTabWidget()
        
        # 1. Signals Tab
        self.tab_signals = QWidget()
        sig_layout = QVBoxLayout(self.tab_signals)
        splitter = QSplitter(Qt.Vertical)
        
        self.ecg_plot = PlotWidget()
        self.ecg_plot.set_title("ECG Signal")
        self.ecg_plot.plot_widget.setLabel('left', "Amplitude", units='mV')
        self.ecg_plot.plot_widget.setLabel('bottom', "Time", units='s')
        
        self.ppg_plot = PlotWidget()
        self.ppg_plot.set_title("PPG Signal")
        self.ppg_plot.plot_widget.setLabel('left', "Amplitude", units='a.u.')
        self.ppg_plot.plot_widget.setLabel('bottom', "Time", units='s')
        self.ppg_plot.plot_widget.setXLink(self.ecg_plot.plot_widget)
        
        splitter.addWidget(self.ecg_plot)
        splitter.addWidget(self.ppg_plot)
        sig_layout.addWidget(splitter)
        
        
        # 2. Metrics Tab (Scrollable)
        self.tab_metrics_widget = QWidget()
        self.tab_metrics_layout = QVBoxLayout(self.tab_metrics_widget)
        
        # --- Toolbar for Metrics Controls ---
        controls_layout = QHBoxLayout()
        
        # View Mode
        controls_layout.addWidget(QLabel("View Mode:"))
        self.combo_view_mode = QComboBox()
        self.combo_view_mode.addItems(["Multi-Row (All)", "Single Focus"])
        self.combo_view_mode.currentIndexChanged.connect(self.update_view_mode)
        controls_layout.addWidget(self.combo_view_mode)
        
        # Focus Selector (Visible only in Single Focus mode, handled by update_view_mode)
        self.combo_focus_metric = QComboBox()
        self.combo_focus_metric.setVisible(False)
        self.combo_focus_metric.currentIndexChanged.connect(self.update_focused_plot)
        controls_layout.addWidget(self.combo_focus_metric)

        # Theme (Background)
        controls_layout.addWidget(QLabel("Bg:"))
        self.combo_theme = QComboBox()
        self.combo_theme.addItems(["Black", "White"])
        self.combo_theme.currentIndexChanged.connect(self.update_plot_theme)
        controls_layout.addWidget(self.combo_theme)

        # Mean Line Color
        controls_layout.addWidget(QLabel("Mean Line:"))
        self.combo_mean_color = QComboBox()
        self.combo_mean_color.addItems(["White", "Black", "Red", "Blue", "Green", "Yellow"])
        self.combo_mean_color.currentIndexChanged.connect(self.update_plot_theme) # Reuse theme update
        controls_layout.addWidget(self.combo_mean_color)
        
        controls_layout.addStretch()
        
        # --- Metric Selector (Menu Button) ---
        self.btn_select_metrics = QPushButton("Select Visible Metrics")
        self.menu_metrics = QMenu(self)
        self.btn_select_metrics.setMenu(self.menu_metrics)
        controls_layout.addWidget(self.btn_select_metrics)
        
        self.tab_metrics_layout.addLayout(controls_layout)

        # Scroll Area
        self.scroll_metrics = QScrollArea()
        self.scroll_metrics.setWidgetResizable(True)
        self.metrics_container = QWidget()
        self.metrics_layout = QVBoxLayout(self.metrics_container)
        self.scroll_metrics.setWidget(self.metrics_container)
        
        self.tab_metrics_layout.addWidget(self.scroll_metrics)
        
        # Metrics Dictionary to manage visibility
        self.metric_plots = {} 
        self.metric_titles = {} # Map key -> title for dropdown
        
        def create_plot(key, title, y_label, units):
            p = pg.PlotWidget(title=title)
            # Styling: Large Labels
            label_style = {'color': '#EEE', 'font-size': '14pt'}
            p.setLabel('left', y_label, units=units, **label_style)
            p.setLabel('bottom', "Time", units='min', **label_style)
            p.getAxis('left').setPen(pg.mkPen(color='#EEE', width=1))
            p.getAxis('bottom').setPen(pg.mkPen(color='#EEE', width=1))
            
            # Title Style: HTML for size
            p.setTitle(f'<span style="font-size: 16pt">{title}</span>')

            p.showGrid(x=True, y=True, alpha=0.3)
            p.setMinimumHeight(300) # Slightly taller
            
            nonlocal self
            self.metrics_layout.addWidget(p)
            
            self.metric_plots[key] = p
            self.metric_titles[key] = title
            self.combo_focus_metric.addItem(title, userData=key)
            
            # Add to Menu (for Multi-Row visibility)
            action = QAction(title, self, checkable=True)
            action.setChecked(True)
            action.triggered.connect(lambda checked, k=key: self.toggle_metric(k, checked))
            self.menu_metrics.addAction(action)
            return p

        # 1. PAT Metrics
        self.plot_pat_f = create_plot('pat_f', "PAT (Foot)", "PAT", "ms")
        self.plot_pat_p = create_plot('pat_p', "PAT (Peak)", "PAT", "ms")
        self.plot_pat_var = create_plot('pat_var', "PAT Variability (SDNN)", "SDNN", "ms")
        
        # 2. HRV Metrics
        self.plot_rr = create_plot('rr', "RR Interval", "RR", "ms")
        self.plot_hrv = create_plot('sdnn', "HRV (SDNN)", "SDNN", "ms")
        self.plot_rmssd = create_plot('rmssd', "HRV (RMSSD)", "RMSSD", "ms")
        self.plot_lfhf = create_plot('lfhf', "HRV (LF/HF Ratio)", "Ratio", "")
        self.plot_tp = create_plot('tp', "HRV Total Power", "Power", "ms²")

        # 3. PPG Morphology
        self.plot_rise_t = create_plot('rise_t', "PPG Rise Time", "Time", "ms")
        self.plot_rise_a = create_plot('rise_a', "PPG Rise Amp", "Amp", "a.u.")
        self.plot_pw50 = create_plot('pw50', "PPG Pulse Width (PW50)", "Width", "ms")
        
        # Link X-Axes
        for p in self.metric_plots.values():
            if p != self.plot_pat_f:
                p.setXLink(self.plot_pat_f)
        
        self.metrics_layout.addStretch()
        
        self.right_tabs.addTab(self.tab_signals, "Time Series Signals")
        self.right_tabs.addTab(self.tab_metrics_widget, "Metrics & Trends")
        
        right_layout.addWidget(self.right_tabs)
        main_layout.addWidget(right_panel)

    def update_view_mode(self):
        mode = self.combo_view_mode.currentText()
        is_single = (mode == "Single Focus")
        
        self.combo_focus_metric.setVisible(is_single)
        self.btn_select_metrics.setVisible(not is_single) # Hide checkboxes in single mode
        
        if is_single:
            self.update_focused_plot()
        else:
            # Show all (respecting checkboxes? or just show all?)
            # Let's show all for now, or reset to checked state.
            # Simple approach: Show all that are checked in menu
            for action in self.menu_metrics.actions():
                key = [k for k,v in self.metric_titles.items() if v == action.text()][0]
                if action.isChecked():
                    self.metric_plots[key].setVisible(True)
                    self.metric_plots[key].setMinimumHeight(300)
            # Hide those unchecked
            for action in self.menu_metrics.actions():
                 if not action.isChecked():
                    key = [k for k,v in self.metric_titles.items() if v == action.text()][0]
                    self.metric_plots[key].setVisible(False)

    def update_focused_plot(self):
        # Only relevant in Single Focus mode
        if self.combo_view_mode.currentText() != "Single Focus": return
        
        target_idx = self.combo_focus_metric.currentIndex()
        target_key = self.combo_focus_metric.itemData(target_idx)
        
        for key, p in self.metric_plots.items():
            if key == target_key:
                p.setVisible(True)
                p.setMinimumHeight(600) # Make it big
            else:
                p.setVisible(False)

    def update_plot_theme(self):
        bg_color = 'k' if self.combo_theme.currentText() == "Black" else 'w'
        fg_color = 'w' if bg_color == 'k' else 'k'
        mean_color = self.combo_mean_color.currentText().lower()
        if mean_color == 'white' and bg_color == 'w': mean_color = 'k' # prevent invisible line
        
        # Update all plots
        for p in self.metric_plots.values():
            p.setBackground(bg_color)
            
            # Update Axis/Labels
            label_style = {'color': '#EEE' if bg_color=='k' else '#333', 'font-size': '14pt'}
            p.setLabel('left', p.getAxis('left').labelText, units=p.getAxis('left').labelUnits, **label_style)
            p.setLabel('bottom', "Time", units='min', **label_style)
            
            # Title
            p.setTitle(f'<span style="font-size: 16pt; color: {fg_color}">{p.plotItem.titleLabel.text}</span>')
            
            axis_pen = pg.mkPen(color=label_style['color'], width=1)
            p.getAxis('left').setPen(axis_pen)
            p.getAxis('bottom').setPen(axis_pen)
            p.getAxis('left').setTextPen(axis_pen)
            p.getAxis('bottom').setTextPen(axis_pen)
            
        self.current_mean_color = mean_color # Store for update_metrics_tab
        self.update_metrics_tab() # Refresh data/plots

    def toggle_metric(self, key, visible):
        if self.combo_view_mode.currentText() == "Single Focus": return # Ignore in single mode
        if key in self.metric_plots:
            self.metric_plots[key].setVisible(visible)

    def select_biopac(self):
        # Allow multi-select
        files, _ = QFileDialog.getOpenFileNames(self, "Select BIOPAC File(s)", "", "BIOPAC Files (*.txt *.acq);;All Files (*)")
        if files:
            if len(files) == 1:
                self.biopac_path = files[0]
                self.btn_biopac.setText(f"File: {os.path.basename(self.biopac_path)}")
            else:
                self.biopac_path = files # Store list
                self.btn_biopac.setText(f"{len(files)} files selected")
                
            self.check_ready()

    def select_label(self):
        fpath, _ = QFileDialog.getOpenFileName(self, "Load Labels", "", "Data Files (*.csv *.xlsx *.xls);;All (*)")
        if fpath:
            self.label_path = fpath
            current_txt = self.lbl_status.text()
            if "Labels:" not in current_txt:
                self.lbl_status.setText(f"{current_txt}\nLabels: {os.path.basename(fpath)}")
            else:
                 # Replace existing label line
                 lines = current_txt.split('\n')
                 lines = [l for l in lines if "Labels:" not in l]
                 lines.append(f"Labels: {os.path.basename(fpath)}")
                 self.lbl_status.setText('\n'.join(lines))
            self.check_ready()

    def check_ready(self):
        if self.biopac_path:
            self.btn_process.setEnabled(True)
            # self.lbl_status.setText("Ready to Process") # Don't overwrite detailed status
        else:
            self.btn_process.setEnabled(False)

    def start_processing(self):
        if not self.biopac_path: return
        self.btn_process.setEnabled(False)
        self.progress_bar.setValue(0)
        self.lbl_progress_details.setText("Starting...")
        
        invert = self.chk_invert.isChecked()
        self.worker = LoadWorker(self.ingestion, self.preprocessor, self.biopac_path, self.label_path, invert_ecg=invert)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.status_msg.connect(self.lbl_progress_details.setText)
        self.worker.finished.connect(self.on_load_finished)
        self.worker.start()

    def on_load_finished(self, df, fs, analysis_dir, error, raw_labels_df=None):
        self.btn_process.setEnabled(True)
        self.progress_bar.setValue(100)
        self.lbl_progress_details.setText("Completed.")
        
        if error:
            QMessageBox.critical(self, "Load Error", error)
            return
            
        # Duration Cropping
        hrs = self.d_duration.value()
        if hrs > 0 and df is not None:
             max_ms = hrs * 3600 * 1000
             start_ms = df['timestamp_ms'].iloc[0]
             df = df[df['timestamp_ms'] <= (start_ms + max_ms)].copy()
             self.lbl_status.setText(f"Loaded (Cropped to {hrs}h): {os.path.basename(analysis_dir)}")
        else:
             self.lbl_status.setText(f"Loaded: {os.path.basename(analysis_dir)}")
            
        self.processed_data = df
        self.fs = fs
        self.analysis_dir = analysis_dir
        
        t = (df['timestamp_ms'] - df['timestamp_ms'].iloc[0])/1000
        
        # Plot styling from config
        plot_conf = self.config_manager.get('plots', {})
        ecg_conf = plot_conf.get('ecg', {'color': '#0000FF', 'width': 1})
        ppg_conf = plot_conf.get('ppg', {'color': '#00FF00', 'width': 1})
        
        self.ecg_plot.plot_signal(t, df['ECG'], ecg_conf.get('color', '#0000FF'), width=ecg_conf.get('width', 1))
        self.ppg_plot.plot_signal(t, df['PPG'], ppg_conf.get('color', '#00FF00'), width=ppg_conf.get('width', 1))
        
        # Set default downsampling to '100 Hz' for responsiveness if high res
        if fs > 100:
            self.combo_downsample.setCurrentText("100 Hz")
        else:
            self.combo_downsample.setCurrentText("Full Res")
            
        # Plot labels if exist
        # Plot labels if exist (Use raw_labels_df if available)
        self.list_labels.clear() 
        lbl_points = pd.DataFrame()
        self.raw_labels_df = raw_labels_df # Save for exact marker plotting
        
        # Populate Events List
        self.list_labels.clear()
        
        # Use raw labels if available (most accurate), else merged from DF
        if raw_labels_df is not None:
            raw_labels_df = raw_labels_df.sort_values('timestamp_ms')
            t0 = df['timestamp_ms'].iloc[0]
            
            for _, row in raw_labels_df.iterrows():
                t_ms = row['timestamp_ms']
                if t_ms < t0: continue
                
                label = str(row['label'])
                t_sec = (t_ms - t0) / 1000.0
                
                item = QListWidgetItem(f"{t_sec:.1f}s: {label}")
                item.setData(Qt.UserRole, t_sec)
                self.list_labels.addItem(item)
                
                # Plot Vertical Line on Main Plot
                for plot_wrapper in [self.ecg_plot, self.ppg_plot]:
                    v_line = pg.InfiniteLine(
                        pos=t_sec, 
                        angle=90, 
                        pen=pg.mkPen('y', width=2, style=Qt.DashLine), 
                        label=label,
                        labelOpts={'color': 'y', 'position': 0.95, 'angle': -90, 'anchor': (1, 0.5)}
                    )
                    plot_wrapper.plot_widget.addItem(v_line)
                    
        elif 'label' in df.columns:
            # Fallback: Use merged labels in DF
            t0 = df['timestamp_ms'].iloc[0]
            # Find rows with non-empty labels
            lbl_series = df['label'].fillna('').astype(str).str.strip()
            lbl_points = df[lbl_series != ""]
            
            # Limit to prevent UI freeze if many
            if len(lbl_points) > 2000:
                self.lbl_progress_details.setText(f"Warning: Too many events ({len(lbl_points)}). Plotting first 500.")
                lbl_points = lbl_points.iloc[:500]
            
            for i, row in lbl_points.iterrows():
                label = str(row['label'])
                
                t_sec = (row['timestamp_ms'] - t0) / 1000.0
                item = QListWidgetItem(f"{t_sec:.1f}s: {label}")
                item.setData(Qt.UserRole, t_sec)
                self.list_labels.addItem(item)
                
                for plot_wrapper in [self.ecg_plot, self.ppg_plot]:
                    v_line = pg.InfiniteLine(
                        pos=t_sec, angle=90, pen=pg.mkPen('y', width=2, style=Qt.DashLine), label=label,
                        labelOpts={'color': 'y', 'position': 0.95, 'angle': -90, 'anchor': (1, 0.5)}
                    )
                    plot_wrapper.plot_widget.addItem(v_line)
                    
        self.btn_sqi.setEnabled(True)
        self.btn_save_merged.setEnabled(True)
        QMessageBox.information(self, "Success", f"Data loaded & merged.\nOutput Folder: {self.analysis_dir}")

    def load_settings_from_config(self):
        sqi_conf = self.config_manager.get('sqi', {})
        self.d_ecg_sqi.setValue(sqi_conf.get('ecg', {}).get('sqi_score_min', 0.8))
        self.d_ppg_sqi.setValue(sqi_conf.get('ppg', {}).get('sqi_score_min', 0.8))
        
        dets = self.config_manager.get('detectors', {})
        idx_e = self.combo_ecg.findText(dets.get('ecg_default', 'neurokit'))
        if idx_e >= 0: self.combo_ecg.setCurrentIndex(idx_e)
        idx_p = self.combo_ppg.findText(dets.get('ppg_default', 'msptd'))
        if idx_p >= 0: self.combo_ppg.setCurrentIndex(idx_p)

    def save_settings_to_config(self):
        self.config_manager.config['sqi']['ecg']['sqi_score_min'] = self.d_ecg_sqi.value()
        self.config_manager.config['sqi']['ppg']['sqi_score_min'] = self.d_ppg_sqi.value()
        self.config_manager.config['detectors']['ecg_default'] = self.combo_ecg.currentText()
        self.config_manager.config['detectors']['ppg_default'] = self.combo_ppg.currentText()
        self.config_manager.save_config()

    def toggle_pan(self, checked):
        self.ecg_plot.toggle_pan_mode(checked)
        self.ppg_plot.toggle_pan_mode(checked)
        self.btn_pan.setText("Disable Pan Mode" if checked else "Enable Pan Mode")
        
    def toggle_measure(self, checked):
        self.ecg_plot.enable_measure_mode(checked)
        self.ppg_plot.enable_measure_mode(checked)
        self.btn_measure.setText("Measure Mode On" if checked else "Measure Mode")
    
    def toggle_laptop_mode(self, checked):
        """Toggle between normal and laptop (compact) UI mode"""
        self.laptop_mode = checked
        
        if checked:
            # Laptop Mode: Compact layout
            self.left_panel.setMaximumWidth(300)
            self.left_panel.setMinimumWidth(280)
            
            # Make group boxes collapsible
            self.grp_import.setCheckable(True)
            self.grp_import.setChecked(True)
            self.grp_sqi.setCheckable(True)
            self.grp_sqi.setChecked(False)  # Start collapsed
            self.grp_beats.setCheckable(True)
            self.grp_beats.setChecked(False)  # Start collapsed
            
            # Hide less frequently used widgets
            self.lbl_duration.setVisible(False)
            self.d_duration.setVisible(False)
            self.chk_invert.setVisible(False)
            
            # Reduce table height
            self.table_metrics.setMaximumHeight(150)
            
            # Compact spacing
            self.left_panel.layout().setSpacing(5)
            self.left_panel.layout().setContentsMargins(5, 5, 5, 5)
            
        else:
            # Normal Mode: Full layout
            self.left_panel.setMaximumWidth(400)
            self.left_panel.setMinimumWidth(350)
            
            # Remove collapsibility
            self.grp_import.setCheckable(False)
            self.grp_sqi.setCheckable(False)
            self.grp_beats.setCheckable(False)
            
            # Show all widgets
            self.lbl_duration.setVisible(True)
            self.d_duration.setVisible(True)
            self.chk_invert.setVisible(True)
            
            # Reset table height
            self.table_metrics.setMaximumHeight(16777215)  # Qt max
            
            # Normal spacing
            self.left_panel.layout().setSpacing(10)
            self.left_panel.layout().setContentsMargins(10, 10, 10, 10)
        
        
    def update_downsampling(self):
        txt = self.combo_downsample.currentText()
        if txt == "Full Res":
            factor = 1
        elif txt == "100 Hz":
            factor = max(1, self.fs // 100)
        elif txt == "50 Hz":
            factor = max(1, self.fs // 50)
            
        self.ecg_plot.set_downsample_factor(factor)
        self.ppg_plot.set_downsample_factor(factor)

    def save_merged_data(self):
        if self.processed_data is None or self.analysis_dir is None: return
        try:
            if not os.path.exists(self.analysis_dir): os.makedirs(self.analysis_dir)
            out_path = os.path.join(self.analysis_dir, "merged_data.parquet")
            self.processed_data.to_parquet(out_path)
            QMessageBox.information(self, "Saved", f"Merged data saved to:\n{out_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save: {e}")

    def run_sqi(self):
        self.save_settings_to_config()
        
        # Prepare Custom Config for real-time update
        custom_config = self.config_manager.config['sqi'].copy()
        custom_config['ecg']['sqi_score_min'] = self.d_ecg_sqi.value()
        custom_config['ppg']['sqi_score_min'] = self.d_ppg_sqi.value()
        
        # Determine Method
        method_txt = self.combo_sqi_method.currentText()
        is_clustering = 'Clustering' in method_txt
        method = 'template' if 'Template' in method_txt else 'rule_based'
        
        df = self.processed_data
        n = len(df)
        
        # Window Settings (Apply generic win size to all methods as requested)
        win_sec = self.d_sqi_win.value()
        step_sec = win_sec # Non-overlapping step for simplicity, or we could add step control? 
        # User asked for "window size manually", usually implies step matches or is 50%.
        # For now, let's keep step = window for cleanest "tiles", but maybe 50% overlap for rule based?
        # Let's stick to step = window for Clustering, but maybe allow overlap for others?
        # Re-reading: "keep it configurable... default 3/5 seconds"
        
        win = int(win_sec * self.fs)
        if is_clustering:
             step = win # Non-overlapping for clustering
        else:
             # Standard SQI often uses sliding window. 
             # Let's use 50% overlap or user defined? 
             # Existing code had window_step_sec = 5s (half of 10).
             # Let's set step = win / 2 for standard/template
             step = int(max(1, win // 2))
        
        self.windows_results = []
        bad = []
        
        num_steps = (n - win) // step
        dlg = QProgressDialog(f"Running SQI ({method_txt})...", "Cancel", 0, num_steps, self)
        dlg.setWindowModality(Qt.WindowModal)
        
        t = df['timestamp_ms'].values
        ecg = df['ECG'].values
        ppg = df['PPG'].values
        
        # 1. Feature Extraction Loop
        update_interval = max(1, num_steps // 100)
        
        for i, idx in enumerate(range(0, n-win, step)):
            if dlg.wasCanceled(): break
            if i % update_interval == 0: dlg.setValue(i)
            
            end = idx + win
            
            # Pass custom config explicitly
            res = self.sqi_analyzer.analyze_window(
                ecg[idx:end], ppg[idx:end], 
                method=method, 
                custom_config=custom_config
            )
            
            # --- Check Sensor Contact (Worn Status) ---
            # ECG Threshold: 5 uV = 0.005 mV
            # PPG Threshold: 500 uV = 0.5 (assuming a.u./mV scale matches)
            # Calculate Peak-to-Peak
            ecg_chunk = ecg[idx:end]
            ppg_chunk = ppg[idx:end]
            
            ecg_amp = np.ptp(ecg_chunk) if len(ecg_chunk) > 0 else 0
            ppg_amp = np.ptp(ppg_chunk) if len(ppg_chunk) > 0 else 0
            
            is_ecg_worn = ecg_amp >= 0.005
            is_ppg_worn = ppg_amp >= 0.5
            is_worn = is_ecg_worn and is_ppg_worn
            
            res.update({
                'start_idx': idx, 
                'end_idx': end, 
                'start_ts': t[idx],
                'is_worn': is_worn,
                'ecg_amp': ecg_amp,
                'ppg_amp': ppg_amp
            })
            
            # If not worn, it shouldn't be "good" regardless of SQI
            if not is_worn:
                res['is_good'] = False
                
            self.windows_results.append(res)
            
            # If standard method, collect bad windows immediately
            if not is_clustering:
                if not res['is_good']:
                    bad.append( ((t[idx]-t[0])/1000, (t[end-1]-t[0])/1000) )
                
        dlg.close()
        
        # 2. Clustering Step
        n_clust = self.d_n_clusters.value()
        if is_clustering and len(self.windows_results) > n_clust:
            cluster_res = self.sqi_analyzer.cluster_windows(self.windows_results, n_clusters=n_clust)
            if cluster_res:
                # Show Dialog
                # Pass both ECG and PPG for inspection
                c_dlg = ClusterSelectionDialog(cluster_res, self.windows_results, ecg_data=ecg, ppg_data=ppg, parent=self)
                
                if c_dlg.exec_() == QDialog.Accepted:
                    good_clusters = c_dlg.get_selected()
                    
                    # Update is_good based on selection
                    labels = cluster_res['labels']
                    bad = []
                    
                    for i, res in enumerate(self.windows_results):
                        cluster_idx = labels[i]
                        w_good = (int(cluster_idx) in good_clusters) 
                        
                        # IMPORTANT: If sensors are not worn, it CANNOT be good, 
                        # even if it fell into a "good" cluster.
                        if not res.get('is_worn', True):
                            w_good = False
                        
                        res['cluster'] = cluster_idx
                        res['is_good'] = w_good
                        
                        if not w_good:
                            t_start = (res['start_ts'] - df['timestamp_ms'].iloc[0])/1000
                            # Use actual window length based on override
                            t_end = t_start + win_sec
                            bad.append((t_start, t_end))
                        
                    print(f"DEBUG: Clustering Applied. Good Clusters: {good_clusters}. Bad Windows Count: {len(bad)}")
                    
                    # --- Post-Clustering Refinement (Phase 28) ---
                    refine_mode = self.combo_refine.currentText()
                    
                    if "No Refinement" not in refine_mode:
                        print(f"DEBUG: Running Refinement Strategy: {refine_mode}")
                        
                        num_refine = len(self.windows_results)
                        update_int_ref = max(1, num_refine // 100)
                        
                        dlg_refine = QProgressDialog(f"Refining ({refine_mode})...", "Cancel", 0, num_refine, self)
                        dlg_refine.setWindowModality(Qt.WindowModal)
                        
                        refined_bad = 0
                        
                        enc_thresh = self.d_ecg_sqi.value()
                        ppg_thresh = self.d_ppg_sqi.value()
                        
                        is_template = "Template" in refine_mode

                        for i, res in enumerate(self.windows_results):
                            if dlg_refine.wasCanceled(): break
                            if i % update_int_ref == 0: dlg_refine.setValue(i)
                            
                            # Only refine windows that are currently "Good"
                            if res['is_good']:
                                idx, end = res['start_idx'], res['end_idx']
                                
                                pass_refine = True
                                
                                if is_template:
                                    # Template Matching (Expensive)
                                    tmpl_ecg = self.sqi_analyzer.compute_template_sqi(ecg[idx:end], 'ecg')
                                    tmpl_ppg = self.sqi_analyzer.compute_template_sqi(ppg[idx:end], 'ppg')
                                    if tmpl_ecg < enc_thresh or tmpl_ppg < ppg_thresh:
                                        pass_refine = False
                                else:
                                    # Rule-Based Thresholds (Fast)
                                    # Check existing 'ecg_sqi_score' and 'ppg_sqi_score'
                                    if res['ecg_sqi_score'] < enc_thresh or res['ppg_sqi_score'] < ppg_thresh:
                                        pass_refine = False
                                
                                # Remove if failed refinement
                                if not pass_refine:
                                    res['is_good'] = False
                                    res['refine_fail'] = True
                                    t_start = (res['start_ts'] - df['timestamp_ms'].iloc[0])/1000
                                    t_end = t_start + win_sec
                                    bad.append((t_start, t_end))
                                    refined_bad += 1
                                    
                        dlg_refine.close()
                        print(f"DEBUG: Refinement Complete. Removed {refined_bad} additional windows.")
                        QMessageBox.information(self, "Clustering Applied", f"Selected Good Clusters: {good_clusters}\nMarked {len(bad) - refined_bad} bad windows.\nRefinement ({refine_mode}) removed {refined_bad} additional windows.")
                    else:
                        print("DEBUG: No Refinement selected.")
                        QMessageBox.information(self, "Clustering Applied", f"Selected Good Clusters: {good_clusters}\nMarked {len(bad)} bad windows.")
                else:
                    QMessageBox.warning(self, "Cancelled", "Clustering selection cancelled. Retaining original rule-based scores.")
                    # Re-calc bad windows based on rule_based
                    bad = []
                    for res in self.windows_results:
                        if not res['is_good']:
                            t_start = (res['start_ts'] - df['timestamp_ms'].iloc[0])/1000
                            t_end = t_start + win_sec
                            bad.append((t_start, t_end))
            else:
                 QMessageBox.warning(self, "Error", f"Not enough data for clustering (need > {n_clust} windows).")

        self.ecg_plot.mark_bad_windows(bad)
        self.ppg_plot.mark_bad_windows(bad)
        self.populate_metrics_table()
        self.btn_beats.setEnabled(True)

        
    def populate_metrics_table(self):
        self.table_metrics.setRowCount(0)
        self.table_metrics.setRowCount(len(self.windows_results))
        t0 = self.processed_data['timestamp_ms'].iloc[0]
        
        for i, r in enumerate(self.windows_results):
            time_s = (r['start_ts'] - t0) / 1000.0
            item_t = QTableWidgetItem(f"{time_s:.1f}")
            item_e = QTableWidgetItem(f"{r['ecg_sqi_score']:.2f}")
            item_p = QTableWidgetItem(f"{r['ppg_sqi_score']:.2f}")
            item_g = QTableWidgetItem("YES" if r['is_good'] else "NO")
            if not r['is_good']:
                for it in [item_t, item_e, item_p, item_g]:
                    it.setBackground(Qt.red)     
            self.table_metrics.setItem(i, 0, item_t)
            self.table_metrics.setItem(i, 1, item_e)
            self.table_metrics.setItem(i, 2, item_p)
            self.table_metrics.setItem(i, 3, item_g)

    def on_table_click(self, row, col):
        if row < len(self.windows_results):
             r = self.windows_results[row]
             t0 = self.processed_data['timestamp_ms'].iloc[0]
             start_s = (r['start_ts'] - t0) / 1000.0
             end_s = start_s + self.window_size_sec
             self.ecg_plot.plot_widget.setXRange(start_s, end_s)

    def on_table_double_click(self, row, col):
        # 3 is "Is Good" column (0:Start, 1:ECG, 2:PPG, 3:Is Good)
        if col == 3 and row < len(self.windows_results):
            r = self.windows_results[row]
            # Toggle
            r['is_good'] = not r['is_good']
            
            # Recalculate bad list for plotting
            t0 = self.processed_data['timestamp_ms'].iloc[0]
            bad = []
            for res in self.windows_results:
                if not res['is_good']:
                    start_s = (res['start_ts']-t0)/1000
                    end_s = (self.processed_data['timestamp_ms'].iloc[res['end_idx']-1]-t0)/1000
                    bad.append((start_s, end_s))
            
            self.ecg_plot.mark_bad_windows(bad)
            self.ppg_plot.mark_bad_windows(bad)
            
            # Update row color
            item_g = self.table_metrics.item(row, 3)
            item_g.setText("YES" if r['is_good'] else "NO")
            
            color = Qt.white if r['is_good'] else Qt.red
            for c in range(4):
                it = self.table_metrics.item(row, c)
                it.setBackground(color)

    def on_label_click(self, item):
        t_sec = item.data(Qt.UserRole)
        if t_sec is not None:
             # Zoom to window around event
             window = 30.0 # 30 seconds
             self.ecg_plot.plot_widget.setXRange(t_sec - window/2, t_sec + window/2)
             # PPG linked so it updates too

    def detect_beats(self):
        if self.processed_data is None: return
        self.save_settings_to_config()
        
        # Disable UI
        self.btn_beats.setEnabled(False)
        self.btn_exp.setEnabled(False)
        
        # Setup Progress
        self.progress_bar.setValue(0)
        self.lbl_progress_details.setText("Starting Detection...")
        
        # Create Worker
        self.detect_worker = DetectWorker(
            self.beat_detector, 
            self.processed_data, 
            self.combo_ecg.currentText(), 
            self.combo_ppg.currentText()
        )
        
        self.detect_worker.status_msg.connect(self.lbl_progress_details.setText)
        self.detect_worker.progress.connect(self.progress_bar.setValue)
        self.detect_worker.error_msg.connect(self.on_detection_error)
        self.detect_worker.finished.connect(self.on_detection_finished)
        
        self.detect_worker.start()

    def on_detection_error(self, msg):
        import logging
        logging.getLogger(__name__).error(f"Detection Error: {msg}")
        QMessageBox.critical(self, "Detection Error", f"An error occurred during detection:\n{msg}")
        self.lbl_progress_details.setText("Failed")
        self.btn_beats.setEnabled(True)

    def on_detection_finished(self, rp, p, refined_onsets):
        try:
            self.lbl_progress_details.setText("Visualizing...")
            QApplication.processEvents()
            
            self.beat_results = {
                'rpeaks': rp, 
                'ppg_peaks': p, 
                'ppg_onsets': refined_onsets
            }
            
            display_df = self.processed_data
            t = (display_df['timestamp_ms'] - display_df['timestamp_ms'].iloc[0]).values / 1000.0
            
            # Plot (This must be on main thread, which we are)
            self.ecg_plot.plot_beats(t, display_df['ECG'].values, rp)
            self.ppg_plot.plot_beats(t, display_df['PPG'].values, p)
            self.ppg_plot.plot_onsets(t, display_df['PPG'].values, refined_onsets)
            
            self.btn_exp.setEnabled(True)
            self.btn_beats.setEnabled(True)
            
            # Update Metrics Tab
            self.update_metrics_tab()
            
            self.progress_bar.setValue(100)
            self.lbl_progress_details.setText("Completed")
            QMessageBox.information(self, "Done", f"Found {len(rp)} R-peaks, {len(p)} PPG peaks")
            
        except Exception as e:
            self.on_detection_error(str(e))
            


    def update_metrics_tab(self):
        if not self.beat_results: return
        
        # 1. Prepare Data
        rpeaks = self.beat_results['rpeaks']
        ppg_peaks = self.beat_results['ppg_peaks']
        ppg_onsets = self.beat_results['ppg_onsets']
        timestamps = self.processed_data['timestamp_ms'].values
        
        # 2. Filter input beats based on SQI Windows ("Good" only)
        # This prevents "Bad" windows from polluting rolling stats (HRV, SDNN).
        if self.windows_results:
             valid_ranges = []
             for w in self.windows_results:
                 if w['is_good']:
                     valid_ranges.append((w['start_ts'], w['start_ts'] + self.window_size_sec*1000))
             
             # Create masks for indices
             def filter_indices(indices, timestamps, ranges):
                 if len(indices) == 0: return indices
                 t_vals = timestamps[indices]
                 mask = np.zeros(len(indices), dtype=bool)
                 for start, end in ranges:
                     mask |= (t_vals >= start) & (t_vals < end)
                 return indices[mask]

             rpeaks = filter_indices(rpeaks, timestamps, valid_ranges)
             ppg_peaks = filter_indices(ppg_peaks, timestamps, valid_ranges)
             ppg_onsets = filter_indices(ppg_onsets, timestamps, valid_ranges)

        # 3. Extract Features (on filtered data)
        feats = self.feature_extractor.extract_metrics(
             rpeaks, ppg_peaks, ppg_onsets,
             self.processed_data['PPG'].values, timestamps
        )
        
        if feats.empty: return
        
        t0 = self.processed_data['timestamp_ms'].iloc[0]
        t_sec = (feats['timestamp_ms'] - t0) / 1000.0
        t_min = t_sec / 60.0
        
        # Helper to plot with rolling mean
        def plot_series(plot_widget, x, y, color, symbol='o', title_suffix=""):
            plot_widget.clear()
            # Scatter
            plot_widget.plot(x, y, pen=None, symbol=symbol, symbolSize=5, symbolBrush=color)
            
            # Rolling Mean (1 minute window)
            # x is in minutes.
            if len(x) > 10:
                # Create a Series for rolling
                s = pd.Series(y, index=x)
                # Rolling window of 1 minute (x is in minutes)
                # But 'index' is floats, rolling on float index is tricky in older pandas
                # simpler: use window size in samples approx?
                # or use time-aware rolling if converted to datetime?
                # Simplest: use numeric window of N samples for robustness: window=50
                r_mean = pd.Series(y).rolling(window=50, center=True, min_periods=5).mean()
                
                # Plot line
                mean_color = getattr(self, 'current_mean_color', 'w')
                plot_widget.plot(x, r_mean.values, pen=pg.mkPen(mean_color, width=2))
            
            plot_widget.setLabel('bottom', "Time", units='min')

        plot_series(self.plot_pat_f, t_min.values, feats['pat_f_ms'].values, 'b', symbol='o')
        plot_series(self.plot_pat_p, t_min.values, feats['pat_p_ms'].values, 'c', symbol='o')
        plot_series(self.plot_pat_var, t_min.values, feats['pat_f_sdnn'].values, 'm', symbol='x')
        plot_series(self.plot_rr, t_min.values, feats['rr_interval_ms'].values, 'y', symbol='o')
        plot_series(self.plot_hrv, t_min.values, feats['hrv_sdnn'].values, 'r', symbol='d')
        plot_series(self.plot_rmssd, t_min.values, feats['hrv_rmssd'].values, 'r', symbol='+')
        if 'hrv_lfhf' in feats.columns:
            plot_series(self.plot_lfhf, t_min.values, feats['hrv_lfhf'].values, 'g', symbol='t')
        if 'hrv_tp' in feats.columns:
            plot_series(self.plot_tp, t_min.values, feats['hrv_tp'].values, 'g', symbol='s')
        
        plot_series(self.plot_rise_t, t_min.values, feats['ppg_rise_time_ms'].values, 'g', symbol='s')
        plot_series(self.plot_rise_a, t_min.values, feats['ppg_rise_amp'].values, 'w', symbol='t')
        if 'ppg_pw50_ms' in feats.columns:
            plot_series(self.plot_pw50, t_min.values, feats['ppg_pw50_ms'].values, 'c', symbol='o')
        
        # Medication Markers Logic (Updated Phase 25)
        # 1. Try to find exact markers from raw_labels_df (Source of Truth)
        # 2. If not available, check merged data
        # 3. Fallback to 28 mins
        
        med_times_min = []
        
        # Strategy 1: Raw Labels
        if hasattr(self, 'raw_labels_df') and self.raw_labels_df is not None:
            mask = self.raw_labels_df['label'].fillna('').astype(str).str.contains("Medication Intake|Drug", case=False)
            found_rows = self.raw_labels_df[mask]
            
            if not found_rows.empty:
                for _, row in found_rows.iterrows():
                    t_ms = row['timestamp_ms']
                    t_sec = (t_ms - t0) / 1000.0
                    t_min_val = t_sec / 60.0
                    
                    # Sanity check? User said "if timing is way off... like over few hours"
                    # Let's assume trusting the label is better unless it's negative or huge
                    if t_min_val > 0 and t_min_val < (len(timestamps)/self.fs)/60 + 60: # Allow 1 hour buffer?
                        med_times_min.append(t_min_val)
        
        # Strategy 2: Merged Data (Fallback if raw not saved for some reason)
        if not med_times_min and 'label' in self.processed_data.columns:
            med_mask = self.processed_data['label'].fillna('').astype(str).str.contains("Medication Intake|Drug", case=False)
            med_indices = np.where(med_mask)[0]
            if len(med_indices) > 0:
                 # Find unique consecutive blocks? Or just take first? 
                 # Usually merged labels might spam multiple samples.
                 # Taking first is safer than taking all 1000 samples.
                 t_ms = self.processed_data['timestamp_ms'].iloc[med_indices[0]]
                 med_times_min.append( ((t_ms - t0) / 1000.0) / 60.0 )

        # Strategy 3: Fallback
        if not med_times_min:
             med_times_min.append(28.0) # Default 28 minutes
             
        # Plot Markers
        for t_min_val in med_times_min:
            for p in self.metric_plots.values():
                # Vertical Line
                line = pg.InfiniteLine(
                    pos=t_min_val, 
                    angle=90, 
                    pen=pg.mkPen('r', width=3, style=Qt.DashLine), # Red, thicker
                    label="Medication", 
                    labelOpts={'color': 'r', 'position': 0.1, 'angle': -90}
                )
                p.addItem(line)

    def calculate_yield_stats(self):
        """
        Calculate data yield statistics from windows results.
        Refined to exclude "Not Worn" periods (Low Amplitude).
        """
        if not self.windows_results:
            return None
            
        total_windows = len(self.windows_results)
        win_sec = self.window_size_sec
        
        # Count Worn Windows
        worn_windows = sum(1 for w in self.windows_results if w.get('is_worn', True)) # Default True if key missing (old analyses)
        
        # Good windows (must also be worn, but our logic ensures is_good=False if not worn)
        # DEFENSIVE: Explicitly check is_worn here too to prevent > 100% bugs if logic elsewhere fails.
        good_windows = sum(1 for w in self.windows_results if w.get('is_good', False) and w.get('is_worn', True))
        
        total_sec = total_windows * win_sec
        worn_sec = worn_windows * win_sec
        not_worn_sec = total_sec - worn_sec
        good_sec = good_windows * win_sec
        
        # Yield based on WORN duration
        yield_worn_pct = (good_windows / worn_windows * 100) if worn_windows > 0 else 0.0
        yield_total_pct = (good_windows / total_windows * 100) if total_windows > 0 else 0.0
        
        return {
            'total_windows': total_windows,
            'worn_windows': worn_windows,
            'good_windows': good_windows,
            'total_sec': total_sec,
            'worn_sec': worn_sec,
            'not_worn_sec': not_worn_sec,
            'good_sec': good_sec,
            'yield_worn_pct': yield_worn_pct,
            'yield_total_pct': yield_total_pct
        }

    def export_results(self):
        if not self.beat_results: return
        try:
            if not os.path.exists(self.analysis_dir): os.makedirs(self.analysis_dir)
            
            # Re-extract
            feats = self.feature_extractor.extract_metrics(
                 self.beat_results['rpeaks'], self.beat_results['ppg_peaks'], self.beat_results['ppg_onsets'],
                 self.processed_data['PPG'].values, self.processed_data['timestamp_ms'].values
            )
            
            # --- Add Labels to Beat Metrics (Phase 29) ---
            # Try to use raw_labels_df for precision, or fallback to processed_data
            lbl_source = None
            if hasattr(self, 'raw_labels_df') and self.raw_labels_df is not None and not self.raw_labels_df.empty:
                lbl_source = self.raw_labels_df[['timestamp_ms', 'label']].sort_values('timestamp_ms')
            elif 'label' in self.processed_data.columns:
                # Extract non-null labels
                temp = self.processed_data[['timestamp_ms', 'label']].dropna()
                if not temp.empty:
                    lbl_source = temp.sort_values('timestamp_ms')
            
            # 1. Initialize Columns
            feats['medication_label'] = ""
            feats['medication_sqi'] = "" # New Column
            feats['COWS_label'] = ""
            feats['COWS_sqi'] = "" # New Column
            
            if lbl_source is not None:
                t_beats = feats['timestamp_ms'].values
                if len(t_beats) > 0:
                     print(f"DEBUG: Export Beats Range: {t_beats[0]}ms to {t_beats[-1]}ms. Total: {len(t_beats)}")
                
                # --- Medication Time Logic (New) ---
                # 1. Extract Medication Timestamps
                med_mask = lbl_source['label'].fillna('').astype(str).str.contains("Medication Intake|Drug", case=False)
                med_rows = lbl_source[med_mask]
                med_times = sorted(med_rows['timestamp_ms'].unique())
                
                print(f"DEBUG: Found {len(med_times)} Medication Doses at: {med_times}")
                
                # 2. Calculate Time Since Last Dose
                # Logic: 
                #   - If t >= med_time: Delta = (t - med_time) (Positive)
                #   - If multiple meds, use the most recent one (resetting).
                #   - If t < first_med_time: Delta = (t - first_med_time) (Negative)
                
                t_since_med = np.full(len(feats), np.nan)
                med_ref_t = np.full(len(feats), np.nan)
                
                if med_times:
                    # Case 1: Pre-Medication (Before the first dose)
                    first_dose = med_times[0]
                    # Mask for beats before first dose
                    mask_pre = t_beats < first_dose
                    
                    # For pre-med, ref is first_dose. Delta is negative.
                    t_since_med[mask_pre] = (t_beats[mask_pre] - first_dose) / 1000.0
                    med_ref_t[mask_pre] = first_dose
                    
                    # Case 2: Post-Medication (After or at first dose)
                    # We iterate through doses to handle resets
                    # We can use np.searchsorted to find the injection index for each beat
                    
                    # Find indices where elements should be inserted to maintain order
                    # For a beat t, idx_r is the index of the first med_time > t.
                    # So the med_time <= t is at idx_r - 1.
                    idx_r = np.searchsorted(med_times, t_beats, side='right')
                    
                    # Filter for beats that are actually after at least one dose
                    mask_post = idx_r > 0
                    
                    if mask_post.any():
                        # Get the index of the med time to use (idx_r - 1)
                        # We only care about beats where t >= first_dose (which matches mask_post essentially)
                        
                        # Indices into med_times array
                        dose_indices = idx_r[mask_post] - 1
                        
                        # Get the actual med timestamps
                        # Use np.array for indexing
                        med_times_arr = np.array(med_times)
                        effective_doses = med_times_arr[dose_indices]
                        
                        # Calculate
                        t_current = t_beats[mask_post]
                        delta_sec = (t_current - effective_doses) / 1000.0
                        
                        t_since_med[mask_post] = delta_sec
                        med_ref_t[mask_post] = effective_doses
                        
                feats['time_since_medication'] = t_since_med
                feats['medication_ref_time'] = med_ref_t
                
                event_logs = []
                
                # 2. Iterate and Assign
                for _, row in lbl_source.iterrows():
                    t_lbl = row['timestamp_ms']
                    raw_txt = str(row['label']).strip()
                    l_lower = raw_txt.lower()
                    
                    target_col = None
                    target_sqi_col = None
                    clean_val = None
                    
                    # Determine Type
                    import re 
                    if "medication" in l_lower or "drug" in l_lower:
                        target_col = 'medication_label'
                        target_sqi_col = 'medication_sqi'
                        clean_val = raw_txt 
                        print(f"DEBUG: Found Med Label: '{raw_txt}' at {t_lbl}")
                        
                    elif "cow" in l_lower:
                        target_col = 'COWS_label'
                        target_sqi_col = 'COWS_sqi'
                        match = re.search(r'(\d+)', raw_txt)
                        if match: clean_val = match.group(1)
                        else: clean_val = raw_txt

                    # Check SQI Status of the Event Timestamp itself
                    status = "Unknown"
                    if self.windows_results:
                         found_win = False
                         for w in self.windows_results:
                             if w['start_ts'] <= t_lbl <= (w['start_ts'] + self.window_size_sec*1000):
                                 status = "Good" if w['is_good'] else "Bad"
                                 found_win = True
                                 break
                         if not found_win: status = "Out of Range"

                    assigned_beat_ts = None
                    assignment_note = "Skipped (Type)"
                    
                    if target_col:
                        # Find insertion point (Forward Assignment)
                        idx = np.searchsorted(t_beats, t_lbl)
                        
                        nearest_idx = -1
                        if idx < len(t_beats):
                            nearest_idx = idx
                            assignment_note = "Next Beat (Forward)"
                        elif len(t_beats) > 0:
                            nearest_idx = len(t_beats) - 1
                            assignment_note = "Last Beat (Fallback)"
                        else:
                            assignment_note = "No Beats"
                            
                        if 0 <= nearest_idx < len(feats):
                             # Assign Label
                             current = feats.iat[nearest_idx, feats.columns.get_loc(target_col)]
                             if not current:
                                 feats.iloc[nearest_idx, feats.columns.get_loc(target_col)] = clean_val
                                 # Set SQI status only if empty (primary status)
                                 feats.iloc[nearest_idx, feats.columns.get_loc(target_sqi_col)] = status
                             else:
                                 feats.iloc[nearest_idx, feats.columns.get_loc(target_col)] = f"{current}; {clean_val}"
                                 # Append SQI status
                                 curr_sqi = feats.iat[nearest_idx, feats.columns.get_loc(target_sqi_col)]
                                 feats.iloc[nearest_idx, feats.columns.get_loc(target_sqi_col)] = f"{curr_sqi}; {status}"
                                 
                             assigned_beat_ts = t_beats[nearest_idx]
                             print(f"DEBUG: Assigned '{clean_val}' to beat {assigned_beat_ts}ms. Status: {status}")
                        else:
                             print(f"DEBUG: Failed to assign '{clean_val}'. Idx {nearest_idx}")
                             assignment_note = f"Failed (Idx {nearest_idx})"

                    # Log for CSV
                    event_logs.append({
                        'timestamp_ms': t_lbl,
                        'event': raw_txt,
                        'sqi_status': status,
                        'assigned_beat_ms': assigned_beat_ts,
                        'assignment_note': assignment_note
                    })
                        
                # Export Audit Log
                pd.DataFrame(event_logs).to_csv(os.path.join(self.analysis_dir, 'events_summary.csv'), index=False)

                # 3. Clean up Nones -> Empty Strings or NaNs? 
                # Parquet handles None/NaN fine, but user might prefer empty string.
                feats['medication_label'] = feats['medication_label'].fillna('')
                feats['COWS_label'] = feats['COWS_label'].fillna('')
                feats['medication_sqi'] = feats['medication_sqi'].fillna('')
                feats['COWS_sqi'] = feats['COWS_sqi'].fillna('')
                
             # Apply SQI Filter for export too
            if self.windows_results:
                 mask = np.zeros(len(feats), dtype=bool) 
                 valid_ranges = []
                 for w in self.windows_results:
                    if w['is_good']:
                        valid_ranges.append( (w['start_ts'], w['start_ts'] + self.window_size_sec*1000) )
                 
                 # Apply mask directly to feats['timestamp_ms']
                 ts_vals = feats['timestamp_ms'].values # Re-assign ts_vals to the potentially modified feats
                 for r in valid_ranges:
                    mask |= (ts_vals >= r[0]) & (ts_vals < r[1])
                 feats = feats[mask]
            
            feats_path = os.path.join(self.analysis_dir, "beat_metrics.parquet")
            feats.to_parquet(feats_path)
            
            # --- Kubios Export (RR Intervals in Seconds) ---
            # Filter NaNs (from outlier removal) and convert to seconds
            rr_clean = feats['rr_interval_ms'].dropna()
            if not rr_clean.empty:
                rr_sec = rr_clean / 1000.0
                kubios_path = os.path.join(self.analysis_dir, "rr_intervals_kubios.txt")
                # Save as single column
                np.savetxt(kubios_path, rr_sec.values, fmt='%.5f')
                print(f"DEBUG: Saved Kubios RR file to {kubios_path} ({len(rr_sec)} beats)")

            
                win_df = pd.DataFrame(self.windows_results)
                win_path = os.path.join(self.analysis_dir, "sqi_window_metrics.parquet")
                win_df.to_parquet(win_path)
                
            # --- Yield Stats ---
            yield_msg = ""
            stats = self.calculate_yield_stats()
            if stats:
                yield_txt = (
                    f"Data Yield Summary\n"
                    f"==================\n"
                    f"Total Recording Duration: {stats['total_sec']/60:.2f} min\n"
                    f"  - Device Not Worn:      {stats['not_worn_sec']/60:.2f} min (Low Amplitude)\n"
                    f"  - Device Worn:          {stats['worn_sec']/60:.2f} min\n"
                    f"\n"
                    f"Good Data (during worn):  {stats['good_sec']/60:.2f} min\n"
                    f"----------------------------------\n"
                    f"Yield (Worn Only):        {stats['yield_worn_pct']:.1f}%\n"
                    f"Yield (Total):            {stats['yield_total_pct']:.1f}%\n"
                )
                
                yield_path = os.path.join(self.analysis_dir, "data_yield.txt")
                with open(yield_path, 'w') as f:
                    f.write(yield_txt)
                
                yield_msg = f"\n\nYield (Worn Only): {stats['yield_worn_pct']:.1f}% ({stats['good_sec']/60:.1f} min)"
                
            QMessageBox.information(self, "Export Successful", f"All results saved to:\n{self.analysis_dir}{yield_msg}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))
