import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QWizard, QWizardPage, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QComboBox, QCheckBox, QGroupBox,
    QFormLayout, QLineEdit, QMessageBox, QWidget
)
from PyQt5.QtCore import pyqtSignal

# Ensure src is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from gui.main_window import MainWindow

class ConfigPage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Configuration")
        self.setSubTitle("Select data source and analysis parameters.")

        layout = QVBoxLayout()

        # --- Data Selection ---
        data_group = QGroupBox("Data Source")
        data_layout = QFormLayout()
        
        self.mdaq_path_edit = QLineEdit()
        self.mdaq_btn = QPushButton("Browse...")
        self.mdaq_btn.clicked.connect(self.browse_mdaq)
        data_layout.addRow("mDAQ Folder:", self.create_row(self.mdaq_path_edit, self.mdaq_btn))
        
        self.label_path_edit = QLineEdit()
        self.label_btn = QPushButton("Browse...")
        self.label_btn.clicked.connect(self.browse_labels)
        data_layout.addRow("Label File (Optional):", self.create_row(self.label_path_edit, self.label_btn))
        
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)

        # --- Detectors ---
        det_group = QGroupBox("Peak Detectors")
        det_layout = QFormLayout()
        
        self.ecg_det_combo = QComboBox()
        self.ecg_det_combo.addItems(['promac', 'pantompkins', 'ecg2rr', 'vg', 'xqrs'])
        det_layout.addRow("ECG Detector:", self.ecg_det_combo)
        
        self.ppg_det_combo = QComboBox()
        self.ppg_det_combo.addItems(['msptd', 'e2e'])
        det_layout.addRow("PPG Detector:", self.ppg_det_combo)
        
        det_group.setLayout(det_layout)
        layout.addWidget(det_group)
        
        # --- Preprocessing ---
        prep_group = QGroupBox("Preprocessing")
        prep_layout = QFormLayout()
        
        self.prep_method_combo = QComboBox()
        self.prep_method_combo.addItems(['linear', 'wavelet'])
        prep_layout.addRow("Filter Method:", self.prep_method_combo)
        
        prep_group.setLayout(prep_layout)
        layout.addWidget(prep_group)

        # --- SQI Metrics ---
        sqi_group = QGroupBox("SQI Metrics")
        sqi_layout = QHBoxLayout()
        
        # ECG Metrics
        ecg_sqi_layout = QVBoxLayout()
        ecg_sqi_layout.addWidget(QLabel("<b>ECG Metrics</b>"))
        self.ecg_metrics = {
            'rel_power_qrs': QCheckBox("Relative Power"),
            'mean': QCheckBox("Mean"),
            'std': QCheckBox("Std Dev"),
            'autocorr': QCheckBox("Autocorrelation"),
            'entropy': QCheckBox("Entropy"),
            'zcr': QCheckBox("ZCR"),
            'snr': QCheckBox("SNR")
        }
        for cb in self.ecg_metrics.values():
            cb.setChecked(True)
            ecg_sqi_layout.addWidget(cb)
        sqi_layout.addLayout(ecg_sqi_layout)
        
        # PPG Metrics
        ppg_sqi_layout = QVBoxLayout()
        ppg_sqi_layout.addWidget(QLabel("<b>PPG Metrics</b>"))
        self.ppg_metrics = {
            'rel_power_1_225': QCheckBox("Relative Power"),
            'perfusion_index': QCheckBox("Perfusion Index"),
            'autocorr': QCheckBox("Autocorrelation"),
            'entropy': QCheckBox("Entropy"),
            'zcr': QCheckBox("ZCR"),
            'snr': QCheckBox("SNR")
        }
        for cb in self.ppg_metrics.values():
            cb.setChecked(True)
            ppg_sqi_layout.addWidget(cb)
        sqi_layout.addLayout(ppg_sqi_layout)
        
        sqi_group.setLayout(sqi_layout)
        layout.addWidget(sqi_group)

        # --- SQI Source ---
        source_group = QGroupBox("SQI Calculation Source")
        source_layout = QHBoxLayout()
        self.sqi_source_combo = QComboBox()
        self.sqi_source_combo.addItems(['Clean (Filtered)', 'Raw'])
        source_layout.addWidget(QLabel("Calculate SQI on:"))
        source_layout.addWidget(self.sqi_source_combo)
        source_group.setLayout(source_layout)
        layout.addWidget(source_group)

        self.setLayout(layout)

    def create_row(self, edit, btn):
        w = QWidget()
        l = QHBoxLayout()
        l.setContentsMargins(0,0,0,0)
        l.addWidget(edit)
        l.addWidget(btn)
        w.setLayout(l)
        return w

    def browse_mdaq(self):
        d = QFileDialog.getExistingDirectory(self, "Select mDAQ Folder")
        if d:
            self.mdaq_path_edit.setText(d)
            self.completeChanged.emit()

    def browse_labels(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select Label File", "", "CSV Files (*.csv)")
        if f:
            self.label_path_edit.setText(f)

    def isComplete(self):
        return bool(self.mdaq_path_edit.text())

class AnalysisWizard(QWizard):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Analysis Configuration Wizard")
        self.resize(600, 500)
        
        self.config_page = ConfigPage()
        self.addPage(self.config_page)
        
        self.main_window = None

    def accept(self):
        # Gather config
        config = {
            'mdaq_folder': self.config_page.mdaq_path_edit.text(),
            'label_file': self.config_page.label_path_edit.text(),
            'ecg_detector': self.config_page.ecg_det_combo.currentText(),
            'ppg_detector': self.config_page.ppg_det_combo.currentText(),
            'filter_method': self.config_page.prep_method_combo.currentText(),
            'ecg_metrics': [k for k, cb in self.config_page.ecg_metrics.items() if cb.isChecked()],
            'ppg_metrics': [k for k, cb in self.config_page.ppg_metrics.items() if cb.isChecked()],
            'sqi_source': self.config_page.sqi_source_combo.currentText()
        }
        
        self.main_window = MainWindow(config)
        self.main_window.show()
        super().accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    wizard = AnalysisWizard()
    wizard.show()
    sys.exit(app.exec_())
