import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QCheckBox, QGroupBox, QGridLayout, QScrollArea, QWidget, QTabWidget)
from PyQt5.QtCore import Qt

class ClusterSelectionDialog(QDialog):
    def __init__(self, cluster_results, windows_data, ecg_data, ppg_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cluster Quality Inspection (ECG + PPG)")
        self.resize(1200, 800)
        
        self.results = cluster_results
        self.windows = windows_data
        self.ecg_data = ecg_data
        self.ppg_data = ppg_data
        
        # Determine number of clusters
        self.labels = self.results['labels']
        self.unique_labels = np.unique(self.labels)
        self.checks = {}
        
        self.init_ui()
        self.showMaximized()
        
    def init_ui(self):
        main_layout = QVBoxLayout(self)
        
        # Header
        header = QLabel("Inspect 2-second windows for each cluster. Select clusters that represent GOOD signal quality.")
        header.setStyleSheet("font-size: 14px; font-weight: bold; margin: 10px;")
        main_layout.addWidget(header)
        
        # Tabs for clusters
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        for k in self.unique_labels:
            tab = QWidget()
            tab_layout = QVBoxLayout(tab)
            
            # Cluster Info
            indices = np.where(self.labels == k)[0]
            count = len(indices)
            
            # Controls for this cluster
            ctrl_layout = QHBoxLayout()
            chk = QCheckBox(f"Mark Cluster {k} as GOOD Quality")
            chk.setStyleSheet("font-size: 14px; font-weight: bold; color: green;")
            self.checks[k] = chk
            
            ctrl_layout.addWidget(QLabel(f"Cluster {k} (Count: {count})"))
            ctrl_layout.addStretch()
            ctrl_layout.addWidget(chk)
            
            tab_layout.addLayout(ctrl_layout)
            
            # Scroll Area for Grid
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            container = QWidget()
            grid = QGridLayout(container)
            
            # Load up to 20 examples
            limit = 20
            samples = indices[:limit]
            if len(indices) > limit:
                samples = np.random.choice(indices, limit, replace=False)
            
            cols = 2 # 2 columns of plots
            for idx_i, win_idx in enumerate(samples):
                if win_idx >= len(self.windows): continue
                
                w = self.windows[win_idx]
                s, e = w['start_idx'], w['end_idx']
                
                ecg_chunk = self.ecg_data[s:e] if self.ecg_data is not None else []
                ppg_chunk = self.ppg_data[s:e] if self.ppg_data is not None else []
                
                # Create Stacked Plot (ECG top, PPG bottom)
                p_widget = pg.GraphicsLayoutWidget()
                p_widget.setFixedHeight(200)
                p_widget.setBackground('w')
                
                # PPG Plot (Bottom)
                p2 = p_widget.addPlot(row=1, col=0)
                p2.plot(ppg_chunk, pen=pg.mkPen('g', width=2))
                p2.hideAxis('bottom')
                p2.hideAxis('left')
                p2.setLabel('left', 'PPG')
                
                # ECG Plot (Top) - Linked X?
                p1 = p_widget.addPlot(row=0, col=0)
                p1.plot(ecg_chunk, pen=pg.mkPen('b', width=2))
                p1.hideAxis('bottom')
                p1.hideAxis('left')
                p1.setLabel('left', 'ECG')
                p1.setXLink(p2)
                
                # Add to grid
                row = idx_i // cols
                col = idx_i % cols
                grid.addWidget(p_widget, row, col)
                
            scroll.setWidget(container)
            tab_layout.addWidget(scroll)
            
            self.tabs.addTab(tab, f"Cluster {k}")
            
        # Footer Buttons
        btns = QHBoxLayout()
        btn_apply = QPushButton("Apply Selection")
        btn_apply.clicked.connect(self.accept)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        
        btns.addStretch()
        btns.addWidget(btn_cancel)
        btns.addWidget(btn_apply)
        main_layout.addLayout(btns)

    def get_selected(self):
        selected = []
        for k, chk in self.checks.items():
            if chk.isChecked():
                selected.append(k)
        return selected
