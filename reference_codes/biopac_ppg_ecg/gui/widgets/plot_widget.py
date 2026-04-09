import pyqtgraph as pg
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import pyqtSignal
import numpy as np

class PlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w') # White background
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.layout.addWidget(self.plot_widget)
        
        self.curve = self.plot_widget.plot(pen='b')
        self.beat_scatter = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 120))
        self.onset_scatter = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(0, 0, 255, 120)) # Blue for onsets
        self.plot_widget.addItem(self.beat_scatter)
        self.plot_widget.addItem(self.beat_scatter)
        self.plot_widget.addItem(self.onset_scatter)
        
        # Measure Tool Items
        self.vLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('k', style=2)) # Dashed black
        self.hLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('k', style=2))
        self.measure_label = pg.TextItem("", anchor=(0, 1), color='k', fill=pg.mkBrush(255, 255, 255, 200))
        self.measure_mode = False
        self.measuring_frozen = False
        
        # Proxy
        self.proxy = pg.SignalProxy(self.plot_widget.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved)
        self.plot_widget.scene().sigMouseClicked.connect(self.mouseClicked)
        
        # Regions for bad quality
        self.bad_regions = [] 
        
        
        # Internal Data Storage
        self.full_time = None
        self.full_signal = None
        self.downsample_factor = 1 # 1 = Full resolution
        self.width = 1
        self.color = 'b'
        
    def plot_signal(self, time, signal, color='b', width=1, clear=True):
        if clear:
            self.plot_widget.clear()
            self.bad_regions = []
            self.beat_scatter.clear()
            self.onset_scatter.clear()
            self.plot_widget.addItem(self.beat_scatter)
            self.plot_widget.addItem(self.onset_scatter)
            self.curve = None # Reset curve ref as clear() removes it
            
        self.full_time = time
        self.full_signal = signal
        self.color = color
        self.width = width
        
        self._update_plot_data()
        
    def set_downsample_factor(self, factor):
        self.downsample_factor = max(1, int(factor))
        self._update_plot_data()
        
    def _update_plot_data(self):
        if self.full_time is None or self.full_signal is None: return
        
        # Slice for visualization
        step = self.downsample_factor
        
        # Ensure we work with numpy arrays for pyqtgraph compatibility
        t_view = np.array(self.full_time[::step])
        s_view = np.array(self.full_signal[::step])
        
        # If we just clear(), we lose items. 
        # Better to setData on the curve if it exists.
        pen = pg.mkPen(color=self.color, width=self.width)
        
        if self.curve is None:
             self.curve = self.plot_widget.plot(t_view, s_view, pen=pen)
        else:
             self.curve.setData(t_view, s_view, pen=pen)

    def toggle_pan_mode(self, enabled):
        if enabled:
            self.plot_widget.getViewBox().setMouseMode(pg.ViewBox.PanMode)
        else:
            self.plot_widget.getViewBox().setMouseMode(pg.ViewBox.RectMode)

    def plot_beats(self, time, signal, beat_indices):
        if len(beat_indices) == 0:
            self.beat_scatter.clear()
            return
        
        # Ensure indices are within bounds
        valid_indices = beat_indices[beat_indices < len(signal)]
        
        x = time[valid_indices]
        y = signal[valid_indices]
        
        self.beat_scatter.setData(x, y)

    def plot_onsets(self, time, signal, onset_indices):
        if len(onset_indices) == 0:
            self.onset_scatter.clear()
            return

        valid_indices = onset_indices[onset_indices < len(signal)]
        x = time[valid_indices]
        y = signal[valid_indices]
        self.onset_scatter.setData(x, y)
        
    def mark_bad_windows(self, windows):
        """
        windows: list of tuples (start_time, end_time)
        """
        # Clear existing regions
        # Clear existing regions
        # Use a copy of list to avoid iteration issues
        for r in list(self.bad_regions):
            try: 
                self.plot_widget.removeItem(r)
            except: pass
        self.bad_regions = []
            
        for start, end in windows:
            region = pg.LinearRegionItem([start, end], movable=False, brush=pg.mkBrush(255, 0, 0, 50))
            self.plot_widget.addItem(region)
            self.bad_regions.append(region)
            
    def set_title(self, title):
        self.plot_widget.setTitle(title, color='k')

    def enable_measure_mode(self, enabled):
        self.measure_mode = enabled
        if enabled:
            self.plot_widget.addItem(self.vLine, ignoreBounds=True)
            self.plot_widget.addItem(self.hLine, ignoreBounds=True)
            self.plot_widget.addItem(self.measure_label, ignoreBounds=True)
        else:
            self.plot_widget.removeItem(self.vLine)
            self.plot_widget.removeItem(self.hLine)
            self.plot_widget.removeItem(self.measure_label)
            self.measuring_frozen = False

    def mouseMoved(self, evt):
        if not self.measure_mode or self.measuring_frozen: return
        pos = evt[0]
        if self.plot_widget.sceneBoundingRect().contains(pos):
            mousePoint = self.plot_widget.plotItem.vb.mapSceneToView(pos)
            self.vLine.setPos(mousePoint.x())
            self.hLine.setPos(mousePoint.y())
            self.measure_label.setPos(mousePoint.x(), mousePoint.y())
            self.measure_label.setText(f"t={mousePoint.x():.3f}s\ny={mousePoint.y():.3f}")

    def mouseClicked(self, evt):
        if not self.measure_mode: return
        if evt.double(): return # Ignore double clicks
        # Toggle freeze
        self.measuring_frozen = not self.measuring_frozen
