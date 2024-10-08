import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout,QHBoxLayout ,QPushButton, QSlider, QComboBox, QSpinBox, QTextEdit, QTableWidget, QTableWidgetItem, QWidget,QFrame
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt6.uic import loadUi
import numpy as np
from .magnitude_plotter import MagnitudePlotter
from .custom_navigation_toolbar import CustomNavigationToolbar
from backend.solver_init import SolverInit

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("GUI/ui_files/FIR.ui", self)
        self.widget_connect()

        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.destroyed.connect(self.on_closing)

        #plotter data
        self.magnitude_plotter_data = np.array([])

        #logger line number
        self.loggerline = 1

        self

    def widget_connect(self):
        #connect the widget
        self.magnitude_wid = self.findChild(QWidget, 'magnitude_wid')
        self.phase_wid = self.findChild(QWidget, 'phase_wid')
        self.group_delay_wid = self.findChild(QWidget, 'group_delay_wid')

        self.mag_plotter_but = self.findChild(QPushButton, 'mag_plotter_but')
        self.nyquist_freq_but = self.findChild(QPushButton, 'nyquist_freq_but')
        self.normal_freq_but = self.findChild(QPushButton, 'normal_freq_but')
        self.mirror_uptolo_but = self.findChild(QPushButton, 'mirror_uptolo_but')
        self.mirror_lotoup = self.findChild(QPushButton, 'mirror_lotoup')
        self.equiripple_but = self.findChild(QPushButton, 'equiripple_but')
        self.selection_but = self.findChild(QPushButton, 'selection_but')
        self.delete_select_but = self.findChild(QPushButton, 'delete_select_but')
        self.flatten_but = self.findChild(QPushButton, 'flatten_but')
        self.merge_plot_but = self.findChild(QPushButton, 'merge_plot_but')
        self.average_select_but = self.findChild(QPushButton, 'average_select_but')
        self.filtering_select_but = self.findChild(QPushButton, 'filtering_select_but')
        self.undo_but = self.findChild(QPushButton, 'undo_but')
        self.redo_but = self.findChild(QPushButton, 'redo_but')
        self.average_but = self.findChild(QPushButton, 'average_but')
        self.filtering_but = self.findChild(QPushButton, 'filtering_but')
        self.run_solver_but = self.findChild(QPushButton, 'run_solver_but')

        self.magnitude_plotter_table = self.findChild(QTableWidget, 'magnitude_plotter_table')
        self.sampling_rate_box = self.findChild(QSpinBox, 'sampling_rate_box')
        self.flatten_box = self.findChild(QSpinBox, 'flatten_box')
        self.filter_type_drop = self.findChild(QComboBox, 'filter_type_drop')
        self.order_lower_box = self.findChild(QTextEdit, 'order_lower_box')
        self.order_upper_box = self.findChild(QTextEdit, 'order_upper_box')
        self.wordlength_box = self.findChild(QTextEdit, 'wordlength_box')
        self.gausian_slid = self.findChild(QSlider, 'gausian_slid')
        self.average_points_slid = self.findChild(QSlider, 'average_points_slid')
        self.cutoff_slid = self.findChild(QSlider, 'cutoff_slid')

        # Connect buttons to functions
        self.mag_plotter_but.clicked.connect(self.on_mag_plotter_but_click)
        self.nyquist_freq_but.clicked.connect(self.on_nyquist_freq_but_click)
        self.normal_freq_but.clicked.connect(self.on_normal_freq_but_click)
        self.mirror_uptolo_but.clicked.connect(self.on_mirror_uptolo_but_click)
        self.mirror_lotoup.clicked.connect(self.on_mirror_lotoup_click)
        self.equiripple_but.clicked.connect(self.on_equiripple_but_click)
        self.selection_but.clicked.connect(self.on_selection_but_click)
        self.delete_select_but.clicked.connect(self.on_delete_select_but_click)
        self.flatten_but.clicked.connect(self.on_flatten_but_click)
        self.merge_plot_but.clicked.connect(self.on_merge_plot_but_click)
        self.average_select_but.clicked.connect(self.on_average_select_but_click)
        self.filtering_select_but.clicked.connect(self.on_filtering_select_but_click)
        self.undo_but.clicked.connect(self.on_undo_but_click)
        self.redo_but.clicked.connect(self.on_redo_but_click)
        self.average_but.clicked.connect(self.on_average_but_click)
        self.filtering_but.clicked.connect(self.on_filtering_but_click)
        self.run_solver_but.clicked.connect(self.on_run_solver_but_click)




        self.magnitude_plotter = MagnitudePlotter(self)
        self.canvas = FigureCanvas(self.magnitude_plotter.fig)

        self.mag_layout = QVBoxLayout(self.magnitude_wid)
        self.mag_layout.addWidget(self.canvas)

        self.toolbar = CustomNavigationToolbar(self.canvas, self)
        self.mag_toolbar = QHBoxLayout(self.toolbar_wid)
        self.mag_toolbar.addWidget(self.toolbar)

        

    def on_mag_plotter_but_click(self):
        self.mag_array_gen()
        self.magnitude_plotter.initiate_plot(self.magnitude_plotter_data)

    def mag_array_gen(self):
        self.live_logger(f"Generating...")
        rows = self.magnitude_plotter_table.rowCount()
        cols = self.magnitude_plotter_table.columnCount()
        data = []

        for row in range(rows):
            row_data = []
            row_empty = True
            for col in range(cols):
                item = self.magnitude_plotter_table.item(row, col)
                if item is not None and item.text():
                    row_data.append(item.text())
                    row_empty = False
                else:
                    row_data.append("")
            if not row_empty:
                data.append(row_data)

        if self.validate_mag(data) == 1:
            return

        self.magnitude_plotter_data = np.array(data, dtype=float)

    def validate_mag(self, data):
        for row in range(len(data)):
            for col in range(len(data[row])):
                if data[row][col] == "":
                    self.live_logger(f"Element at [{row}, {col}] in \"Magnitude Plotted\" can't be empty: {data[row][col]}")
                    return 1
                try:
                    float(data[row][col])
                except ValueError:
                    self.live_logger(f"Element at [{row}, {col}] in \"Magnitude Plotted\" is not a valid float: {data[row][col]}")
                    return 1
        return 0
    
    def on_mag_plotter_but_click(self):
        pass

    def on_nyquist_freq_but_click(self):
        pass

    def on_normal_freq_but_click(self):
        pass

    def on_mirror_uptolo_but_click(self):
        pass

    def on_mirror_lotoup_click(self):
        pass

    def on_equiripple_but_click(self):
        pass

    def on_selection_but_click(self):
        pass

    def on_delete_select_but_click(self):
        pass

    def on_flatten_but_click(self):
        pass

    def on_merge_plot_but_click(self):
        pass

    def on_average_select_but_click(self):
        pass

    def on_filtering_select_but_click(self):
        pass

    def on_undo_but_click(self):
        pass

    def on_redo_but_click(self):
        pass

    def on_average_but_click(self):
        pass

    def on_filtering_but_click(self):
        pass

    def on_run_solver_but_click(self):
        pass

    def on_motion(self, event):
        # self.magnitude_plotter.on_motion(event)
        pass

    def on_click(self, event):
        # self.magnitude_plotter.on_click(event)
        pass

    def on_release(self, event):
        # self.magnitude_plotter.on_release(event)
        pass

    def on_closing(self):
        # plt.close(self.magnitude_plotter.fig)
        pass

    def live_logger(self, text):
        self.logger.append(f"[{self.loggerline}] {text}")
        self.loggerline += 1

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
