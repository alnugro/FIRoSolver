import sys
import traceback
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QComboBox, QSpinBox, QTextEdit, QTableWidget, QTableWidgetItem, QWidget, QFrame
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt6.uic import loadUi
import numpy as np
from .magnitude_plotter import MagnitudePlotter
from .custom_navigation_toolbar import CustomNavigationToolbar
from backend.solver_init import SolverInit
from concurrent.futures import ThreadPoolExecutor


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("GUI/ui_files/FIR.ui", self)

        #connect widget to program
        self.widget_connect()

        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.destroyed.connect(self.on_closing)

        # Plotter data
        self.magnitude_plotter_data = np.array([])
        self.magnitude_plotter_final = np.array([])

        # Logger line number
        self.loggerline = 1

        self.filter_order = None
        self.run_solver_but_on = False

        #create a separate thread for Backend that will be used later on
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.solver_is_running = False

        #initiate object for other classes
        self.magnitude_plotter = MagnitudePlotter(self)
        self.solver=SolverInit()
        self.canvas = FigureCanvas(self.magnitude_plotter.fig)

        #connect widget with plotter
        self.mag_layout = QVBoxLayout(self.magnitude_wid)
        self.mag_layout.addWidget(self.canvas)

        self.toolbar = CustomNavigationToolbar(self.canvas, self)
        self.mag_toolbar = QHBoxLayout(self.toolbar_wid)
        self.mag_toolbar.addWidget(self.toolbar)




    def widget_connect(self):
        # Connect the widgets
        self.magnitude_wid = self.findChild(QWidget, 'magnitude_wid')
        self.phase_wid = self.findChild(QWidget, 'phase_wid')
        self.group_delay_wid = self.findChild(QWidget, 'group_delay_wid')
        self.toolbar_wid = self.findChild(QFrame, 'toolbar_wid')

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
        self.order_lower_box = self.findChild(QSpinBox, 'order_lower_box')
        self.order_upper_box = self.findChild(QSpinBox, 'order_upper_box')
        self.wordlength_box = self.findChild(QSpinBox, 'wordlength_box')
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

    def get_data_dict(self):
        data_dict = {
        "filter_type": self.filter_type_drop.currentIndex(),
        "order_upper": self.order_upper_box.value(),
        "order_lower": self.order_lower_box.value(),
        "wordlength": self.wordlength_box.value(),
        "sampling_rate": self.sampling_rate_box.value(),
        "flatten_value": self.flatten_box.value(),
        "gaussian_smoothing_value": self.gausian_slid.value(),
        "average_flaten_value": self.average_points_slid.value(),
        "cutoff_smoothing_value": self.cutoff_slid.value(),
    }
        return data_dict



    def on_mag_plotter_but_click(self):
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

        self.magnitude_plotter.initiate_plot(self.magnitude_plotter_data)
        
    def validate_mag(self, data):
        for row in range(len(data)):
            for col in range(len(data[row])):
                if data[row][col] == "":
                    self.live_logger(f"Element at [{row}, {col}] in \"Magnitude Plotter\" can't be empty: {data[row][col]}")
                    return 1
                try:
                    float(data[row][col])
                except ValueError:
                    self.live_logger(f"Element at [{row}, {col}] in \"Magnitude Plotter\" is not a valid float: {data[row][col]}")
                    return 1
        return 0

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
        #solver running flag
        if self.solver_is_running:
            self.live_logger("Solver is currently running")
            return
        
        self.solver_is_running = True
        self.live_logger("Solving FIR problem...")
        
        # Check if all necessary inputs are valid
        if self.input_validation_before_run() == 1:
            return

        #update input data
        self.magnitude_plotter.update_plotter_data(self.get_data_dict())
        self.solver.update_plotter_data(self.get_data_dict())

        # Set bounds in solver from plotter
        self.solver.set_input_arg(self.magnitude_plotter.get_frequency_bounds())

        try:    
            future = self.executor.submit(self.solver.run_solver)
            future.add_done_callback(self.solver_done)

        except Exception as e:
            self.live_logger(f"Backend Error: {e}")
            print(f"Backend Error: {e}")
            traceback.print_exc()



    def solver_done(self, future):
        self.solver_is_running = False
        self.live_logger("Solving Done, see Result!")

        try:
            # Check if the future resulted in an exception
            exception = future.exception()
            if exception:
                raise exception

            # Retrieve the result from the solver
            result = self.solver.get_result()

            # Plot the result using MagnitudePlotter
            self.magnitude_plotter.plot_result(result)

        except Exception as e:
            # Capture and print detailed exception information
            self.live_logger(f"Error during result processing: {e}")
            print(f"Error during result processing: {e}")
            traceback.print_exc()
        

    def input_parser_tuple(self):
        input_arg = (
            self.filter_type_drop.currentIndex(),
            self.order_upper_box.value(),
            self.order_lower_box.value(),
            self.wordlength_box.value(),
            self.magnitude_plotter.get_upper_ydata(),
            self.magnitude_plotter.get_lower_ydata(),
            self.magnitude_plotter.get_xdata(),
        )
        return input_arg

    def on_closing(self):
        pass        

    def live_logger(self, text):
        self.logger.append(f"[{self.loggerline}] {text}")
        self.loggerline += 1

    def input_validation_before_run(self):
        if self.order_upper_box.value() == 0:
            self.live_logger(f"Filter Order can't be 0")
            return 1  
        if self.order_lower_box.value() == 0:
            self.live_logger(f"Filter Order can't be 0")
            return 1
        if self.wordlength_box.value() == 0:
            self.live_logger(f"Maximum Wordlength can't be 0")
            return 1
        return 0             


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
