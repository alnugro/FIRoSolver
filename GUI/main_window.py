import sys
import os
import traceback
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QComboBox, QSpinBox, QTextEdit, QTableWidget, QTableWidgetItem, QWidget, QFrame, QMessageBox
from PyQt6.QtCore import Qt
from PyQt6.uic import loadUi
from time import sleep

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


import numpy as np

from concurrent.futures import ThreadPoolExecutor

try:
    from .magnitude_plotter import MagnitudePlotter
    from .custom_navigation_toolbar import CustomNavigationToolbar
    from .live_logger import LiveLogger
    from .backend_mediator import BackendMediator
    from .ui_func import UIFunc

except:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from magnitude_plotter import MagnitudePlotter
    from custom_navigation_toolbar import CustomNavigationToolbar
    from live_logger import LiveLogger
    from GUI.backend_mediator import BackendMediator
    from ui_func import UIFunc


# New Window with Matplotlib Plot
class PlotWindow(QWidget):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Plot Window")
        self.setGeometry(100, 100, 600, 400)
        
        # Create a layout
        layout = QVBoxLayout()
        
        # Create a Matplotlib figure and add it to the layout
        self.canvas = FigureCanvas(Figure(figsize=(5, 3)))
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
        
        # Plot something
        self.plot()

    def plot(self):
        ax = self.canvas.figure.add_subplot(111)
        
        # Example plot: A simple sine wave
        t = np.linspace(0, 10, 500)
        y = np.sin(t)
        
        ax.plot(t, y)
        ax.set_title("Sine Wave")
        
        # Draw the canvas to show the plot
        self.canvas.draw()


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

        self.filter_order = None
        self.run_solver_but_on = False

        #create a separate thread for Backend that will be used later on
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.solver_is_running = False

        #initiate object for other classes
        self.magnitude_plotter = MagnitudePlotter(self)
        self.canvas = FigureCanvas(self.magnitude_plotter.fig)

        #connect widget with plotter
        self.mag_layout = QVBoxLayout(self.magnitude_wid)
        self.mag_layout.addWidget(self.canvas)

        self.toolbar = CustomNavigationToolbar(self.canvas, self)
        self.mag_toolbar = QHBoxLayout(self.toolbar_wid)
        self.mag_toolbar.addWidget(self.toolbar)

        self.input = {}

        self.order_current = None
        self.order_lower = None
        self.order_upper = None
        

        #declare additional classes
        self.logger = LiveLogger(self)

        self.plot_generated_flag = False    

        




    def widget_connect(self):
        #widget to add
        self.interpolate_transition_band_but = True
        self.start_with_error_prediction_but = True
        self.solver_accuracy_multiplier_box = 10 #according to Mr. Kumm this should be 16
        self.ignore_lowerbound_box = -10
        self.adder_depth_box = 0 #0 means infinite adder depth as maximum
        self.adder_wordlength_ext_box = 2 #default 2
        self.available_dsp_box = 0
        self.gain_wordlength_box = 6 #default
        self.gain_integer_width_box = 2 #default
        self.gain_upperbound_box = 4 #default
        self.gain_lowerbound_box = 1 #default
        self.coef_accuracy_box = 4 #default
        self.integer_width_box = 4 #default including signbit
        self.gurobi_thread_box = 1 #default 0 to use all threads available
        self.pysat_thread_box = 0 #default 0
        self.z3_thread_box = 0 #default 1
        self.solver_timeout_box = 0 #default 0 no timeout
        self.start_with_error_prediction_but = True #start with error pred flag
        self.deepsearch = False #deep search option
        self.patch_multiplier = 1 #how many dot to patch the leak
        self.gurobi_auto_thread = False #assert gurobi to use auto thread, not recommended


        


        # Connect the widgets
        self.magnitude_wid = self.findChild(QWidget, 'magnitude_wid')
        self.phase_wid = self.findChild(QWidget, 'phase_wid')
        self.group_delay_wid = self.findChild(QWidget, 'group_delay_wid')
        self.toolbar_wid = self.findChild(QFrame, 'toolbar_wid')
        self.logger_box = self.findChild(QTextEdit, 'logger') 


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

        
        self.sampling_rate_box = self.findChild(QSpinBox, 'sampling_rate_box')
        self.flatten_box = self.findChild(QSpinBox, 'flatten_box')
        self.filter_type_drop = self.findChild(QComboBox, 'filter_type_drop')
        self.bound_accuracy_box = self.findChild(QSpinBox, 'bound_accuracy')
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

        #connect tables
        self.magnitude_plotter_table = self.findChild(QTableWidget, 'magnitude_plotter_table')
        self.result_valid_table = self.findChild(QTableWidget, 'result_valid_tab')
        self.result_invalid_table = self.findChild(QTableWidget, 'result_invalid_tab')


        
    def get_ui_data_dict(self):
        data_dict = {
        "filter_type": self.filter_type_drop.currentIndex(),
        "order_upper": self.order_upper_box.value(),
        "wordlength": self.wordlength_box.value(),
        "sampling_rate": self.sampling_rate_box.value(),
        "flatten_value": self.flatten_box.value(),
        "gaussian_smoothing_value": self.gausian_slid.value(),
        "average_flaten_value": self.average_points_slid.value(),
        "cutoff_smoothing_value": self.cutoff_slid.value(),
    }
        return data_dict


    def on_mag_plotter_but_click(self):
        self.logger.plog(f"Generating...")
        rows = self.magnitude_plotter_table.rowCount()
        cols = self.magnitude_plotter_table.columnCount()
        data = []

        #Flag to check if the table is totally empty
        general_row_empty_flag = True


        for row in range(rows):
            row_data = []
            row_empty = True
            for col in range(cols):
                item = self.magnitude_plotter_table.item(row, col)
                if item is not None and item.text():
                    row_data.append(item.text())
                    row_empty = False

                    #Table is filled atleast once
                    general_row_empty_flag = False
                else:
                    row_data.append("")
            if not row_empty:
                data.append(row_data)

        if general_row_empty_flag == False:
            self.plot_generated_flag = True
        if self.validate_mag(data) == 1:
            return

        self.magnitude_plotter_data = np.array(data, dtype=float)

        self.magnitude_plotter.initiate_plot(self.magnitude_plotter_data)
        
    def validate_mag(self, data):
        for row in range(len(data)):
            for col in range(len(data[row])):
                if data[row][col] == "":
                    self.logger.plog(f"Element at [{row}, {col}] in \"Magnitude Plotter\" can't be empty: {data[row][col]}")
                    return 1
                try:
                    float(data[row][col])
                except ValueError:
                    self.logger.plog(f"Element at [{row}, {col}] in \"Magnitude Plotter\" is not a valid float: {data[row][col]}")
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
            self.logger.plog("Solver is currently running")
            return
        
        if not(self.plot_generated_flag):
            self.logger.plog("No bounds found, Please generate the bounds first in magnitude plotter!")
            return
        
        self.solver_is_running = True
        self.logger.plog("Solving FIR problem...")
        
        # Check if all necessary inputs are valid
        if self.input_validation_before_run() == 1:
            return
        
        #update input data
        self.magnitude_plotter.update_plotter_data(self.get_ui_data_dict())
        xdata, upper_ydata, lower_ydata ,cutoffs_x, cutoffs_upper_ydata, cutoffs_lower_ydata = self.magnitude_plotter.get_frequency_bounds()

        #validate data
        for i in range(len(xdata)):
            if np.isnan(upper_ydata[i]) and np.isnan(lower_ydata[i]):
                continue
            if not(np.isnan(upper_ydata[i])) and not(np.isnan(lower_ydata[i])):
                continue
            raise ValueError(f"Bounds is not the same at {i}, upper: {upper_ydata[i]} and lower: {lower_ydata[i]}. Contact Developer")
        
        ui_functionality = UIFunc(self, xdata, upper_ydata, lower_ydata, cutoffs_x, cutoffs_upper_ydata, cutoffs_lower_ydata)
        
        #do this if transition band interpolation is chosen
        if self.interpolate_transition_band_but:
            upper_ydata, lower_ydata = ui_functionality.interpolate_transition_band()

        
        initial_input_dictionary = ui_functionality.solver_input_dict_generator()
        
        #create the parallel mediator for the ui
        self.mediator  = BackendMediator(initial_input_dictionary)
        
        # Connect signals to slots
        self.mediator.log_message.connect(self.logger.plog)
        self.mediator.exception_message.connect(self.show_error_dialog)

        #start backend
        self.mediator.start()


        # sleep(10)

        

    def input_parser_tuple(self):
        input_arg = (
            self.filter_type_drop.currentIndex(),
            self.order_upper_box.value(),
            self.wordlength_box.value(),
            self.magnitude_plotter.get_upper_ydata(),
            self.magnitude_plotter.get_lower_ydata(),
            self.magnitude_plotter.get_xdata(),
        )
        return input_arg

    def on_closing(self):
        pass        


    def input_validation_before_run(self):
        if self.order_upper_box.value() == 0:
            self.logger.plog(f"Filter Order can't be 0")
            return 1  
        if self.wordlength_box.value() == 0:
            self.logger.plog(f"Maximum Wordlength can't be 0")
            return 1
        return 0

    def show_error_dialog(self, error_msg_str):
        # Create a QMessageBox for exception or error
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setWindowTitle("An exception occurred!")
        msg_box.setText(error_msg_str)  # Set the actual error message string here
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)

        msg_box.exec()  # Display the pop-up dialog   

    def addRow(self):
        # Get the current row count
        rowCount = self.table.rowCount()

        # Insert a new row
        self.table.insertRow(rowCount)

        # Add row number in the first column
        self.table.setItem(rowCount, 0, QTableWidgetItem(str(rowCount + 1)))

        # Create a button in the second column
        button = QPushButton(f'Button {rowCount + 1}')
        button.clicked.connect(lambda _, row=rowCount + 1: self.buttonClicked(row))

        # Add the button to the table
        self.table.setCellWidget(rowCount, 1, button)          


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
