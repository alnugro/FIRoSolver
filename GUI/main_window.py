import sys
import os
import traceback
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QComboBox, 
                             QSpinBox, QTextEdit, QTableWidget, QTableWidgetItem, QWidget, QFrame, QMessageBox, 
                             QDoubleSpinBox, QLineEdit, QLabel, QCheckBox, QProgressBar)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.uic import loadUi
from PyQt6.QtGui import QIcon, QPixmap
import time
import json
import copy

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from multiprocessing import Process, Manager
from concurrent.futures import TimeoutError  # Correct import for TimeoutError
from pebble import ProcessPool, ProcessExpired
from concurrent.futures import TimeoutError, CancelledError, wait, ALL_COMPLETED


import numpy as np

from concurrent.futures import ThreadPoolExecutor

try:
    from .magnitude_plotter import MagnitudePlotter
    from .custom_navigation_toolbar import CustomNavigationToolbar
    from .live_logger import LiveLogger
    from .backend_mediator import BackendMediator
    from .ui_func import UIFunc
    from backend.backend_main import SolverBackend
    from .result_handler import JsonUnloader, PydspHandler, DynamicTableWidgetResult, DynamicTableWidgetProblems
    from .save_load_handler import SaveLoadPlotHandler, SaveLoadResHandler
    from .additional_func_mediator import AdditionalFuncMediator
    from .result_subwindow import AutomaticParameterResultSubWindow, DetailsSubWindow

except:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from magnitude_plotter import MagnitudePlotter
    from custom_navigation_toolbar import CustomNavigationToolbar
    from live_logger import LiveLogger
    from GUI.backend_mediator import BackendMediator
    from ui_func import UIFunc
    from backend.backend_main import SolverBackend
    from result_handler import JsonUnloader, PydspHandler, DynamicTableWidgetResult, DynamicTableWidgetProblems
    from save_load_handler import SaveLoadPlotHandler, SaveLoadResHandler
    from GUI.additional_func_mediator import AdditionalFuncMediator
    from result_subwindow import AutomaticParameterResultSubWindow, DetailsSubWindow




# New Window with Matplotlib Plot
class PlotWindow(QWidget):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Plot Window")
        self.setGeometry(100, 100, 600, 400)
        self.setWindowIcon(QIcon("GUI/icon/icon.png"))
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
        self.setWindowIcon(QIcon("GUI/icon/icon.png"))

        self.setWindowTitle("FIRoSolver")
        #connect widget to program
        self.widget_connect()

        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.destroyed.connect(self.on_closing)


        self.image_label.setFixedSize(300, 300)  # Set QLabel dimensions
        pixmap = QPixmap("GUI/icon/icon.png")
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

        # Plotter data
        self.magnitude_plotter_data = np.array([])
        self.magnitude_plotter_final = np.array([])

        self.filter_order = None
        self.run_solver_but_on = False

        self.solver_is_running = False
        
        #initiate object for other classes
        self.magnitude_plotter = MagnitudePlotter(False ,self)
        self.canvas = FigureCanvas(self.magnitude_plotter.fig)

        #connect widget with plotter
        self.mag_layout = QVBoxLayout(self.magnitude_wid)
        self.mag_layout.addWidget(self.canvas)

        self.toolbar = CustomNavigationToolbar(self.canvas, self)
        self.mag_toolbar = QHBoxLayout(self.toolbar_wid)
        self.mag_toolbar.addWidget(self.toolbar)

        # Instantiate DynamicTableWidgetResult for result_valid_table and result_invalid_table
        self.valid_table_widget = DynamicTableWidgetResult(self.result_valid_table, True, self)
        self.invalid_table_widget = DynamicTableWidgetResult(self.result_invalid_table, False, self)
        
        self.table_settings_init(5, self.result_valid_table)
        self.table_settings_init(5, self.result_invalid_table)

        self.solver_runs_table_widget = DynamicTableWidgetProblems(self.solver_runs_table, self)
        self.table_settings_init(3, self.solver_runs_table)

        # Start timers to update tables
        self.valid_table_widget.startTimer()
        self.invalid_table_widget.startTimer()
        self.solver_runs_table_widget.startTimer()

        self.input = {}

        self.order_current = None
        self.order_lower = None
        self.order_upper = None
        self.day_night = False
        

        #declare additional classes
        self.logger = LiveLogger(self)

        self.plot_load_saver = SaveLoadPlotHandler()
        self.res_load_saver = SaveLoadResHandler()

        self.plot_generated_flag = False    
       
        self.mediator = None
        self.solving_canceled = False
        self.selection_mode = False

        self.plotter_updater = None
        

        self.widget_connect_after()

        self.auto_search_dict = None


        
    def table_settings_init(self, len, table_widget):
        for i in range(len):
            table_widget.setColumnWidth(i, 90)
        
            


    def widget_connect(self):
        #widget to add
        # Finding child widgets based on their type
        self.interpolate_transition_band_check = self.findChild(QCheckBox, 'interpolate_transition_band_check')
        self.start_with_error_prediction_check = self.findChild(QCheckBox, 'start_with_error_prediction_check')
        self.deepsearch_check = self.findChild(QCheckBox, 'deepsearch_check')
        self.gurobi_auto_thread_check = self.findChild(QCheckBox, 'gurobi_auto_thread_check')

        self.solver_accuracy_multiplier_box = self.findChild(QSpinBox, 'solver_accuracy_multiplier_box')
        self.adder_depth_box = self.findChild(QSpinBox, 'adder_depth_box')
        self.adder_wordlength_ext_box = self.findChild(QSpinBox, 'adder_wordlength_ext_box')
        self.available_dsp_box = self.findChild(QSpinBox, 'available_dsp_box')
        self.gain_upper_box = self.findChild(QSpinBox, 'gain_upper_box')
        self.gain_lower_box = self.findChild(QSpinBox, 'gain_lower_box')
        self.coef_accuracy_box = self.findChild(QSpinBox, 'coef_accuracy_box')
        self.integer_width_box = self.findChild(QSpinBox, 'integer_width_box')
        self.gurobi_thread_box = self.findChild(QSpinBox, 'gurobi_thread_box')
        self.pysat_thread_box = self.findChild(QSpinBox, 'pysat_thread_box')
        self.z3_thread_box = self.findChild(QSpinBox, 'z3_thread_box')
        self.solver_timeout_box = self.findChild(QSpinBox, 'solver_timeout_box')
        self.patch_multiplier_box = self.findChild(QSpinBox, 'patch_multiplier_box')
        self.worker_box = self.findChild(QSpinBox, 'worker_box')
        self.search_step_box = self.findChild(QSpinBox, 'search_step_box')

        self.vhdl_input_word_box = self.findChild(QSpinBox, 'vhdl_input_word_box')

        # Progress bars
        self.gurobi_progressbar = self.findChild(QProgressBar, 'gurobi_progressbar')
        self.z3_progressbar = self.findChild(QProgressBar, 'z3_progressbar')
        self.pysat_progressbar = self.findChild(QProgressBar, 'pysat_progressbar')

        # Save and Load buttons
        self.res_save_but = self.findChild(QPushButton, 'res_save_but')
        self.res_load_but = self.findChild(QPushButton, 'res_load_but')
        self.res_reset_but = self.findChild(QPushButton, 'res_reset_but')

        
        self.position_label = self.findChild(QLabel, 'position_label')
        self.image_label = self.findChild(QLabel, 'image_label')


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
        self.selection_but = self.findChild(QPushButton, 'selection_but')
        self.delete_select_but = self.findChild(QPushButton, 'delete_select_but')
        self.flatten_but = self.findChild(QPushButton, 'flatten_but')
        self.filtering_select_but = self.findChild(QPushButton, 'filtering_select_but')
        self.undo_but = self.findChild(QPushButton, 'undo_but')
        self.redo_but = self.findChild(QPushButton, 'redo_but')
        self.smoothen_but = self.findChild(QPushButton, 'smoothen_but')
        self.run_solver_but = self.findChild(QPushButton, 'run_solver_but')
        self.cancel_solver_but = self.findChild(QPushButton, 'cancel_solver_but')
        self.day_night_but = self.findChild(QPushButton, 'day_night_but')
        self.reset_dot_but = self.findChild(QPushButton, 'reset_dot_but')
        self.quick_check_sat_but = self.findChild(QPushButton, 'quick_check_sat_but')

        self.asserted_search_check = self.findChild(QCheckBox, 'asserted_search_check')
        self.auto_search_check = self.findChild(QCheckBox, 'auto_search_check')
        self.asserted_search_check.stateChanged.connect(lambda: self.auto_search_check.setChecked(False) if self.asserted_search_check.isChecked() else self.auto_search_check.setChecked(True))
        self.auto_search_check.stateChanged.connect(lambda: self.asserted_search_check.setChecked(False) if self.auto_search_check.isChecked() else self.asserted_search_check.setChecked(True))

        self.real_word_check = self.findChild(QCheckBox, 'real_word_check')
        self.assert_wordlength_check = self.findChild(QCheckBox, 'assert_wordlength_check')
        self.real_word_check.stateChanged.connect(lambda: self.assert_wordlength_check.setChecked(False) if self.real_word_check.isChecked() else self.assert_wordlength_check.setChecked(True))
        self.assert_wordlength_check.stateChanged.connect(lambda: self.real_word_check.setChecked(False) if self.assert_wordlength_check.isChecked() else self.real_word_check.setChecked(True))
        
        self.wordlength_assertion_box = self.findChild(QSpinBox, 'wordlength_assertion_box')
        self.integer_width_box_2 = self.findChild(QSpinBox, 'integer_width_box_2')
        self.gain_upper_box_2 = self.findChild(QSpinBox, 'gain_upper_box_2')
        self.gain_lower_box_2 = self.findChild(QSpinBox, 'gain_lower_box_2')
        self.available_dsp_box_2 = self.findChild(QSpinBox, 'available_dsp_box_2')
        self.adder_wordlength_ext_box_2 = self.findChild(QSpinBox, 'adder_wordlength_ext_box_2')
        self.adder_depth_box_2 = self.findChild(QSpinBox, 'adder_depth_box_2')
        self.solver_accuracy_multiplier_box_2 = self.findChild(QSpinBox, 'solver_accuracy_multiplier_box_2')

        self.coef_accuracy_box_2 = self.findChild(QSpinBox, 'coef_accuracy_box_2')


        self.filter_type_drop = self.findChild(QComboBox, 'filter_type_drop')
        self.filter_type_drop.currentIndexChanged.connect(self.order_changed)

        self.sampling_rate_box = self.findChild(QSpinBox, 'sampling_rate_box')
        self.bound_accuracy_box = self.findChild(QSpinBox, 'bound_accuracy')
        self.wordlength_box = self.findChild(QSpinBox, 'wordlength_box')

        self.order_upper_box = self.findChild(QSpinBox, 'order_upper_box')
        self.order_upper_box.valueChanged.connect(self.order_changed)
        self.order_upper_last_value = self.order_upper_box.value()
        
        self.flatten_box = self.findChild(QDoubleSpinBox, 'flatten_box')

        self.gausian_slid = self.findChild(QSlider, 'gausian_slid')
        self.average_points_slid = self.findChild(QSlider, 'average_points_slid')
        self.smoothing_kernel_slid = self.findChild(QSlider, 'smoothing_kernel_slid')

        self.intfeastol_box = self.findChild(QSpinBox, 'intfeastol_box')

        
        

        # Connect buttons to functions
        self.mag_plotter_but.clicked.connect(self.on_mag_plotter_but_click)
        self.nyquist_freq_but.clicked.connect(self.on_nyquist_freq_but_click)
        self.normal_freq_but.clicked.connect(self.on_normal_freq_but_click)
        self.mirror_uptolo_but.clicked.connect(self.on_mirror_uptolo_but_click)
        self.mirror_lotoup.clicked.connect(self.on_mirror_lotoup_click)
        self.selection_but.clicked.connect(self.on_selection_but_click)
        self.delete_select_but.clicked.connect(self.on_delete_select_but_click)
        self.flatten_but.clicked.connect(self.on_flatten_but_click)
        self.smoothen_but.clicked.connect(self.on_smoothen_but_click)
        self.undo_but.clicked.connect(self.on_undo_but_click)
        self.redo_but.clicked.connect(self.on_redo_but_click)
        self.run_solver_but.clicked.connect(self.on_run_solver_but_click)
        self.run_solver_but.setToolTip('Run the solvers, with your given parameters')
        self.day_night_but.clicked.connect(self.on_day_night_but_click)
        self.reset_dot_but.clicked.connect(self.on_reset_dot_but_click)
        self.cancel_solver_but.clicked.connect(self.kill_solver_instance)
        self.quick_check_sat_but.clicked.connect(self.quick_check_sat_but_click)


        #connect tables
        self.magnitude_plotter_table = self.findChild(QTableWidget, 'magnitude_plotter_table')
        self.set_column_read_only_and_center_align(self.magnitude_plotter_table, 0)
        self.magnitude_plotter_table.setColumnWidth(0, 90)
        self.magnitude_plotter_table.setColumnWidth(1, 90)
        self.magnitude_plotter_table.setColumnWidth(2, 200)
        self.magnitude_plotter_table.setColumnWidth(3, 200)
        self.magnitude_plotter_table.cellChanged.connect(self.on_cell_changed)

        self.result_valid_table = self.findChild(QTableWidget, 'result_valid_tab')
        self.result_invalid_table = self.findChild(QTableWidget, 'result_invalid_tab')
        self.solver_runs_table = self.findChild(QTableWidget, 'solver_runs_tab')

        #plotted data
        self.remove_sel_but = self.findChild(QPushButton, 'remove_sel_but')
        self.add_stopband_but = self.findChild(QPushButton, 'add_stopband_but')
        self.add_passband_but = self.findChild(QPushButton, 'add_passband_but')
        self.plot_save_but = self.findChild(QPushButton, 'plot_save_but')
        self.plot_load_but = self.findChild(QPushButton, 'plot_load_but')

        # Connect buttons to functions
        self.remove_sel_but.clicked.connect(self.on_remove_sel_but_click)
        self.add_stopband_but.clicked.connect(self.on_add_stopband_but_click)
        self.add_passband_but.clicked.connect(self.on_add_passband_but_click)
        self.plot_save_but.clicked.connect(self.on_plot_save_but_click)
        self.plot_load_but.clicked.connect(self.on_plot_load_but_click)

        # Connect the widgets
        self.res_save_but.clicked.connect(self.on_res_save_but_click)
        self.res_load_but.clicked.connect(self.on_res_load_but_click)
        self.res_reset_but.clicked.connect(self.on_res_reset_but_click)


        #not implemented yet
        self.first_coef_check = self.findChild(QCheckBox, 'first_coef_check')
        self.first_coef_check.stateChanged.connect(self.first_coef_check_changed)

    def first_coef_check_changed(self):
        if self.first_coef_check.isChecked():
            self.logger.plog("The feature \"first filter coefficient cannot be flipped\" is not implemented yet, your result might need a bit inversion at the first coefficient")
        else:
            self.logger.plog("The feature \"first filter coefficient cannot be flipped\" is not implemented yet, your result might need a bit inversion at the first coefficient")

    def order_changed(self):
        even = None
        order_current = int(self.order_upper_box.value())
        if order_current%2 == 0:
            even = True
        else:
            even = False

        diff = None

        if self.order_upper_last_value > order_current:
            diff = -1
        else:
            diff = 1
        
        if self.filter_type_drop.currentIndex() == 0 and not even:
            self.order_upper_box.setValue(order_current+diff)
        elif self.filter_type_drop.currentIndex() == 1 and even:
            self.order_upper_box.setValue(order_current+diff)
        elif self.filter_type_drop.currentIndex() == 2 and not even:
            self.order_upper_box.setValue(order_current+diff)
        elif self.filter_type_drop.currentIndex() == 3 and even:
            self.order_upper_box.setValue(order_current+diff)
        
        self.order_upper_last_value = self.order_upper_box.value()

    

    def on_cell_changed(self, row, column):
        # self.magnitude_plotter_table.blockSignals(True)
        print(f"Cell [{row}, {column}] changed")
        if self.magnitude_plotter_table.item(row, 0).text() == 'stopband':
            item = self.magnitude_plotter_table.item(row, column)
            item_text = item.text()
            if column == 2:
                item2 = self.magnitude_plotter_table.item(row, 3)
                if item2:
                    item2.setText(item_text)
            elif column == 3:
                item2 = self.magnitude_plotter_table.item(row, 2)
                if item2:
                    item2.setText(item_text)
        


    def add_row(self, table: QTableWidget, passband_flag: bool):
        if passband_flag:
            band_type = 'passband'
            gain = 1
        else:
            band_type = 'stopband'
            gain = 0

        # Get the current row count
        rowCount = table.rowCount()

        # Insert a new row
        table.insertRow(rowCount)

        table.setItem(rowCount, 0, QTableWidgetItem(str(band_type)))
        table.setItem(rowCount, 1, QTableWidgetItem(str(gain)))

        table.setItem(rowCount, 2, QTableWidgetItem(str(-20)))
        table.setItem(rowCount, 3, QTableWidgetItem(str(-20)))

        table.setItem(rowCount, 4, QTableWidgetItem(""))
        table.setItem(rowCount, 5, QTableWidgetItem(""))

        self.set_column_read_only_and_center_align(table, 0)



    
    def remove_row(self, table):
        selected_rows = set(index.row() for index in table.selectedIndexes())

        for row in sorted(selected_rows, reverse=True):
            table.removeRow(row)


    def on_res_save_but_click(self):
        self.res_load_saver.save_data()

    def on_res_load_but_click(self):
        print("Load button clicked!")
        if self.show_yes_no_dialog("Are you sure you want to Load something else? any unsaved data will be lost"):
            self.res_load_saver.load_data()
            self.valid_table_widget.last_max_key = -1
            self.invalid_table_widget.last_max_key = -1
            self.solver_runs_table_widget.last_max_key = -1
        else:
            return

    def on_res_reset_but_click(self):
        print("Reset button clicked!")
        if self.show_yes_no_dialog("Are you sure you want to reset the results? any unsaved data will be lost"):
            self.file_remover('result_valid.json')
            self.file_remover('result_invalid.json')
            self.file_remover('problem_description.json')
            self.result_valid_table.setRowCount(0)
            self.result_invalid_table.setRowCount(0)
            self.solver_runs_table.setRowCount(0)
            self.valid_table_widget.last_max_key = -1
            self.invalid_table_widget.last_max_key = -1
            self.solver_runs_table_widget.last_max_key = -1
            self.logger.plog("Results reset, deleted all json files")
        else:
            return
    
    def file_remover(self, file_path):
        try:
            os.remove(file_path)
            print(f"{file_path} has been deleted.")
        except FileNotFoundError:
            print(f"{file_path} does not exist.")
        except PermissionError:
            print(f"Permission denied: Cannot delete {file_path}.")
        except Exception as e:
            print(f"An error occurred: {e}")

        
    def widget_connect_after(self):
        #connect value changed
        self.sampling_rate_box.valueChanged.connect(lambda: self.magnitude_plotter.update_plotter(self.get_ui_data_dict()))


    def get_ui_data_dict(self):
        data_dict = {
        "sampling_rate": self.sampling_rate_box.value(),
        "flat_level": self.flatten_box.value(),
        "gaussian_width": self.gausian_slid.value(),
        "smoothing_kernel": self.smoothing_kernel_slid.value(),
        "selection_mode":self.selection_mode,
    }
        return data_dict

    def on_remove_sel_but_click(self):
        self.remove_row(self.magnitude_plotter_table)

    def on_add_stopband_but_click(self):
        self.add_row(self.magnitude_plotter_table, False)
        

    def on_add_passband_but_click(self):
        self.add_row(self.magnitude_plotter_table, True)

    

    def on_plot_save_but_click(self):
        xdata , middle_y , lower_y, upper_y  = self.magnitude_plotter.get_frequency_bounds(True)
        if xdata is None or middle_y is None or lower_y is None or upper_y is None:
            self.logger.plog("No data to save!")
            return
        xdata = np.array(xdata).tolist()
        middle_y = np.array(middle_y).tolist()
        lower_y = np.array(lower_y).tolist()
        upper_y = np.array(upper_y).tolist()

        data_dict = {
            'xdata' : xdata,
            'middle_y' : middle_y,
            'lower_y' : lower_y,
            'upper_y' : upper_y,
        }
        self.plot_load_saver.save_data(data_dict, True)
        


    def on_plot_load_but_click(self):
        loaded_data_dict = self.plot_load_saver.load_data(True)
        if loaded_data_dict is None:
            return
        self.magnitude_plotter.load_plot(self.get_ui_data_dict(), loaded_data_dict)
        self.logger.plog("Data loaded")
        self.logger.plog("Plot Generated")

    def set_column_read_only_and_center_align(self, table, column_index):
        """
        Set a specific column in a QTableWidget to be read-only and center-align all its values.
        """
        for row in range(table.rowCount()):
            item = table.item(row, column_index)
            # Set the flags to disable editing
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            # Align the text to the center

        for row in range(table.rowCount()):
            for column in range(table.columnCount()):
                item = table.item(row, column)
                # Align the text to the center
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

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
                    row_empty = False
                    if col == 0:
                        row_data.append(item.text())
                    else:
                        try:
                            row_data.append(float(item.text()))
                        except ValueError:
                            self.logger.plog(f"Element at [{row}, {col}] in \"Magnitude Plotter\" is not a valid float")
                            return

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
        else:
            print("mag data is valdiated")

        print(data)
        

        self.magnitude_plotter_data = copy.deepcopy(data)

        self.magnitude_plotter.initiate_plot(self.magnitude_plotter_data, self.get_ui_data_dict())

        self.logger.plog("Plot Generated")

        
    def validate_mag(self, data):
        for row in range(len(data)):
            for col in range(len(data[row])):
                if data[row][col] == "":
                    self.logger.plog(f"Element at [{row}, {col}] in \"Magnitude Plotter\" can't be empty: {data[row][col]}")
                    return 1
                try:
                    if col == 0:
                        continue
                    float(data[row][col])
                except ValueError:
                    self.logger.plog(f"Element at [{row}, {col}] in \"Magnitude Plotter\" is not a valid float: {data[row][col]}")
                    return 1
        return 0

    


    def on_selection_but_click(self):
        self.selection_mode = not self.selection_mode
        if self.selection_mode:
            self.selection_but.setText("Exit Selection Mode")
            self.logger.plog("Selection mode enabled")
        else:
            self.selection_but.setText("Selection Mode")
            self.logger.plog("Selection mode disabled")
        self.magnitude_plotter.update_plotter(self.get_ui_data_dict())

    def on_day_night_but_click(self):
        self.day_night = not self.day_night
        load_flag = False
        if self.day_night:
            self.day_night_but.setText("Black Background")
        else:
            self.day_night_but.setText("White Background")
            
        if self.magnitude_plotter.draggable_lines is not None:
            x, middle_y, lower_y, upper_y , history = self.magnitude_plotter.get_current_data()
            load_flag = True

        # Remove the old canvas and toolbar if they exist
        if self.canvas is not None:
            self.mag_layout.removeWidget(self.canvas)
            self.canvas.deleteLater()

        if self.toolbar is not None:
            self.mag_toolbar.removeWidget(self.toolbar)
            self.toolbar.deleteLater()

        # Recreate the magnitude plotter, canvas, and toolbar
        self.magnitude_plotter = MagnitudePlotter(self.day_night, self)
        self.canvas = FigureCanvas(self.magnitude_plotter.fig)

        # Add the new canvas back to the layout
        self.mag_layout.addWidget(self.canvas)

        # Create a new toolbar for the updated canvas
        self.toolbar = CustomNavigationToolbar(self.canvas, self)

        # Add the toolbar back to the toolbar layout
        self.mag_toolbar.addWidget(self.toolbar)
        print("Day night button clicked")
        print("Updating plot")

        if load_flag:
            self.magnitude_plotter.load_plot(self.get_ui_data_dict(), {'xdata': x, 'middle_y': middle_y, 'upper_y': upper_y, 'lower_y': lower_y}, history)

    def on_nyquist_freq_but_click(self):
        self.magnitude_plotter.update_plotter(self.get_ui_data_dict())
        if self.magnitude_plotter.sampling_to_nyquist():
            self.logger.plog("Plotter is already in Nyquist")


    def on_normal_freq_but_click(self):
        self.magnitude_plotter.update_plotter(self.get_ui_data_dict())
        if self.magnitude_plotter.nyquist_to_sampling():
            self.logger.plog("Plotter is already in half of sampling frequency")

    def on_reset_dot_but_click(self):
        if self.magnitude_plotter.reset_draggable_points():
            self.logger.plog("Plotter only has Nan values, no valid lines found!")
        else:
            self.logger.plog("Dots have been reset, don't lose it again!")

    def on_mirror_uptolo_but_click(self):
        self.magnitude_plotter.mirror_upper_to_lower()

    def on_mirror_lotoup_click(self):
        self.magnitude_plotter.mirror_lower_to_upper()

    def on_delete_select_but_click(self):
        self.magnitude_plotter.delete_selected()

    def on_flatten_but_click(self):
        self.magnitude_plotter.update_plotter(self.get_ui_data_dict(), False)
        self.magnitude_plotter.apply_flatten()

    def on_undo_but_click(self):
        self.magnitude_plotter.undo_plot()

    def on_redo_but_click(self):
        self.magnitude_plotter.redo_plot()


    def on_smoothen_but_click(self):
        self.magnitude_plotter.update_plotter(self.get_ui_data_dict(), False)
        self.magnitude_plotter.smoothen_plot()

    def continue_solver(self, data):
        if self.solver_is_running:
            self.logger.plog("Solver is currently running")
            return
        
        self.solver_is_running = True
        self.logger.plog("Continue Solving FIR problem...")


        initial_input_dictionary = data

        print(initial_input_dictionary)
        self.mediator = BackendMediator(initial_input_dictionary)

        # Connect signals to print output and errors
        self.mediator.log_message.connect(self.logger.plog)
        self.mediator.exception_message.connect(self.show_error_dialog)
        self.mediator.finished.connect(self.solver_run_done)

        # Start the mediator
        self.mediator.run()
    
    def quick_check_sat_but_click(self):
        if self.solver_is_running:
            self.logger.plog("Solver is currently running")
            return
        self.logger.plog("Quick check SAT running...")


        if self.gurobi_thread_box.value() == 0 and not self.gurobi_auto_thread_check.isChecked() and self.z3_thread_box.value() == 0 and self.pysat_thread_box.value() == 0:
            self.logger.plog("Solver threads can't be 0")
            return
            

        if not(self.plot_generated_flag):
            self.logger.plog("No bounds found, Please generate the bounds first in magnitude plotter!")
            return
        
        # Check if all necessary inputs are valid
        if self.input_validation_before_run() == 1:
            return
        
        self.solver_is_running = True
        #update input data
        xdata, upper_ydata, lower_ydata ,cutoffs_x, cutoffs_upper_ydata, cutoffs_lower_ydata = self.magnitude_plotter.get_frequency_bounds()
        print(f"cutoffs_x {cutoffs_x}, cutoffs_upper_ydata {cutoffs_upper_ydata}, cutoffs_lower_ydata {cutoffs_lower_ydata}")
        #validate data
        for i in range(len(xdata)):
            if np.isnan(upper_ydata[i]) and np.isnan(lower_ydata[i]):
                continue
            if not(np.isnan(upper_ydata[i])) and not(np.isnan(lower_ydata[i])):
                continue
            raise ValueError(f"Bounds is not the same at {i}, upper: {upper_ydata[i]} and lower: {lower_ydata[i]}. This is a bug, Contact Developer")
        
        ui_functionality = UIFunc(self)
        
        #do this if transition band interpolation is chosen
        if self.interpolate_transition_band_check.isChecked():
            upper_ydata, lower_ydata = ui_functionality.interpolate_transition_band()

        
        self.quick_check_dict = ui_functionality.solver_input_dict_generator(xdata, upper_ydata, lower_ydata, cutoffs_x, cutoffs_upper_ydata, cutoffs_lower_ydata)
    
        self.quick_check_mediator = AdditionalFuncMediator(self.quick_check_dict, 'quick_check_sat')

        self.quick_check_mediator.exception_message.connect(self.show_error_dialog)
        self.quick_check_mediator.result_signal.connect(self.quick_check_done)

        
        self.quick_check_mediator.start()
    
    def quick_check_done(self, result, _, __):
        self.logger.plog("Quick check done")
        self.solver_is_running = False
        
        if result == None:
            self.logger.plog("Quick check failed")
            self.quick_check_dict = None
            return
        print(result)
        if result['satisfiability'] == 'sat':
            self.logger.plog(f"Problem is satisfiable")
            if result['leak_flag'] == True:
                self.logger.plog(f"Problem is satisfiable but Given Frequency Accuracy has leakage, increase it!!!")
            else:
                self.logger.plog(f"Problem is satisfiable and Given Frequency Accuracy should be safe from leak")
            self.subwindow_quick_check = DetailsSubWindow(self.quick_check_dict, result)
            self.subwindow_quick_check.show()
        else:
            self.logger.plog(f"Problem is unsatisfiable")

        
        
        


    def auto_search(self):
        
        #update input data
        xdata, upper_ydata, lower_ydata ,cutoffs_x, cutoffs_upper_ydata, cutoffs_lower_ydata = self.magnitude_plotter.get_frequency_bounds()
        print(f"cutoffs_x {cutoffs_x}, cutoffs_upper_ydata {cutoffs_upper_ydata}, cutoffs_lower_ydata {cutoffs_lower_ydata}")
        #validate data
        for i in range(len(xdata)):
            if np.isnan(upper_ydata[i]) and np.isnan(lower_ydata[i]):
                continue
            if not(np.isnan(upper_ydata[i])) and not(np.isnan(lower_ydata[i])):
                continue
            raise ValueError(f"Bounds is not the same at {i}, upper: {upper_ydata[i]} and lower: {lower_ydata[i]}. Contact Developer")
        
        ui_functionality = UIFunc(self)
        
        #do this if transition band interpolation is chosen
        if self.interpolate_transition_band_check.isChecked():
            upper_ydata, lower_ydata = ui_functionality.interpolate_transition_band()

        
        self.auto_search_dict = ui_functionality.solver_input_dict_generator_auto(xdata, upper_ydata, lower_ydata, cutoffs_x, cutoffs_upper_ydata, cutoffs_lower_ydata)
    
        self.auto_param_mediator = AdditionalFuncMediator(self.auto_search_dict, 'automatic')

        self.auto_param_mediator.exception_message.connect(self.show_error_dialog)
        self.auto_param_mediator.result_signal.connect(self.auto_param_done)

        
        self.auto_param_mediator.start()

    def auto_param_done(self, best_target_result, best_filter_type, wordlength):
        if best_target_result == None:
            self.logger.plog("Auto parameter search failed")
            self.solver_is_running = False
            return
        best_target_result.update({
            'best_filter_type': best_filter_type+1,
            'wordlength': wordlength,
        })
        self.auto_search_dict.update(
            {
                'filter_type': best_filter_type,
            }
        )
        # Create and show the subwindow
        self.subwindow = AutomaticParameterResultSubWindow(best_target_result)
        self.subwindow.continue_signal.connect(self.handle_subwindow_continue)
        self.subwindow.cancel_signal.connect(self.handle_subwindow_cancel)
        self.subwindow.show()
        del self.auto_param_mediator

    def handle_subwindow_continue(self, filter_order, word_length):
        # Proceed with the solver using the given parameters
        print(f"Continuing with Filter Order: {filter_order}, Word Length: {word_length}")
        self.auto_search_dict.update(
            {
                'wordlength': word_length,
                'order_upperbound': filter_order
            }
        )
        print({f"self.auto_search_dict {self.auto_search_dict}"})
        self.mediator = BackendMediator(self.auto_search_dict)

        # Connect signals to print output and errors
        self.mediator.log_message.connect(self.logger.plog)
        self.mediator.exception_message.connect(self.show_error_dialog)
        self.mediator.finished.connect(self.solver_run_done)

        # Start the mediator
        self.mediator.run()

    def handle_subwindow_cancel(self):
        # Handle cancellation
        self.logger.plog("Auto parameter search canceled")
        self.solver_is_running = False
        self.auto_search_dict = None
   
    def on_run_solver_but_click(self):
        if self.auto_search_check.isChecked():
            if self.solver_is_running:
                self.logger.plog("Solver is currently running")
                return
            self.logger.plog("Auto search is enabled, Searching for the best parameters")


            if self.gurobi_thread_box.value() == 0 and not self.gurobi_auto_thread_check.isChecked() and self.z3_thread_box.value() == 0 and self.pysat_thread_box.value() == 0:
                self.logger.plog("Solver threads can't be 0")
                return
                

            if not(self.plot_generated_flag):
                self.logger.plog("No bounds found, Please generate the bounds first in magnitude plotter!")
                return
            
            # Check if all necessary inputs are valid
            if self.input_validation_before_run() == 1:
                return
            
            self.solver_is_running = True
            self.auto_search()
            return
        
        #solver running flag
        if self.solver_is_running:
            self.logger.plog("Solver is currently running")
            return
        

        if self.gurobi_thread_box.value() == 0 and not self.gurobi_auto_thread_check.isChecked() and self.z3_thread_box.value() == 0 and self.pysat_thread_box.value() == 0:
            self.logger.plog("Solver threads can't be 0")
            return
            

        if not(self.plot_generated_flag):
            self.logger.plog("No bounds found, Please generate the bounds first in magnitude plotter!")
            return
        
        
        self.logger.plog("Solving FIR problem...")
        
        # Check if all necessary inputs are valid
        if self.input_validation_before_run() == 1:
            return
        
        self.solver_is_running = True
        
        #update input data
        xdata, upper_ydata, lower_ydata ,cutoffs_x, cutoffs_upper_ydata, cutoffs_lower_ydata = self.magnitude_plotter.get_frequency_bounds()

        #validate data
        for i in range(len(xdata)):
            if np.isnan(upper_ydata[i]) and np.isnan(lower_ydata[i]):
                continue
            if not(np.isnan(upper_ydata[i])) and not(np.isnan(lower_ydata[i])):
                continue
            raise ValueError(f"Bounds is not the same at {i}, upper: {upper_ydata[i]} and lower: {lower_ydata[i]}. Contact Developer")
        
        ui_functionality = UIFunc(self)
        
        #do this if transition band interpolation is chosen
        if self.interpolate_transition_band_check.isChecked():
            upper_ydata, lower_ydata = ui_functionality.interpolate_transition_band()

        
        initial_input_dictionary = ui_functionality.solver_input_dict_generator(xdata, upper_ydata, lower_ydata, cutoffs_x, cutoffs_upper_ydata, cutoffs_lower_ydata)

        self.mediator = BackendMediator(initial_input_dictionary)

        # Connect signals to print output and errors
        self.mediator.log_message.connect(self.logger.plog)
        self.mediator.exception_message.connect(self.show_error_dialog)
        self.mediator.finished.connect(self.solver_run_done)

        # Start the mediator
        self.mediator.run()
        

    def kill_solver_instance(self):
        if self.mediator == None:
            self.logger.plog(f"Nothing is running currently!")
            return
        self.mediator.stop()
        self.solving_canceled = True
        self.logger.plog(f"Solving canceled")



    def solver_run_done(self):
        self.solver_is_running = False
        self.mediator = None
        if self.solving_canceled == True:
            #reset and return
            self.solving_canceled == False
            return
        self.logger.plog(f"Solving done! check result data.")
        
        

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
    
    def show_yes_no_dialog(self, msg_str):
        # Create a QMessageBox for Yes/No prompt
        reply = QMessageBox.question(self, 'Confirmation', msg_str,
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)

        if reply == QMessageBox.StandardButton.Yes:
            print("User selected Yes")
            return True
        else:
            print("User selected No")
            return False

    def show_error_dialog(self, error_msg_str):
        # Create a QMessageBox for exception or error
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setWindowTitle("An exception occurred!")
        msg_box.setText(error_msg_str)  # Set the actual error message string here
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)

        msg_box.exec()  # Display the pop-up dialog  
  




if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
