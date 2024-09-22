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
from .live_logger import LiveLogger
from .ui_backend_mediator import BackendMediator

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
        self.interpolate_transition_band_but = False
        self.start_with_error_prediction_but = True
        self.solver_accuracy_multiplier_box = 6 #according to Mr. Kumm this should be 16
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
        self.start_with_error_prediction_but = True

        


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

        self.magnitude_plotter_table = self.findChild(QTableWidget, 'magnitude_plotter_table')
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

    # def solver_get_input_data(self):
    #     # Assigning variables to buffer variables first
    #     # filter_type = self.filter_type_drop.currentIndex()
    #     filter_type = 0
    #     order_current = 0  # Example buffer value, adjust based on your logic
    #     ignore_lowerbound = True
    #     adder_count = 4
    #     # wordlength = self.wordlength_box.value()
    #     wordlength = 12
    #     adder_depth = 3
    #     avail_dsp = 2
    #     adder_wordlength_ext = 1
    #     gain_upperbound = 1.5
    #     gain_lowerbound = 0.5
    #     coef_accuracy = 0.001
    #     intW = 4
    #     gain_wordlength = 8
    #     gain_intW = 3
    #     gurobi_thread = 4
    #     pysat_thread = 2
    #     z3_thread = 2
    #     timeout = 300  # Timeout in seconds
    #     max_iteration = 1000
    #     start_with_error_prediction = False

    #     # Updating the input dictionary using buffer variables
    #     self.input.update({
    #         'filter_type': filter_type,
    #         'order_current': order_current,
    #         'ignore_lowerbound': ignore_lowerbound,
    #         'adder_count': adder_count,
    #         'wordlength': wordlength,
    #         'adder_depth': adder_depth,
    #         'avail_dsp': avail_dsp,
    #         'adder_wordlength_ext': adder_wordlength_ext,
    #         'gain_upperbound': gain_upperbound,
    #         'gain_lowerbound': gain_lowerbound,
    #         'coef_accuracy': coef_accuracy,
    #         'intW': intW,
    #         'gain_wordlength': gain_wordlength,
    #         'gain_intW': gain_intW,
    #         'gurobi_thread': gurobi_thread,
    #         'pysat_thread': pysat_thread,
    #         'z3_thread': z3_thread,
    #         'timeout': timeout,
    #         'max_iteration': max_iteration,
    #         'start_with_error_prediction': start_with_error_prediction
    #     })
        
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
        
        self.mediator  = BackendMediator(self, xdata, upper_ydata, lower_ydata, cutoffs_x, cutoffs_upper_ydata, cutoffs_lower_ydata)
        
        #do this if transition band interpolation is chosen
        if self.interpolate_transition_band_but:
            self.mediator.interpolate_transition_band()

        self.mediator.start_solver()



        # print(f"xdata {xdata}")
        # print(f"upper {upper_ydata}")
        # print(f"lower {lower_ydata}")
        # self.solver.update_plotter_data(self.get_ui_data_dict())

        # Set bounds in solver from plotter
        # self.solver.set_input_arg(self.magnitude_plotter.get_frequency_bounds())

        # try:    
        #     future = self.executor.submit(self.solver.run_solver)
        #     future.add_done_callback(self.solver_done)

        # except Exception as e:
        #     self.logger.plog(f"Backend Error: {e}")
        #     print(f"Backend Error: {e}")
        #     traceback.print_exc()



    # def solver_done(self, future):
    #     self.solver_is_running = False
    #     self.logger.plog("Solving Done, see Result!")

    #     try:
    #         # Check if the future resulted in an exception
    #         exception = future.exception()
    #         if exception:
    #             raise exception

    #         # Retrieve the result from the solver
    #         result = self.solver.get_result()

    #         # Plot the result using MagnitudePlotter
    #         self.magnitude_plotter.plot_result(result)

    #     except Exception as e:
    #         # Capture and print detailed exception information
    #         self.logger.plog(f"Error during result processing: {e}")
    #         print(f"Error during result processing: {e}")
    #         traceback.print_exc()
        

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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
