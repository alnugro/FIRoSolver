import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QComboBox, QSpinBox, QTextEdit, QTableWidget, QTableWidgetItem, QWidget, QFrame
from PyQt6.QtCore import QThread, pyqtSignal, Qt
import sys
import os

try:
    from .live_logger import LiveLogger
    from backend.backend_main import SolverBackend

except:
    from live_logger import LiveLogger
    from backend.backend_main import SolverBackend



class BackendMediator(QThread):
    log_message = pyqtSignal(str)
    exception_message = pyqtSignal(str)
    # presolve_done = pyqtSignal(tuple)

    def __init__(self ,initial_solver_input):
        super().__init__()

        # Explicit declaration of instance variables with default values (if applicable)
        self.filter_type = None
        self.order_upperbound = None

        self.ignore_lowerbound = None
        self.wordlength = None
        self.adder_depth = None
        self.avail_dsp = None
        self.adder_wordlength_ext = None
        self.gain_upperbound = None
        self.gain_lowerbound = None
        self.coef_accuracy = None
        self.intW = None

        self.gain_wordlength = None
        self.gain_intW = None

        self.gurobi_thread = None
        self.pysat_thread = None
        self.z3_thread = None

        self.timeout = None
        self.start_with_error_prediction = None

        self.original_xdata = None
        self.original_upperbound_lin = None
        self.original_lowerbound_lin = None

        self.cutoffs_x = None
        self.cutoffs_upper_ydata_lin = None
        self.cutoffs_lower_ydata_lin = None

        self.solver_accuracy_multiplier = None

        # Dynamically assign values from input_data, skipping any keys that don't have matching attributes
        for key, value in initial_solver_input.items():
            if hasattr(self, key):  # Only set attributes that exist in the class
                setattr(self, key, value)
            

        self.initial_solver_input = initial_solver_input

    
    def run(self):
        try:
            #initiate backend
            backend = SolverBackend(self.initial_solver_input)

            if self.gurobi_thread > 0:
                self.log_message.emit(f"Gurobi is chosen, running compatibility test")            
                result = backend.gurobi_test()
            self.log_message.emit(f"Gurobi compatibility test done")

            self.log_message.emit(f"Running presolve")
            #start presolve
            presolve_result_gurobi, presolve_result_z3 = backend.solver_presolve()
            self.log_message.emit(f"Presolve done")


        except ValueError as e:
                if str(e) == "problem is unsat":
                     self.log_message.emit(f"Given problem is unsat")
        except Exception as e:
            print(f"{e}")
            self.exception_message.emit(f"{e}")







