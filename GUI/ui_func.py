import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QComboBox, QSpinBox, QTextEdit, QTableWidget, QTableWidgetItem, QWidget, QFrame
from PyQt6.QtCore import QThread, pyqtSignal, Qt
import sys
import os
import copy

try:
    from .live_logger import LiveLogger
    from backend.backend_main import SolverBackend

except:
    from live_logger import LiveLogger
    from backend.backend_main import SolverBackend



class UIFunc:
    def __init__(self, main_window):
        #to add on ui
        self.interpolate_transition_band_flag = True

        # Store reference to the main window
        self.main_window = main_window
        
        # self.ignore_lowerbound_lin = self.db_to_linear(self.main_window.ignore_lowerbound_box)

        self.logger = LiveLogger(self.main_window)




    def solver_input_dict_generator_re(self, xdata, upper_ydata, lower_ydata, cutoffs_x, cutoffs_upper_ydata, cutoffs_lower_ydata):
        input_data = {
        'filter_type': self.main_window.filter_type_drop.currentIndex(),
        'order_upperbound': self.main_window.order_upper_box.value(),
        'original_xdata': xdata,
        'original_upperbound_lin': upper_ydata,
        'original_lowerbound_lin': lower_ydata,
        'cutoffs_x': cutoffs_x,
        'cutoffs_upper_ydata_lin': cutoffs_upper_ydata,
        'cutoffs_lower_ydata_lin': cutoffs_lower_ydata,
        'ignore_lowerbound': self.db_to_linear(self.main_window.ignore_lowerbound_box.value()),
        'wordlength': self.main_window.wordlength_box.value(),
        'adder_depth': self.main_window.adder_depth_box.value(),
        'avail_dsp': self.main_window.available_dsp_box.value(),
        'adder_wordlength_ext': self.main_window.adder_wordlength_ext_box.value(),
        'gain_upperbound': self.main_window.gain_upper_box.value(),
        'gain_lowerbound': self.main_window.gain_lower_box.value(),
        'coef_accuracy': self.main_window.coef_accuracy_box.value(),
        'intW': self.main_window.integer_width_box.value(),
        'gurobi_thread': self.main_window.gurobi_thread_box.value(),
        'pysat_thread': self.main_window.pysat_thread_box.value(),
        'z3_thread': self.main_window.z3_thread_box.value(),
        'timeout': self.main_window.solver_timeout_box.value(),
        'start_with_error_prediction': self.main_window.start_with_error_prediction_check.isChecked(),
        'solver_accuracy_multiplier': self.main_window.solver_accuracy_multiplier_box.value(),
        'start_with_error_prediction': self.main_window.start_with_error_prediction_check.isChecked(),
        'deepsearch': self.main_window.deepsearch_check.isChecked(),
        'patch_multiplier' : self.main_window.patch_multiplier_box.value(),
        'gurobi_auto_thread': self.main_window.gurobi_auto_thread_check.isChecked()
        }
        return input_data
    
    def solver_input_dict_generator(self):
        input_data = {
        'filter_type': 0,
        'order_upperbound': 30,
        'original_xdata': self.original_xdata,
        'original_upperbound_lin': self.upperbound_lin,
        'original_lowerbound_lin': self.lowerbound_lin,
        'ignore_lowerbound': 0.1,
        'cutoffs_x': self.cutoffs_x,
        'cutoffs_upper_ydata_lin': self.cutoffs_upper_ydata_lin,
        'cutoffs_lower_ydata_lin': self.cutoffs_lower_ydata_lin,
        'wordlength': 15,
        'adder_depth': 0,
        'avail_dsp': 0,
        'adder_wordlength_ext': 2, #this is extension not the adder wordlength
        'gain_wordlength' : 6,
        'gain_intW' : 2,
        'gain_upperbound': 3,
        'gain_lowerbound': 1,
        'coef_accuracy': 6,
        'intW': 6,
        'gurobi_thread': 5,
        'pysat_thread': 5,
        'z3_thread': 4,
        'timeout': 0,
        'start_with_error_prediction': False,
        'solver_accuracy_multiplier': 9,
        'deepsearch': True,
        'patch_multiplier' : 1,
        'gurobi_auto_thread': False
        }
        return input_data
    
    def delete_json_files(self):
        pass

    def interpolate_transition_band(self):
        last_entry = []
        first_entry = []

        upperbound_lin_t_interp = np.copy(self.upperbound_lin)
        lowerbound_lin_t_interp = np.copy(self.lowerbound_lin)
        
        for i in range(len(self.original_xdata)):
            if not(np.isnan(self.upperbound_lin[i])) and np.isnan(self.upperbound_lin[i+1]):
                last_entry.append(i)
            if not(np.isnan(self.upperbound_lin[i])) and np.isnan(self.upperbound_lin[i-1]):
                first_entry.append(i)
        
        # ignore the first and last nan
        if last_entry[-1] >  first_entry[-1]:
            del last_entry[-1]

        if first_entry[0] < last_entry[0] :
            del first_entry[0]

        #if no nan is in transition band return
        if not(first_entry) or not(last_entry):
            return
        
        for i in range(len(last_entry)):
            upperbound_lin_t_interp[last_entry[i]:first_entry[i]] = self.upperbound_lin[last_entry[i]]
            lowerbound_lin_t_interp[last_entry[i]:first_entry[i]] = self.lowerbound_lin[first_entry[i]]

        return upperbound_lin_t_interp,lowerbound_lin_t_interp
        
    
    def db_to_linear(self,value):
        linear_value = 10 ** (value / 20)
        return linear_value

    
