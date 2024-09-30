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
    def __init__(self, main_window ,xdata, upper_ydata, lower_ydata, cutoffs_x, cutoffs_upper_ydata, cutoffs_lower_ydata):
        #to add on ui
        self.interpolate_transition_band_flag = True

        # Store reference to the main window
        self.main_window = main_window
        self.order_upper = self.main_window.order_upper_box
        self.solver_accuracy_multiplier = self.main_window.solver_accuracy_multiplier_box

        self.original_xdata = xdata
        self.original_upperbound = upper_ydata
        self.original_lowerbound = lower_ydata

        self.cutoffs_x = cutoffs_x
        self.cutoffs_upper_ydata = cutoffs_upper_ydata
        self.cutoffs_lower_ydata = cutoffs_lower_ydata

        #linearize all input
        self.upperbound_lin = [np.array(self.db_to_linear(f)).item() if not np.isnan(f) else np.nan for f in self.original_upperbound]
        self.lowerbound_lin = [np.array(self.db_to_linear(f)).item()  if not np.isnan(f) else np.nan for f in self.original_lowerbound]

        self.cutoffs_upper_ydata_lin = [np.array(self.db_to_linear(f)).item() if not np.isnan(f) else np.nan for f in self.cutoffs_upper_ydata] 
        self.cutoffs_lower_ydata_lin= [np.array(self.db_to_linear(f)).item()  if not np.isnan(f) else np.nan for f in self.cutoffs_lower_ydata]

        self.ignore_lowerbound_lin = self.db_to_linear(self.main_window.ignore_lowerbound_box)

        self.logger = LiveLogger(self.main_window)




    def solver_input_dict_generator_re(self):
        input_data = {
        'filter_type': self.main_window.filter_type_drop.currentIndex(),
        'order_upperbound': self.main_window.order_upper_box.value(),
        'original_xdata': self.original_xdata,
        'original_upperbound_lin': self.upperbound_lin,
        'original_lowerbound_lin': self.lowerbound_lin,
        'cutoffs_x': self.cutoffs_x,
        'cutoffs_upper_ydata_lin': self.cutoffs_upper_ydata_lin,
        'cutoffs_lower_ydata_lin': self.cutoffs_lower_ydata_lin,
        'ignore_lowerbound': self.ignore_lowerbound_lin,
        'wordlength': self.main_window.wordlength_box.value(),
        'adder_depth': self.main_window.adder_depth_box.value(),
        'avail_dsp': self.main_window.available_dsp_box.value(),
        'adder_wordlength_ext': self.main_window.adder_wordlength_ext_box.value(),
        'gain_wordlength' : self.main_window.gain_wordlength_box.value(),
        'gain_intW' : self.main_window.gain_integer_width_box.value(),
        'gain_upperbound': self.main_window.gain_upperbound_box.value(),
        'gain_lowerbound': self.main_window.gain_lowerbound_box.value(),
        'coef_accuracy': self.main_window.coef_accuracy_box.value(),
        'intW': self.main_window.integer_width_box.value(),
        'gurobi_thread': self.main_window.gurobi_thread_box.value(),
        'pysat_thread': self.main_window.pysat_thread_box.value(),
        'z3_thread': self.main_window.z3_thread_box.value(),
        'timeout': self.main_window.solver_timeout_box.value(),
        'start_with_error_prediction': self.start_with_error_prediction,
        'solver_accuracy_multiplier': self.solver_accuracy_multiplier,
        'start_with_error_prediction': False,
        'solver_accuracy_multiplier': accuracy,
        'deepsearch': True,
        'patch_multiplier' : 1,
        'gurobi_auto_thread': False
        }
        return input_data
    
    def solver_input_dict_generator(self):
        input_data = {
        'filter_type': 0,
        'order_upperbound': self.main_window.order_upper_box.value(),
        'original_xdata': self.original_xdata,
        'original_upperbound_lin': self.upperbound_lin,
        'original_lowerbound_lin': self.lowerbound_lin,
        'ignore_lowerbound': self.ignore_lowerbound_lin,
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
        'gurobi_thread': 4,
        'pysat_thread': 0,
        'z3_thread': 0,
        'timeout': 0,
        'start_with_error_prediction': False,
        'solver_accuracy_multiplier': 9,
        'deepsearch': True,
        'patch_multiplier' : 1,
        'gurobi_auto_thread': False
        }
        return input_data
    
    

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

    
