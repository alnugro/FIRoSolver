from pebble import ProcessPool, ProcessExpired, ThreadPool
from concurrent.futures import TimeoutError, CancelledError, wait, ALL_COMPLETED
import multiprocessing
import traceback
import time
import copy
import numpy as np
from filelock import FileLock
import json
from collections import OrderedDict
import math
import threading
import os
from functools import partial
import ast



try:
    from .formulation_pysat import FIRFilterPysat
    from .formulation_z3_pbsat import FIRFilterZ3
    from .formulation_gurobi import FIRFilterGurobi
    from .solver_func import SolverFunc

except ImportError:
    from formulation_pysat import FIRFilterPysat
    from formulation_z3_pbsat import FIRFilterZ3
    from formulation_gurobi import FIRFilterGurobi
    from solver_func import SolverFunc


class AutomaticSearch:
    def __init__(self, input_data):
        # AS search
        self.worker = None
        self.search_step = None
        self.problem_id = None
        self.continue_solver = None
        

        # Explicit declaration of instance variables with default values (if applicable)
        self.filter_type = None
        self.order_upperbound = None

        self.wordlength = None
        
        self.ignore_lowerbound = None
        self.gain_upperbound = None
        self.gain_lowerbound = None
        self.coef_accuracy = None
        self.intW = None

        self.gain_wordlength = None
        self.gain_intW = None

        self.gurobi_thread = None
        self.z3_thread = None
        self.pysat_thread = None

        self.timeout = None
        self.start_with_error_prediction = None

        self.xdata = None
        self.upperbound_lin = None
        self.lowerbound_lin = None

        self.adder_count = None
        self.adder_depth = None
        self.avail_dsp = None
        self.adder_wordlength_ext = None
        self.am_start = None
        self.half_order = None

        self.asserted_wordlength = None
        self.real_wordlength = None

        # Dynamically assign values from input_data, skipping any keys that don't have matching attributes
        for key, value in input_data.items():
            if hasattr(self, key):  # Only set attributes that exist in the class
                setattr(self, key, value)

        


    def automatic_search(self):
        # Find the best half order
        target_result = self.find_half_order()
        half_order = target_result['half_order_best']
        # Find the best wordlength
        if self.real_wordlength:
            target_result, wordlength = self.find_real_wordlength(half_order)
        else:
            wordlength = self.wordlength
            target_result, half_order = self.find_half_order_asserted(half_order, wordlength)

        # Find the best filter type
        best_target_result, best_filter_type, filter_order_best = self.find_best_filter_type(half_order, wordlength)
        print(f"Best filter type: {best_filter_type}, best filter type order: {filter_order_best}, target result: {best_target_result}, wordlength: {wordlength}")
        print(f"asserted_wordlength: {self.asserted_wordlength}, real_wordlength: {self.real_wordlength}")
        return best_target_result, best_filter_type, wordlength

    def try_asserted_real(self , half_order, filter_type):
        gurobi_instance = FIRFilterGurobi(
            filter_type, 
            half_order,
            self.xdata, 
            self.upperbound_lin, 
            self.lowerbound_lin, 
            self.ignore_lowerbound, 
            1, 
            100, 
            1,
            1,
            1,
            self.gain_upperbound,
            self.gain_lowerbound,
            self.coef_accuracy,
            self.intW)
        
        target_result = gurobi_instance.run_barebone_real(self.gurobi_thread, None ,None, None)
        satisfiability_loc = target_result['satisfiability']
        return target_result, satisfiability_loc
    
    def try_asserted(self, half_order, filter_type, wordlength, solver_options = None, h_zero = None):
        gurobi_instance = FIRFilterGurobi(
            filter_type, 
            half_order,
            self.xdata, 
            self.upperbound_lin, 
            self.lowerbound_lin, 
            self.ignore_lowerbound, 
            1, 
            wordlength, 
            1,
            1,
            1,
            self.gain_upperbound,
            self.gain_lowerbound,
            self.coef_accuracy,
            self.intW)
        
        target_result = gurobi_instance.run_barebone(self.gurobi_thread, solver_options ,h_zero)
        satisfiability_loc = target_result['satisfiability']
        return target_result, satisfiability_loc

    
    def find_half_order(self):
        half_order = 0
        while True:
            print(f"\nTrying half order: {half_order}, in find_half_order")
            target_result, satisfiability_loc = self.try_asserted_real(half_order, 0)
            if satisfiability_loc == 'sat':
                break
            half_order += 1
        target_result.update({'half_order_best': half_order})
        print(f"half_order_best: {half_order}")
        return target_result
    
    def find_half_order_asserted(self, half_order, wordlength):
        half_order = half_order
        max_half_order = 400
        while True:
            print(f"\nTrying half order: {half_order} in find_half_order_asserted")
            target_result, satisfiability_loc = self.try_asserted(half_order, 0, wordlength)
            if satisfiability_loc == 'sat':
                break
            if half_order >= max_half_order:
                raise Exception(f"half order is larger than {max_half_order}, this is too high, there should not be any solution from the given wordlength. If you think this is not the case, change max_half_order in automatic_search.py")
            half_order += 1
        target_result.update({'half_order_best': half_order})
        print(f"half_order_best: {half_order}")
        return target_result, half_order
    
    def find_real_wordlength(self, half_order):
        wordlength = self.intW + 1
        while True:
            print(f"\nTrying wordlength: {wordlength} in find_real_wordlength")
            target_result, satisfiability_loc = self.try_asserted(half_order, 0, wordlength)
            if satisfiability_loc == 'sat':
                break
            wordlength += 1
        return target_result, wordlength
    
    def find_best_filter_type(self, half_order, wordlength):
        filter_type = [0,1,2,3]

        half_order_nozer_best = [None, None, None, None]
        half_order_best = [half_order, half_order, half_order, half_order]
        best_target_result = [None, None, None, None]
    
        
        for i in range(4):
            print(f"\nTrying filter type: {i}")
            target_result, satisfiability_loc = self.try_asserted(half_order_best[i], i,wordlength ,'find_max_zero')
            if satisfiability_loc == 'sat':
                best_target_result[i] = target_result
                half_order_nozer_best[i] = half_order_best[i]-target_result['max_h_zero']
                print(f"\nhalf_order_nozer_best: {half_order_nozer_best[i]}")
            else:
                half_order_nozer_best[i] = None
                best_target_result[i] = None
                half_order_best[i] = None
        
        for i in range(4):
            while True:
                if half_order_best[i] == None:
                    break
                half_order_best[i] = half_order_best[i] - 1
                target_result, satisfiability_loc = self.try_asserted(half_order_best[i], i,wordlength,'find_max_zero')
                if satisfiability_loc == 'sat':
                    best_target_result[i] = target_result
                    half_order_nozer_best[i] = half_order_best[i]-target_result['max_h_zero']
                else:
                    half_order_best[i] = half_order_best[i] + 1
                    break
                    
        best_half_order_nozer_glob = min([x for x in half_order_nozer_best if x is not None])
        best_filter_type = half_order_nozer_best.index(best_half_order_nozer_glob)
        best_half_order_glob = half_order_best[best_filter_type]
        filter_order_best = 0

        if best_filter_type == 0:
            filter_order_best = (best_half_order_glob-1)*2
        elif best_filter_type == 2:
            filter_order_best = best_half_order_glob*2
        else:
            filter_order_best = best_half_order_glob*2-1

        best_target_result[best_filter_type].update({'filter_order': filter_order_best})

        return best_target_result[best_filter_type], best_filter_type, filter_order_best
    

    #for future suport of other solvers
    def z3_instance_creator(self):
        z3_instance = FIRFilterZ3(
                    self.filter_type, 
                    self.half_order, 
                    self.xdata, 
                    self.upperbound_lin, 
                    self.lowerbound_lin, 
                    self.ignore_lowerbound, 
                    self.adder_count, 
                    self.wordlength, 
                    self.adder_depth,
                    self.avail_dsp,
                    self.adder_wordlength_ext,
                    self.gain_upperbound,
                    self.gain_lowerbound,
                    self.coef_accuracy,
                    self.intW,
                    self.gain_wordlength,
                    self.gain_intW
                    )
        
        return z3_instance
    


if __name__ == "__main__":
    # Test inputs
    filter_type = 0
    order_current = 7
    accuracy = 4
    wordlength = 5
    gain_upperbound = 1
    gain_lowerbound = 1
    coef_accuracy = 5
    intW = 1

    adder_count = 4
    adder_depth = 0
    avail_dsp = 0
    adder_wordlength_ext = 4

    gain_wordlength = 13
    gain_intW = 4

    gurobi_thread = 5
    pysat_thread = 0
    z3_thread = 0

    timeout = 0


    passband_error = 0.030034
    stopband_error = 0.030034
    space = order_current * accuracy
    # Initialize freq_upper and freq_lower with NaN values
    freqx_axis = np.linspace(0, 1, space)
    freq_upper = np.full(space, np.nan)
    freq_lower = np.full(space, np.nan)

    # Manually set specific values for the elements of freq_upper and freq_lower in dB
    lower_half_point = int(0.3 * space)
    upper_half_point = int(0.5 * space)
    end_point = space

    freq_upper[0:lower_half_point] = 1 + passband_error
    freq_lower[0:lower_half_point] = 1 - passband_error

    freq_upper[upper_half_point:end_point] = 0 + stopband_error
    freq_lower[upper_half_point:end_point] = 0

    cutoffs_x = []
    cutoffs_upper_ydata = []
    cutoffs_lower_ydata = []

    cutoffs_x.append(freqx_axis[0])
    cutoffs_x.append(freqx_axis[lower_half_point - 1])
    cutoffs_x.append(freqx_axis[upper_half_point])
    cutoffs_x.append(freqx_axis[end_point - 1])

    cutoffs_upper_ydata.append(freq_upper[0])
    cutoffs_upper_ydata.append(freq_upper[lower_half_point - 1])
    cutoffs_upper_ydata.append(freq_upper[upper_half_point])
    cutoffs_upper_ydata.append(freq_upper[end_point - 1])

    cutoffs_lower_ydata.append(freq_lower[0])
    cutoffs_lower_ydata.append(freq_lower[lower_half_point - 1])
    cutoffs_lower_ydata.append(freq_lower[upper_half_point])
    cutoffs_lower_ydata.append(freq_lower[end_point - 1])

    # Beyond this bound, lowerbound will be ignored
    ignore_lowerbound = -60

    # Linearize the bounds
    upperbound_lin = np.copy(freq_upper)
    lowerbound_lin = np.copy(freq_lower)
    ignore_lowerbound_lin = 10 ** (ignore_lowerbound / 20)

    cutoffs_upper_ydata_lin = np.copy(cutoffs_upper_ydata)
    cutoffs_lower_ydata_lin = np.copy(cutoffs_lower_ydata)


    input_data = {
        'filter_type': filter_type,
        'half_order': order_current,
        'xdata': freqx_axis,
        'upperbound_lin': upperbound_lin,
        'lowerbound_lin': lowerbound_lin,
        'ignore_lowerbound': ignore_lowerbound_lin,
        'cutoffs_x': cutoffs_x,
        'cutoffs_upper_ydata_lin': cutoffs_upper_ydata_lin,
        'cutoffs_lower_ydata_lin': cutoffs_lower_ydata_lin,
        'wordlength': wordlength,
        'adder_depth': adder_depth,
        'avail_dsp': avail_dsp,
        'adder_wordlength_ext': adder_wordlength_ext, #this is extension not the adder wordlength
        'gain_wordlength' : gain_wordlength,
        'gain_intW' : gain_intW,
        'gain_upperbound': gain_upperbound,
        'gain_lowerbound': gain_lowerbound,
        'coef_accuracy': coef_accuracy,
        'intW': intW,
        'gurobi_thread': gurobi_thread,
        'pysat_thread': pysat_thread,
        'z3_thread': z3_thread,
        'timeout': 0,
        'start_with_error_prediction': False,
        'solver_accuracy_multiplier': 6,
        'asserted_wordlength': True,
        'real_wordlength': True,

    }

    automatic_search = AutomaticSearch(input_data)
    best_target_result, best_filter_type, wordlength = automatic_search.automatic_search()
    #Best filter type: 1, best filter type order: 7, target result: {'satisfiability': 'sat', 'h_res': [0.375, 0.1875, 0.0, -0.0625, -0.03125], 'max_h_zero': 1.0, 'gain_res': 1.0}, wordlength: 6