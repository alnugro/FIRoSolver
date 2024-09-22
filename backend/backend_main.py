import random
import time
import numpy as np
import copy
from pebble import ProcessPool
from concurrent.futures import TimeoutError  # Correct import for TimeoutError
import multiprocessing
from .solver_func import SolverFunc
from .error_predictor import ErrorPredictor
from .solver_presolve import OrderCompressor


class SolverBackend():
    def __init__(self, input_data):
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
        for key, value in input_data.items():
            if hasattr(self, key):  # Only set attributes that exist in the class
                setattr(self, key, value)
        
        #declare data that is local to backend
        self.input = input_data  

        #initiate different bounds, for error predictor later
        self.upperbound_gurobi_lin = None
        self.lowerbound_gurobi_lin = None
        
        self.upperbound_z3_lin = None
        self.lowerbound_z3_lin = None

        self.upperbound_pysat_lin = None
        self.lowerbound_pysat_lin = None


    
    def compress_solver_order(self):
        #interpolate original data first
        xdata, upperbound_lin, lowerbound_lin = self.interpolate_bounds_to_order(self.order_upperbound)

        #update input data with interpolated data
        self.input.update({
            'xdata' : xdata,
            'upperbound_lin': upperbound_lin,
            'lowerbound_lin':lowerbound_lin
        })
        #always run order compressor first
        compressor = OrderCompressor(self.input)
        if self.gurobi_thread > 0:
            
            #if gurobi is available then use gurobi, because it is way faster to find the minimum solver order
            min_order, h_res = compressor.run_order_compressor_gurobi()
        else:
            pass
        
            

    def gurobi_test(self):
        try:
            from gurobipy import Model, GRB
            
            # Create a new model
            model = Model("test_model")

            # Create variables
            x = model.addVar(name="x")
            y = model.addVar(name="y")

            # Set objective: maximize 3x + 4y
            model.setObjective(3 * x + 4 * y, GRB.MAXIMIZE)

            # Add constraint: x + 2y <= 14
            model.addConstr(x + 2 * y <= 14, "c1")

            # Optimize the model
            model.optimize()

            # Check if an optimal solution was found
            if model.status == GRB.OPTIMAL:
                return True
            else: return False

        except Exception as e:
            print(f"Gurobi encountered an error: {e}")
            raise ModuleNotFoundError(f"Gurobi encountered an error: {e}")

    def no_gurobi_presolve(self):
        pass

    def gurobi_presolve(self):
        pass

    def interpolate_bounds_to_order(self, order_current):
        # Ensure step is an integer
        self.step = int(order_current * self.solver_accuracy_multiplier)
        
        # Create xdata with self.step points between 0 and 1
        xdata = np.linspace(0, 1, self.step)

        # Interpolate upper and lower bounds to the user multiplier
        upper_ydata_lin = np.interp(xdata, self.original_xdata, self.original_upperbound_lin)
        lower_ydata_lin = np.interp(xdata, self.original_xdata, self.original_lowerbound_lin)

        for x_index, x in enumerate(self.cutoffs_x):
            if x in xdata:
                continue
            xdata_index = np.searchsorted(xdata, x)
            xdata = np.insert(xdata, xdata_index, x)
            upper_ydata_lin = np.insert(upper_ydata_lin, xdata_index, self.cutoffs_upper_ydata_lin[x_index])
            lower_ydata_lin = np.insert(lower_ydata_lin, xdata_index, self.cutoffs_lower_ydata_lin[x_index])

        return xdata, upper_ydata_lin, lower_ydata_lin

    
    
    # def run_backend(self):
        


    #     #run gurobi test to find out if its available
    #     if self.gurobi_thread > 0:
    #         self.gurobi_test()

    #     parallel_exec_instance = ParallelExecutor(self.input,
    #                                      self.upperbound_gurobi_lin, 
    #                                      self.lowerbound_gurobi_lin,
    #                                      self.upperbound_z3_lin, 
    #                                      self.lowerbound_z3_lin, 
    #                                      self.upperbound_pysat_lin,
    #                                      self.lowerbound_pysat_lin)

    #     #iterate the order from smallest to highest
    #     self.order_current = self.order_lower
    #     while self.order_current <= self.order_upper:
    #         print(f"current {self.order_current}")
    #         print(f"upper {self.order_upper}")
    #         if self.start_with_error_prediction:
    #             self.upperbound_gurobi_lin, self.lowerbound_gurobi_lin, self.upperbound_z3_lin, self.lowerbound_z3_lin, self.upperbound_pysat_lin, self.lowerbound_pysat_lin = parallel_exec_instance.execute_parallel_error_prediction(self.order_current)
    #         self.order_current += 1
                

if __name__ == "__main__":
    # Test inputs
    filter_type = 0
    order_lower = 4
    order_upper = 4
    accuracy = 1
    adder_count = 3
    wordlength = 14
    
    adder_depth = 2
    avail_dsp = 0
    adder_wordlength_ext = 2
    gain_upperbound = 3
    gain_lowerbound = 1
    coef_accuracy = 4
    intW = 4

    space = int(accuracy*order_upper)
    # Initialize freq_upper and freq_lower with NaN values
    freqx_axis = np.linspace(0, 1, space) #according to Mr. Kumms paper
    freq_upper = np.full(space, np.nan)
    freq_lower = np.full(space, np.nan)

    # Manually set specific values for the elements of freq_upper and freq_lower in dB
    lower_half_point = int(0.5*(space))
    upper_half_point = int(0.6*(space))
    end_point = space

    freq_upper[0:lower_half_point] = 3
    freq_lower[0:lower_half_point] = -1

    freq_upper[upper_half_point:end_point] = -20
    freq_lower[upper_half_point:end_point] = -1000


    #beyond this bound lowerbound will be ignored
    ignore_lowerbound = -40

    


    input_data = {
        'filter_type': filter_type,
        'order_upper': order_upper,
        'order_lower': order_lower,
        'freqx_axis': freqx_axis,
        'freq_upper': freq_upper,
        'freq_lower': freq_lower,
        'ignore_lowerbound': ignore_lowerbound,
        'adder_count': adder_count,
        'wordlength': wordlength,
        'adder_depth': adder_depth,
        'avail_dsp': avail_dsp,
        'adder_wordlength_ext': adder_wordlength_ext,
        'gain_wordlength' : 6,
        'gain_intW' : 2,
        'gain_upperbound': gain_upperbound,
        'gain_lowerbound': gain_lowerbound,
        'coef_accuracy': coef_accuracy,
        'intW': intW,
        'gurobi_thread': 1,
        'pysat_thread': 2,
        'z3_thread': 3,
        'timeout': 1000,
        'start_with_error_prediction': True
    }

    # Create an instance of SolverBackend
    solver_backend_instance = SolverBackend(input_data)
    solver_backend_instance.run_backend()
