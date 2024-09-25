import random
import time
import numpy as np
import copy
from pebble import ProcessPool
from concurrent.futures import TimeoutError  # Correct import for TimeoutError
import multiprocessing
try:
    from .solver_func import SolverFunc
    from .bound_error_handler import BoundErrorHandler
    from .solver_presolve import Presolver
except:
    from solver_func import SolverFunc
    from bound_error_handler import BoundErrorHandler
    from solver_presolve import Presolver


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

        self.sf = SolverFunc(self.input)

    def db_to_linear(self,value):
        linear_value = 10 ** (value / 20)
        return linear_value



    
    def solver_presolve(self):
        #interpolate original data first
        
        xdata, upperbound_lin, lowerbound_lin = self.sf.interpolate_bounds_to_order(self.order_upperbound)

        #update input data with interpolated data
        self.input.update({
            'xdata' : xdata,
            'upperbound_lin': upperbound_lin,
            'lowerbound_lin':lowerbound_lin
        })

        presolve_result_gurobi = None
        presolve_result_z3 = None

        #always run presolver first
        presolver = Presolver(self.input)
        if self.gurobi_thread > 0:
            #if gurobi is available then use gurobi, because it is way faster to find the minimum solver order and can be used to find minmax variables
            presolve_result_gurobi = presolver.run_presolve_gurobi()
        else:
            presolve_result_z3 = presolver.run_presolve_z3_pysat()
        
        return presolve_result_gurobi, presolve_result_z3
            

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
                print("\nGurobi Test completed.......\n")

        
            else: 
                raise ImportError("Gurobi is somehow broken, simple test should be sat: probably contact Gurobi")

            

        except Exception as e:
            raise ImportError(f"Gurobi encountered an error: {e}")



    

    
    
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
    order_current = 10
    accuracy = 1
    adder_count = 3
    wordlength = 10
    
    adder_depth = 2
    avail_dsp = 0
    adder_wordlength_ext = 2
    gain_upperbound = 4
    gain_lowerbound = 1
    coef_accuracy = 4
    intW = 4

    space = 400
    # Initialize freq_upper and freq_lower with NaN values
    freqx_axis = np.linspace(0, 1, space) #according to Mr. Kumms paper
    freq_upper = np.full(space, np.nan)
    freq_lower = np.full(space, np.nan)

    # Manually set specific values for the elements of freq_upper and freq_lower in dB
    lower_half_point = int(0.4*(space))
    upper_half_point = int(0.6*(space))
    end_point = space

    freq_upper[0:lower_half_point] = 21
    freq_lower[0:lower_half_point] = -19

    freq_upper[upper_half_point:end_point] = -30
    freq_lower[upper_half_point:end_point] = -1000


    cutoffs_x = []
    cutoffs_upper_ydata = []
    cutoffs_lower_ydata = []

    cutoffs_x.append(freqx_axis[0])
    cutoffs_x.append(freqx_axis[lower_half_point-1])
    cutoffs_x.append(freqx_axis[upper_half_point])
    cutoffs_x.append(freqx_axis[end_point-1])

    cutoffs_upper_ydata.append(freq_upper[0])
    cutoffs_upper_ydata.append(freq_upper[lower_half_point-1])
    cutoffs_upper_ydata.append(freq_upper[upper_half_point])
    cutoffs_upper_ydata.append(freq_upper[end_point-1])

    cutoffs_lower_ydata.append(freq_lower[0])
    cutoffs_lower_ydata.append(freq_lower[lower_half_point-1])
    cutoffs_lower_ydata.append(freq_lower[upper_half_point])
    cutoffs_lower_ydata.append(freq_lower[end_point-1])


    #beyond this bound lowerbound will be ignored
    ignore_lowerbound = -40

    input_data_sf = {
        'filter_type': filter_type,
        'order_upperbound': order_current,
        'original_xdata': freqx_axis,
        'cutoffs_x': cutoffs_x,
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
        'gurobi_thread': 1,
        'pysat_thread': 0,
        'z3_thread': 1,
        'timeout': 0,
        'start_with_error_prediction': False,
        'solver_accuracy_multiplier': 6,
    }


    sf = SolverFunc(input_data_sf)
    upperbound_lin = [np.array(sf.db_to_linear(f)).item() if not np.isnan(f) else np.nan for f in freq_upper]
    lowerbound_lin = [np.array(sf.db_to_linear(f)).item()  if not np.isnan(f) else np.nan for f in freq_lower]
    ignore_lowerbound_np = np.array(ignore_lowerbound, dtype=float)
    ignore_lowerbound_lin = sf.db_to_linear(ignore_lowerbound_np)

    cutoffs_upper_ydata_lin = [np.array(sf.db_to_linear(f)).item() if not np.isnan(f) else np.nan for f in cutoffs_upper_ydata]
    cutoffs_lower_ydata_lin = [np.array(sf.db_to_linear(f)).item() if not np.isnan(f) else np.nan for f in cutoffs_lower_ydata]



    input_data = {
        'filter_type': filter_type,
        'order_upperbound': order_current,
        'original_xdata': freqx_axis,
        'original_upperbound_lin': upperbound_lin,
        'original_lowerbound_lin': lowerbound_lin,
        'ignore_lowerbound': ignore_lowerbound_lin,
        'cutoffs_x': cutoffs_x,
        'cutoffs_upper_ydata_lin': cutoffs_upper_ydata_lin,
        'cutoffs_lower_ydata_lin': cutoffs_lower_ydata_lin,
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
        'gurobi_thread': 0,
        'pysat_thread': 0,
        'z3_thread': 1,
        'timeout': 0,
        'start_with_error_prediction': False,
        'solver_accuracy_multiplier': 6,
    }

    # Create an instance of SolverBackend
    solver_backend_instance = SolverBackend(input_data)
    solver_backend_instance.solver_presolve()
    # solver_backend_instance.gurobi_test()
