import random
import time
import numpy as np
import copy
from pebble import ProcessPool
from concurrent.futures import TimeoutError  # Correct import for TimeoutError
import multiprocessing
from solver_func import SolverFunc
from parallel_executor import ParallelExecutor

class SolverBackend():
    def __init__(self, input):
        self.filter_type = input['filter_type']
        self.order_upper = input['order_upper']
        self.order_lower = input['order_lower']
        self.order_current = 0

        self.freqx_axis = np.array(input['freqx_axis'], dtype=np.float64)
        self.freq_upper= np.array(input['freq_upper'], dtype=np.float64) 
        self.freq_lower= np.array(input['freq_lower'], dtype=np.float64) 

        self.freqx_axis_gurobi = np.array([], dtype=np.float64) 
        self.freq_upper_gurobi = np.array([], dtype=np.float64) 
        self.freq_lower_gurobi = np.array([], dtype=np.float64) 

        self.freqx_axis_z3= np.array([], dtype=np.float64)
        self.freq_upper_z3= np.array([], dtype=np.float64)
        self.freq_lower_z3= np.array([], dtype=np.float64)

        self.freqx_axis_pysat = np.array([], dtype=np.float64) 
        self.freq_upper_pysat = np.array([], dtype=np.float64)
        self.freq_lower_pysat = np.array([], dtype=np.float64)

        self.freq_upper_gurobi_lin = np.array([], dtype=np.float64) 
        self.freq_lower_gurobi_lin = np.array([], dtype=np.float64)

        self.freq_upper_z3_lin = np.array([], dtype=np.float64) 
        self.freq_lower_z3_lin = np.array([], dtype=np.float64)

        self.freq_upper_pysat_lin = np.array([], dtype=np.float64) 
        self.freq_lower_pysat_lin = np.array([], dtype=np.float64)

        self.ignore_lowerbound = input['ignore_lowerbound']
        self.adder_count = input['adder_count']
        self.wordlength = input['wordlength']
        self.adder_depth = input['adder_depth']
        self.avail_dsp = input['avail_dsp']
        self.adder_wordlength_ext = input['adder_wordlength_ext']
        self.gain_upperbound = input['gain_upperbound']
        self.gain_lowerbound = input['gain_lowerbound']
        self.coef_accuracy = input['coef_accuracy']
        self.intW = input['intW']

        self.gain_wordlength = input['gain_wordlength']
        self.gain_intW = input['gain_intW']
        
        self.gurobi_thread = input['gurobi_thread']
        self.pysat_thread = input['pysat_thread']
        self.z3_thread = input['z3_thread']

        self.timeout = input['timeout']
        self.max_iteration = input['max_iteration']
        self.start_with_error_prediction = input['start_with_error_prediction']

        self.gurobi_available = None

        self.input = input


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

    def db_to_lin_conversion(self, freq_upper, freq_lower):
        sf = SolverFunc(self.filter_type, self.order_current)
        freq_upper_lin = [np.array(sf.db_to_linear(f)).item() if not np.isnan(f) else np.nan for f in freq_upper]
        freq_lower_lin = [np.array(sf.db_to_linear(f)).item()  if not np.isnan(f) else np.nan for f in freq_lower]

        return freq_upper_lin, freq_lower_lin
    
    
    def run_backend(self):
        self.freq_upper_gurobi_lin, self.freq_lower_gurobi_lin = self.db_to_lin_conversion(self.freq_upper, self.freq_lower)
        
        self.freq_upper_z3_lin = copy.deepcopy(self.freq_upper_gurobi_lin)
        self.freq_lower_z3_lin = copy.deepcopy(self.freq_lower_gurobi_lin)

        self.freq_upper_pysat_lin = copy.deepcopy(self.freq_upper_gurobi_lin)
        self.freq_lower_pysat_lin = copy.deepcopy(self.freq_lower_gurobi_lin)


        #run gurobi test if its available
        if self.gurobi_thread > 0:
            self.gurobi_test()

        parallel_exec_instance = ParallelExecutor(self.input,
                                         self.freq_upper_gurobi_lin, 
                                         self.freq_lower_gurobi_lin,
                                         self.freq_upper_z3_lin, 
                                         self.freq_lower_z3_lin, 
                                         self.freq_upper_pysat_lin,
                                         self.freq_lower_pysat_lin)

        #iterate the order from smallest to highest
        self.order_current = self.order_lower
        while self.order_current <= self.order_upper:
            print(f"current {self.order_current}")
            print(f"upper {self.order_upper}")
            if self.start_with_error_prediction:
                self.freq_upper_gurobi_lin, self.freq_lower_gurobi_lin, self.freq_upper_z3_lin, self.freq_lower_z3_lin, self.freq_upper_pysat_lin, self.freq_lower_pysat_lin = parallel_exec_instance.execute_parallel_error_prediction(self.order_current)
            self.order_current += 1
                

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
        'max_iteration': 500,
        'start_with_error_prediction': True
    }

    # Create an instance of SolverBackend
    solver_backend_instance = SolverBackend(input_data)
    solver_backend_instance.run_backend()
