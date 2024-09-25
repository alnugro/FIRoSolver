from pebble import ProcessPool, ProcessExpired
from concurrent.futures import TimeoutError, CancelledError, wait, ALL_COMPLETED
import traceback
import time
import copy
import numpy as np
import random

try:
    from .formulation_pysat import FIRFilterPysat
    from .formulation_z3_pbsat import FIRFilterZ3
    from .formulation_gurobi import FIRFilterGurobi
    from .solver_func import SolverFunc
except:
    from formulation_pysat import FIRFilterPysat
    from formulation_z3_pbsat import FIRFilterZ3
    from formulation_gurobi import FIRFilterGurobi
    from solver_func import SolverFunc

class ErrorPredictor:
    def __init__(self, input_data
                 ):
        
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

        self.xdata = None
        self.upperbound_lin = None
        self.lowerbound_lin = None

        self.cutoffs_upper_ydata_lin = None
        self.cutoffs_lower_ydata_lin = None

        # Dynamically assign values from input_data, skipping any keys that don't have matching attributes
        for key, value in input_data.items():
            if hasattr(self, key):  # Only set attributes that exist in the class
                setattr(self, key, value)

        self.xdata_gurobi_lin = np.copy(self.xdata)
        self.freq_upper_gurobi_lin = np.copy(self.upperbound_lin)
        self.freq_lower_gurobi_lin = np.copy(self.lowerbound_lin)

        self.xdata_z3_lin = np.copy(self.xdata)
        self.freq_upper_z3_lin = np.copy(self.upperbound_lin)
        self.freq_lower_z3_lin = np.copy(self.lowerbound_lin)

        self.xdata_pysat_lin = np.copy(self.xdata)
        self.freq_upper_pysat_lin = np.copy(self.upperbound_lin)
        self.freq_lower_pysat_lin = np.copy(self.lowerbound_lin)

    def get_solver_func_dict(self):
        input_data_sf = {
        'filter_type': self.filter_type,
        'order_upperbound': self.order_upperbound,
        }

        return input_data_sf
    

        



    def execute_parallel_error_prediction(self, order_current):
        pools = []  # To store active pools for cleanup
        futures_gurobi = []  # List to store Gurobi futures
        futures_z3 = []  # List to store Z3 futures
        futures_pysat = []  # List to store PySAT futures
        self.order_current = order_current
        

        try:
            # Conditionally create the Gurobi pool
            if self.gurobi_thread > 0:
                pool_gurobi = ProcessPool(max_workers=1)
                pools.append(pool_gurobi)
                future_single_gurobi = pool_gurobi.schedule(self.gurobi_error_prediction, args=(self.gurobi_thread,), timeout=self.timeout)
                futures_gurobi.append(future_single_gurobi)   
                

            else:
                pool_gurobi = None

            # Conditionally create the Z3 pool
            if self.z3_thread > 0:
                pool_z3 = ProcessPool(max_workers=self.z3_thread)
                pools.append(pool_z3)
                for i in range(self.z3_thread):
                    future_single_z3 = pool_z3.schedule(self.z3_error_prediction, args=(i,), timeout=self.timeout)
                    futures_z3.append(future_single_z3)
                    
            else:
                pool_z3 = None

            # Conditionally create the PySAT pool
            if self.pysat_thread > 0:
                pool_pysat = ProcessPool(max_workers=self.pysat_thread)
                pools.append(pool_pysat)
                for i in range(self.pysat_thread):
                    future_single_pysat = pool_pysat.schedule(self.pysat_error_prediction, args=(i,), timeout=self.timeout)
                    futures_pysat.append(future_single_pysat)
                    
            else:
                pool_pysat = None

            solver_pools = {
                'gurobi': pool_gurobi,
                'z3': pool_z3,
                'pysat': pool_pysat
            }
            
            if self.gurobi_thread > 0:
                future_single_gurobi.add_done_callback(self.task_done('gurobi', futures_gurobi,solver_pools))
            if self.z3_thread > 0:
                for future in futures_z3:
                    future.add_done_callback(self.task_done('z3', futures_z3,solver_pools))
            if self.pysat_thread > 0:
                for future in futures_pysat:
                    future.add_done_callback(self.task_done('pysat', futures_pysat,solver_pools))
            
            # Wait for all futures to complete, handling timeouts as well
            all_futures = futures_gurobi + futures_z3 + futures_pysat
            done, not_done = wait(all_futures, return_when=ALL_COMPLETED)

        finally:
            # Ensure all pools are properly cleaned up
            for pool in pools:
                pool.stop()
                pool.join()
        
        return self.freq_upper_gurobi_lin, self.freq_lower_gurobi_lin, self.freq_upper_z3_lin, self.freq_lower_z3_lin, self.freq_upper_pysat_lin, self.freq_lower_pysat_lin


    def task_done(self, solver_name, futures,solver_pools):
        def callback(future):
            try:
                freq_upper_lin, freq_lower_lin  = future.result()  # blocks until results are ready
                print(f"{solver_name} task done")

                # Cancel all other processes for this solver (only within the same group)
                for f in futures:
                    if f is not future and not f.done():  # Check if `f` is a `Future`
                        if not f.cancel():
                        # If cancel() doesn't work, forcefully stop the corresponding pool
                            print(f"{solver_name} process couldn't be cancelled. Force stopping the pool.")
                            solver_pools[solver_name].stop()

                

                # Handle the result (custom logic depending on the solver)
                if solver_name == 'gurobi':
                    self.freq_upper_gurobi_lin = freq_upper_lin
                    self.freq_lower_gurobi_lin = freq_lower_lin
                elif solver_name == 'z3':
                    self.freq_upper_z3_lin = freq_upper_lin
                    self.freq_lower_z3_lin = freq_lower_lin
                elif solver_name == 'pysat':
                    self.freq_upper_pysat_lin = freq_upper_lin
                    self.freq_lower_pysat_lin = freq_lower_lin
                else:
                    raise ValueError(f"Parallel Executor: {solver_name} is not found")
                

            except ValueError as e:
                if str(e) == "problem is unsat":
                    raise ValueError(f"problem is unsat from the solver: {solver_name}")
            except CancelledError:
                print(f"{solver_name} task was cancelled.")
            except TimeoutError:
                print(f"{solver_name} task timed out.")
            except ProcessExpired as error:
                print(f"{solver_name} process {error.pid} expired.")
            except Exception as error:
                print(f"{solver_name} task raised an exception: {error}")
                traceback.print_exc()  # Print the full traceback to get more details


        return callback
    
    def get_solver_name(self, future, futures_gurobi, futures_z3, futures_pysat):
        """Helper function to identify which solver a future belongs to."""
        if future in futures_gurobi:
            return "Gurobi"
        elif future in futures_z3:
            return "Z3"
        elif future in futures_pysat:
            return "PySAT"
        return "Unknown"
    
    def gurobi_error_prediction(self, thread):
        h_res = []

        gurobi_instance = FIRFilterGurobi(
            self.filter_type, 
            self.order_upperbound, #you pass upperbound directly to gurobi
            self.xdata, 
            self.upperbound_lin, 
            self.lowerbound_lin, 
            self.ignore_lowerbound, 
            0, 
            self.wordlength,
            0,
            0,
            0,
            self.gain_upperbound,
            self.gain_lowerbound,
            self.coef_accuracy,
            self.intW
        )

        target_result = gurobi_instance.run_barebone_real(thread,None)
        satisfiability = target_result['satisfiability']
        if satisfiability == "unsat":
            raise ValueError("problem is unsat")
        h_res = target_result['target_h_res']

        
        freq_upper_lin, freq_lower_lin  = self.calculate_error(h_res,self.freq_upper_gurobi_lin, self.freq_lower_gurobi_lin, 'gurobi', None)
        return freq_upper_lin, freq_lower_lin

    def z3_error_prediction(self, seed):
        h_res = []
        satisfiability, h_res ,gain= self.z3_instance_creator().run_barebone(seed)
        if satisfiability == "unsat":
            raise ValueError("problem is unsat")
        freq_upper_lin, freq_lower_lin  = self.calculate_error(h_res,self.freq_upper_z3_lin, self.freq_lower_z3_lin, 'z3',gain)
        return freq_upper_lin, freq_lower_lin

    def pysat_error_prediction(self, solver_id):
        h_res = []
        satisfiability, h_res ,gain= self.pysat_instance_creator().run_barebone(solver_id)
        if satisfiability == "unsat":
            raise ValueError("problem is unsat")
        freq_upper_lin, freq_lower_lin  = self.calculate_error(h_res,self.freq_upper_pysat_lin, self.freq_lower_pysat_lin, 'pysat',gain)
        return freq_upper_lin, freq_lower_lin
    
    def calculate_error(self, h_res, freq_upper, freq_lower, solver ,gain = None):
        if solver == 'pysat':
            delta_coef = 2**-(self.wordlength-self.intW)
            delta_gain = 2**-(self.wordlength-self.intW)
        else:
            delta_coef = 10 ** - self.coef_accuracy
            delta_gain = 2**-(self.gain_wordlength-self.gain_intW)

        delta_h_res = 2**-(self.wordlength-self.intW)
        sf = SolverFunc(self.get_solver_func_dict())

        half_order = (self.order_current // 2) +1 if self.filter_type == 0 or self.filter_type == 2 else (self.order_current // 2)

        for omega in range(len(self.xdata)):
            delta_omega = []
            omega_result = 0
            if np.isnan(freq_upper[omega]):
                continue

            for m in range(half_order):
                #calculate const
                cm = sf.cm_handler(m, self.xdata[omega])
                z_result_temp = h_res[m] * cm
                
                #calculate error
                h_res_error = (delta_h_res/h_res[m])**2 if h_res[m] != 0 else 0
                cm_error = (delta_coef/cm)**2 if cm != 0 else 0
                z_error_temp = np.sqrt(h_res_error + cm_error)

                delta_omega.append(z_result_temp*z_error_temp)
                omega_result += z_result_temp
            delta_omega = np.array(delta_omega)
            delta_omega = np.square(delta_omega)
            delta_omega_result = np.sqrt(np.sum(delta_omega))

            
            if gain != None:
                omega_error = (delta_omega_result/omega_result)**2 if omega_result != 0 else 0
                gain_error = (delta_gain/gain)**2 if gain != 0 else 0
                delta_error_result = np.sqrt(omega_error+gain_error) 
            else:
                delta_error_result = delta_omega_result

            # print(f"\nError result {delta_error_result}")
            # print(f"Omega Error result {delta_omega_result}")
            # print(f"freq before {freq_upper[omega]}")


            freq_upper[omega] = freq_upper[omega]-delta_error_result
            freq_lower[omega] = freq_lower[omega]+delta_error_result
            # print(f"freq {freq_upper[omega]}")


        return freq_upper,freq_lower




    def gurobi_instance_creator(self):
        gurobi_instance = FIRFilterGurobi(
             self.filter_type, 
            self.order_upperbound, #you pass upperbound directly to gurobi
            self.xdata, 
            self.upperbound_lin, 
            self.lowerbound_lin, 
            self.ignore_lowerbound, 
            0, 
            self.wordlength,
            0,
            0,
            0,
            self.gain_upperbound,
            self.gain_lowerbound,
            self.coef_accuracy,
            self.intW
        )
    
        return gurobi_instance

    def z3_instance_creator(self):
        z3_instance = FIRFilterZ3(
                    self.filter_type, 
                    self.order_current, 
                    self.xdata, 
                    self.freq_upper_z3_lin, 
                    self.freq_lower_z3_lin, 
                    self.ignore_lowerbound, 
                    0, 
                    self.wordlength, 
                    0,
                    0,
                    0,
                    self.gain_upperbound,
                    self.gain_lowerbound,
                    self.coef_accuracy,
                    self.intW,
                    self.gain_wordlength,
                    self.gain_intW
                    )
        
        return z3_instance

    def pysat_instance_creator(self):
        pysat_instance = FIRFilterPysat(
                    self.filter_type, 
                    self.order_current, 
                    self.xdata, 
                    self.freq_upper_pysat_lin,
                    self.freq_lower_pysat_lin,
                    self.ignore_lowerbound, 
                    0, 
                    self.wordlength, 
                    0,
                    0,
                    0,
                    self.gain_upperbound,
                    self.gain_lowerbound,
                    self.intW
                    )
        
        return pysat_instance


def generate_freq_bounds(space, multiplier_to_test ,order_current):
   #random bounds generator
    random.seed(1)
    lower_cutoff = random.choice([0.2, 0.3])
    upper_cutoff = random.choice([ 0.8, 0.85, 0.9])
    

    lower_half_point = int(lower_cutoff * space)
    
    upper_half_point = int(upper_cutoff * space)
   
    
    end_point = space
    freqx_axis = np.linspace(0, 1, space)
    freq_upper = np.full(space, np.nan)
    freq_lower = np.full(space, np.nan)
    passband_upperbound = random.choice([0 , 0.2])
    passband_lowerbound = random.choice([0 , -0.2])
    stopband_upperbound = random.choice([-10,-20, -30])

    stopband_lowerbound = -1000
    
    freq_upper[0:lower_half_point] = stopband_upperbound
    freq_lower[0:lower_half_point] = stopband_lowerbound

    freq_upper[lower_half_point:upper_half_point] = passband_upperbound
    freq_lower[lower_half_point:upper_half_point] = passband_lowerbound

    freq_upper[upper_half_point:end_point] = stopband_upperbound
    freq_lower[upper_half_point:end_point] = stopband_lowerbound

    space_to_test = space * multiplier_to_test
    original_end_point = space_to_test
    original_freqx_axis = np.linspace(0, 1, space_to_test)
    original_freq_upper = np.full(space_to_test, np.nan)
    original_freq_lower = np.full(space_to_test, np.nan)

    
    original_lower_half_point = np.abs(original_freqx_axis - ((lower_half_point-1)/space)).argmin()
    original_upper_half_point = np.abs(original_freqx_axis - ((upper_half_point+1)/space)).argmin()
   
    original_freq_upper[0:original_lower_half_point] = stopband_upperbound
    original_freq_lower[0:original_lower_half_point] = stopband_lowerbound

    original_freq_upper[original_lower_half_point:original_upper_half_point] = passband_upperbound
    original_freq_lower[original_lower_half_point:original_upper_half_point] = passband_lowerbound

    original_freq_upper[original_upper_half_point:original_end_point] = stopband_upperbound
    original_freq_lower[original_upper_half_point:original_end_point] = stopband_lowerbound



     #beyond this bound lowerbound will be ignored
    ignore_lowerbound = -40

    
    #linearize the bound
    upperbound_lin = [10 ** (f / 20) if not np.isnan(f) else np.nan for f in freq_upper]
    lowerbound_lin = [10 ** (f / 20) if not np.isnan(f) else np.nan for f in freq_lower]

    original_upperbound_lin = [10 ** (f / 20) if not np.isnan(f) else np.nan for f in original_freq_upper]
    original_lowerbound_lin = [10 ** (f / 20) if not np.isnan(f) else np.nan for f in original_freq_lower]


    ignore_lowerbound_lin = 10 ** (ignore_lowerbound / 20)

    return freqx_axis,upperbound_lin,lowerbound_lin,ignore_lowerbound_lin,original_freqx_axis,original_upperbound_lin,original_lowerbound_lin

if __name__ == "__main__":
    # Test inputs
    filter_type = 0
    order_current = 20
    accuracy = 6
    adder_count = 3
    wordlength = 14
    
    adder_depth = 2
    avail_dsp = 0
    adder_wordlength_ext = 2
    gain_upperbound = 4
    gain_lowerbound = 1
    coef_accuracy = 3
    intW = 6

    space = order_current * accuracy
    

    freqx_axis,upperbound_lin,lowerbound_lin,ignore_lowerbound_lin,original_freqx_axis,original_upperbound_lin,original_lowerbound_lin = generate_freq_bounds(space,100,order_current)


    input_data = {
        'filter_type': filter_type,
        'order_upperbound': order_current,
        'xdata': freqx_axis,
        'upperbound_lin': upperbound_lin,
        'lowerbound_lin': lowerbound_lin,
        'ignore_lowerbound': ignore_lowerbound_lin,
        'original_xdata': original_freqx_axis,
        'original_upperbound_lin': original_upperbound_lin,
        'original_lowerbound_lin': original_lowerbound_lin,
        'cutoffs_x': None,
        'cutoffs_upper_ydata_lin': None,
        'cutoffs_lower_ydata_lin': None,
        'wordlength': wordlength,
        'adder_depth': 0,
        'avail_dsp': 0,
        'adder_wordlength_ext': 2, #this is extension not the adder wordlength
        'gain_wordlength' : 6,
        'gain_intW' : 2,
        'gain_upperbound': gain_upperbound,
        'gain_lowerbound': gain_lowerbound,
        'coef_accuracy': coef_accuracy,
        'intW': intW,
        'gurobi_thread': 1,
        'pysat_thread': 0,
        'z3_thread': 0,
        'timeout': 0,
        'start_with_error_prediction': False,
        'solver_accuracy_multiplier': 6,
    }

    # Create an instance of SolverBackend
    err_handler = ErrorPredictor(input_data)
    # print(f"before: {err_handler.freq_lower_gurobi_lin}")
    print(f"before: {err_handler.freq_lower_z3_lin}")
    # print(f"before: {err_handler.freq_lower_pysat_lin}")

    err_handler.execute_parallel_error_prediction(order_current)
    # print(f"After: {err_handler.freq_lower_gurobi_lin}")
    print(f"After: {err_handler.freq_lower_z3_lin}")
    # print(f"After: {err_handler.freq_lower_pysat_lin}")
