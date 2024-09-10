from pebble import ProcessPool, ProcessExpired
from concurrent.futures import TimeoutError, CancelledError, wait, ALL_COMPLETED
import traceback
import time
import copy
import numpy as np
from formulation_pysat import FIRFilterPysat
from formulation_z3_pbsat import FIRFilterZ3
from formulation_gurobi import FIRFilterGurobi
from solver_func import SolverFunc


class ParallelExecutor:
    def __init__(self, 
                 input , 
                 freq_upper_gurobi_lin,
                 freq_lower_gurobi_lin,
                 freq_upper_z3_lin,
                 freq_lower_z3_lin,
                 freq_upper_pysat_lin,
                 freq_lower_pysat_lin
                 ):
        
        self.filter_type = input['filter_type']
        self.order_current = None

        self.freqx_axis = np.array(input['freqx_axis'], dtype=np.float64) 

        self.freq_upper_gurobi_lin =    freq_upper_gurobi_lin
        self.freq_lower_gurobi_lin = freq_lower_gurobi_lin

        self.freq_upper_z3_lin = freq_upper_z3_lin
        self.freq_lower_z3_lin = freq_lower_z3_lin

        self.freq_upper_pysat_lin = freq_upper_pysat_lin
        self.freq_lower_pysat_lin = freq_lower_pysat_lin

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
            
            if self.gurobi_thread > 0:
                future_single_gurobi.add_done_callback(self.task_done('gurobi', futures_gurobi))
            if self.z3_thread > 0:
                for future in futures_z3:
                    future.add_done_callback(self.task_done('z3', futures_z3))
            if self.pysat_thread > 0:
                for future in futures_pysat:
                    future.add_done_callback(self.task_done('pysat', futures_pysat))
            
            # Wait for all futures to complete, handling timeouts as well
            all_futures = futures_gurobi + futures_z3 + futures_pysat
            done, not_done = wait(all_futures, return_when=ALL_COMPLETED)

        finally:
            # Ensure all pools are properly cleaned up
            for pool in pools:
                pool.stop()
                pool.join()
        
        return self.freq_upper_gurobi_lin, self.freq_lower_gurobi_lin, self.freq_upper_z3_lin, self.freq_lower_z3_lin, self.freq_upper_pysat_lin, self.freq_lower_pysat_lin


    def task_done(self, solver_name, futures):
        def callback(future):
            try:
                freq_upper_lin, freq_lower_lin  = future.result()  # blocks until results are ready
                print(f"{solver_name} task done")

                # Cancel all other processes for this solver (only within the same group)
                for f in futures:
                    if f is not future and not f.done():  # Check if `f` is a `Future`
                        f.cancel()
                        print(f"{solver_name} process cancelled")

                

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
            filter_type=self.filter_type, 
            order=self.order_current, 
            freqx_axis=self.freqx_axis, 
            freq_upper_lin=self.freq_upper_gurobi_lin, 
            freq_lower_lin=self.freq_lower_gurobi_lin, 
            ignore_lowerbound=self.ignore_lowerbound, 
            adder_count=self.adder_count, 
            wordlength=self.wordlength, 
            adder_depth=self.adder_depth,
            avail_dsp=self.avail_dsp,
            adder_wordlength_ext=self.adder_wordlength_ext,
            gain_upperbound=self.gain_upperbound,
            gain_lowerbound=self.gain_lowerbound,
            coef_accuracy=self.coef_accuracy,
            intW=self.intW)
        satisfiability, h_res, _ = gurobi_instance.run_barebone(thread)

        if satisfiability == "unsat":
            raise ValueError("problem is unsat")
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
        sf = SolverFunc(self.filter_type, self.order_current)

        half_order = (self.order_current // 2) +1 if self.filter_type == 0 or self.filter_type == 2 else (self.order_current // 2)

        for omega in range(len(self.freqx_axis)):
            delta_omega = []
            omega_result = 0
            if np.isnan(freq_upper[omega]):
                continue

            for m in range(half_order):
                #calculate const
                cm = sf.cm_handler(m, self.freqx_axis[omega])
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
            filter_type=self.filter_type, 
            order=self.order_current, 
            freqx_axis=self.freqx_axis, 
            freq_upper_lin=self.freq_upper_gurobi_lin, 
            freq_lower_lin=self.freq_lower_gurobi_lin, 
            ignore_lowerbound=self.ignore_lowerbound, 
            adder_count=self.adder_count, 
            wordlength=self.wordlength, 
            adder_depth=self.adder_depth,
            avail_dsp=self.avail_dsp,
            adder_wordlength_ext=self.adder_wordlength_ext,
            gain_upperbound=self.gain_upperbound,
            gain_lowerbound=self.gain_lowerbound,
            coef_accuracy=self.coef_accuracy,
            intW=self.intW,
        )
    
        return gurobi_instance

    def z3_instance_creator(self):
        z3_instance = FIRFilterZ3(
                    self.filter_type, 
                    self.order_current, 
                    self.freqx_axis, 
                    self.freq_upper_z3_lin, 
                    self.freq_lower_z3_lin, 
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

    def pysat_instance_creator(self):
        pysat_instance = FIRFilterPysat(
                    self.filter_type, 
                    self.order_current, 
                    self.freqx_axis, 
                    self.freq_upper_pysat_lin,
                    self.freq_lower_pysat_lin,
                    self.ignore_lowerbound, 
                    self.adder_count, 
                    self.wordlength, 
                    self.adder_depth,
                    self.avail_dsp,
                    self.adder_wordlength_ext,
                    self.gain_upperbound,
                    self.gain_lowerbound,
                    self.intW
                    )
        
        return pysat_instance



if __name__ == "__main__":
     # Test inputs
    filter_type = 0
    order_current = 14
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

    space = int(accuracy*order_current)
    # Initialize freq_upper and freq_lower with NaN values
    freqx_axis = np.linspace(0, 1, space) #according to Mr. Kumms paper
    freq_upper = np.full(space, np.nan)
    freq_lower = np.full(space, np.nan)

    # Manually set specific values for the elements of freq_upper and freq_lower in dB
    lower_half_point = int(0.4*(space))
    upper_half_point = int(0.6*(space))
    end_point = space

    freq_upper[0:lower_half_point] = 3
    freq_lower[0:lower_half_point] = -1

    freq_upper[upper_half_point:end_point] = -40
    freq_lower[upper_half_point:end_point] = -1000


    #beyond this bound lowerbound will be ignored
    ignore_lowerbound = -40

    def db_to_lin_conversion(freq_upper, freq_lower):
        sf = SolverFunc(filter_type, order_current)
        freq_upper_lin = [np.array(sf.db_to_linear(f)).item() if not np.isnan(f) else np.nan for f in freq_upper]
        freq_lower_lin = [np.array(sf.db_to_linear(f)).item()  if not np.isnan(f) else np.nan for f in freq_lower]

        return freq_upper_lin, freq_lower_lin
    

    #convert db to lin
    freq_upper_gurobi_lin, freq_lower_gurobi_lin = db_to_lin_conversion(freq_upper, freq_lower)
        
    freq_upper_z3_lin = copy.deepcopy(freq_upper_gurobi_lin)
    freq_lower_z3_lin = copy.deepcopy(freq_lower_gurobi_lin)

    freq_upper_pysat_lin = copy.deepcopy(freq_upper_gurobi_lin)
    freq_lower_pysat_lin = copy.deepcopy(freq_lower_gurobi_lin)


    input_data = {
        'filter_type': filter_type,
        'freqx_axis': freqx_axis,
        'freq_upper': freq_upper,
        'freq_lower': freq_lower,
        'ignore_lowerbound': ignore_lowerbound,
        'adder_count': adder_count,
        'wordlength': wordlength,
        'adder_depth': adder_depth,
        'avail_dsp': avail_dsp,
        'adder_wordlength_ext': adder_wordlength_ext,
        'gain_upperbound': gain_upperbound,
        'gain_lowerbound': gain_lowerbound,
        'coef_accuracy': coef_accuracy,
        'intW': intW,
        'gurobi_thread': 1,
        'z3_thread': 3,
        'pysat_thread': 3,
        'timeout': 1000,
        'max_iteration': 500,
        'start_with_error_prediction': True,
        'gain_wordlength': 6,
        'gain_intW' : 4
    }

    # Create an instance of SolverBackend
    parallel_error_pred_instance = ParallelExecutor(input_data,freq_upper_gurobi_lin, freq_lower_gurobi_lin,freq_upper_z3_lin, freq_lower_z3_lin,freq_upper_pysat_lin,freq_lower_pysat_lin)
    print(f"before: {parallel_error_pred_instance.freq_lower_gurobi_lin}")
    print(f"before: {parallel_error_pred_instance.freq_lower_z3_lin}")
    print(f"before: {parallel_error_pred_instance.freq_lower_pysat_lin}")

    parallel_error_pred_instance.execute_parallel_error_prediction(order_current)
    print(f"After: {parallel_error_pred_instance.freq_lower_gurobi_lin}")
    print(f"After: {parallel_error_pred_instance.freq_lower_z3_lin}")
    print(f"After: {parallel_error_pred_instance.freq_lower_pysat_lin}")
