from pebble import ProcessPool, ProcessExpired
from concurrent.futures import TimeoutError, CancelledError, wait, ALL_COMPLETED
import traceback
import time
import copy
import numpy as np


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




class Presolver:
    def __init__(self, input_data):
        
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

        self.cutoffs_upper_ydata_lin = None
        self.cutoffs_lower_ydata_lin = None

        # Dynamically assign values from input_data, skipping any keys that don't have matching attributes
        for key, value in input_data.items():
            if hasattr(self, key):  # Only set attributes that exist in the class
                setattr(self, key, value)

        self.order_current = None

        self.min_order = None
        self.min_gain = None
        self.h_res = None
        self.gain_res = None

        self.z3_gain_res = None
        self.z3_h_res = None
        self.z3_pysat_satisfiability = 'unsat'
        self.z3_or_pysat = None

        self.half_order = (self.order_upperbound // 2) if self.filter_type == 0 or self.filter_type == 2 else (self.order_upperbound // 2) - 1
        self.max_zero_reduced = 0


    def run_presolve_gurobi(self, h_zero_input = None):
        presolve_result = {}

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
            self.intW)
        
        if h_zero_input == None:
            #find max zero
            print("\nFinding Maximum H_zero......\n")
            target_result = gurobi_instance.run_barebone_real(self.gurobi_thread, 'find_max_zero')

            #get result data
            if target_result['satisfiability'] == 'unsat':
                raise ValueError("Gurobi_Presolve: problem is unsat")
            
            max_h_zero = target_result['max_h_zero']

            #decrease the h_zero by one if its not satisfiable
            solve_with_h_zero_sum_sat_flag = False
            while solve_with_h_zero_sum_sat_flag == False:
                target_result = gurobi_instance.run_barebone(self.gurobi_thread,'try_h_zero_count' ,max_h_zero)
                if target_result['satisfiability'] == 'unsat':
                    max_h_zero -= 1
                    print("\n.......h_zero was not satisfiable......\n")
                    self.max_zero_reduced +=1

                else: 
                    solve_with_h_zero_sum_sat_flag = True
                    self.h_res = target_result['h_res']
        else: 
            max_h_zero = h_zero_input

        print("\nFinding Minimum Gain......\n")
        target_result = gurobi_instance.run_barebone_real(self.gurobi_thread, 'find_min_gain',max_h_zero, None)
        if target_result['satisfiability'] == 'unsat':
            raise RuntimeError("Gurobi_Presolve: problem is somehow unsat, but this should be sat, Formulation Error: contact developer")

        min_gain = target_result['min_gain']


        print("\nFinding Minimum and Maximum For each Filter Coefficients......\n")
        h_min = []
        h_max = []

        for m in range(self.half_order + 1 ):
            target_result_max = gurobi_instance.run_barebone_real(self.gurobi_thread, 'maximize_h',max_h_zero, m)
            target_result_min = gurobi_instance.run_barebone_real(self.gurobi_thread, 'minimize_h',max_h_zero, m)

            h_max.append(target_result_max['target_h_res'])
            h_min.append(target_result_min['target_h_res'])


        presolve_result.update({
            'max_zero' : max_h_zero,
            'min_gain' : min_gain,
            'hmax' : h_max,
            'hmin' : h_min,
            'max_zero_reduced' : self.max_zero_reduced,
            'h_res':self.h_res,
            'gain_res': self.gain_res
        })
        print(presolve_result)


        return presolve_result
    
    def run_presolve_z3_pysat(self):
        presolve_result = {}

        if self.z3_thread == None:
            self.z3_thread = self.pysat_thread

        print("run_presolve_z3_pysat")
        #check first if the problem is even satisfiable
        self.execute_z3_pysat()

        if self.z3_pysat_satisfiability == 'unsat':
            raise ValueError("problem is unsat")
        
      
        #binary search to find the h_zero count sat transition point
        low, high = 0, self.order_upperbound
        iteration = 1
        max_h_zero = -1  # Default value if no 'sat' is found
        
        while low <= high:            
            #reset all inputs first
            self.z3_pysat_satisfiability = None
            self.z3_gain_res = None
            self.z3_h_res = None

            print(f"iteration: {iteration}")
            iteration += 1

            mid = (low + high) // 2

            print(f"checking mid: {mid}")
            self.execute_z3_pysat('try_h_zero_count' ,mid)

            if self.z3_pysat_satisfiability == 'sat':
                max_h_zero = mid  # Update max_zero to the current 'sat' index
                z3_gain_res_at_max = self.z3_gain_res
                z3_h_res_at_max = self.z3_h_res
                low = mid + 1

            elif self.z3_pysat_satisfiability == 'unsat':
                high = mid - 1
            else:
                raise TypeError("run_presolve_z3_pysat: Problem should be either sat or unsat, this should never happen contact developer")

        presolve_result.update({
            'max_zero' : max_h_zero,
            'min_gain' : None,
            'hmax' : None,
            'hmin' : None,
            'max_zero_reduced' : None,
            'h_res': z3_h_res_at_max,
            'gain_res': z3_gain_res_at_max
        })

        print(presolve_result)

        return presolve_result
        

    def z3_presolve(self, seed, solver_options=None, h_zero_count= None):

        satisfiability, h_res ,gain= self.z3_instance_creator().run_barebone(seed, solver_options,h_zero_count)
        
        return satisfiability, h_res ,gain
    
    def execute_z3_pysat(self, solver_options=None,h_zero_count = None):
        pools = []  # To store active pools for cleanup
        futures_z3 = []  # List to store Z3 futures

        try:
            # Conditionally create the Z3 pool
            if self.z3_thread > 0:
                pool_z3 = ProcessPool(max_workers=self.z3_thread)
                pools.append(pool_z3)
                for i in range(self.z3_thread):
                    future_single_z3 = pool_z3.schedule(self.z3_presolve, args=(i , solver_options, h_zero_count,), timeout=self.timeout)
                    futures_z3.append(future_single_z3)
                    
            else:
                pool_z3 = None

            
            all_futures = futures_z3   

            if self.z3_thread > 0:
                for future in futures_z3:
                    future.add_done_callback(self.task_done('z3', all_futures))
                    
            
            # Wait for all futures to complete, handling timeouts as well
            
            done, not_done = wait(all_futures, return_when=ALL_COMPLETED)

        finally:
            # Ensure all pools are properly cleaned up
            for pool in pools:
                pool.stop()
                pool.join()
        return

    def task_done(self, solver_name, futures):
        def callback(future):
            try:
                h_res = []
                satisfiability, h_res, gain_res  = future.result()  # blocks until results are ready
                print(f"{solver_name} task done")
                # Cancel all other processes
                for f in futures:
                    if f is not future and not f.done():  # Check if `f` is a `Future`
                        f.cancel()
                        print(f"{solver_name} process cancelled")

                self.z3_pysat_satisfiability = satisfiability
                self.z3_gain_res = gain_res
                self.z3_h_res = h_res

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



    def z3_instance_creator(self):
        z3_instance = FIRFilterZ3(
                    self.filter_type, 
                    self.order_upperbound, 
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
                    self.intW,
                    self.gain_wordlength,
                    self.gain_intW
                    )
        
        return z3_instance




if __name__ == "__main__":
     # Test inputs
    filter_type = 0
    order_current = 14
    accuracy = 1
    adder_count = 3
    wordlength = 15
    
    adder_depth = 2
    avail_dsp = 0
    adder_wordlength_ext = 2
    gain_upperbound = 3
    gain_lowerbound = 1
    coef_accuracy = 4
    intW = 4

    space = int(accuracy*order_current)
    # Initialize freq_upper and freq_lower with NaN values
    xdata = np.linspace(0, 1, space) #according to Mr. Kumms paper
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
        sf = SolverFunc(filter_type)
        upperbound_lin = [np.array(sf.db_to_linear(f)).item() if not np.isnan(f) else np.nan for f in freq_upper]
        freq_lower_lin = [np.array(sf.db_to_linear(f)).item()  if not np.isnan(f) else np.nan for f in freq_lower]

        return upperbound_lin, freq_lower_lin
    

    #convert db to lin
    freq_upper_lin, freq_lower_lin = db_to_lin_conversion(freq_upper, freq_lower)


    input_data = {
        'filter_type': filter_type,
        'order_upperbound': order_current,
        'xdata': xdata,
        'upperbound_lin': freq_upper_lin,
        'lowerbound_lin': freq_lower_lin,
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
        'z3_thread': 10,
        'pysat_thread': 3,
        'timeout': 1000,
        'max_iteration': 500,
        'start_with_error_prediction': True,
        'gain_wordlength': 6,
        'gain_intW' : 4
    }

    # Create an instance of SolverBackend
    presolver = Presolver(input_data)
    presolver.run_presolve_gurobi()
    # presolver.run_presolve_z3_pysat()
