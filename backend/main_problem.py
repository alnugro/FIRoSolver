from pebble import ProcessPool, ProcessExpired, ThreadPool
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




class MainProblem:
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

        self.adder_count = None
        self.adder_depth = None
        self.avail_dsp = None
        self.adder_wordlength_ext = None

        # Dynamically assign values from input_data, skipping any keys that don't have matching attributes
        for key, value in input_data.items():
            if hasattr(self, key):  # Only set attributes that exist in the class
                setattr(self, key, value)


        self.order_current = None

        self.min_order = None
        self.min_gain = None
        self.h_res = None
        self.gain_res = None

        self.gain_res = None
        self.h_res = None
        self.satisfiability = 'unsat'

        self.half_order = (self.order_upperbound // 2) if self.filter_type == 0 or self.filter_type == 2 else (self.order_upperbound // 2) - 1
        self.max_zero_reduced = 0

        self.result_model = {}

    def try_asserted(self, presolve_result,adderm,h_zero):
        gurobi_instance = FIRFilterGurobi(
            self.filter_type, 
            self.order_upperbound, #you pass upperbound directly to gurobi
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
            self.intW)
        


        target_result, satisfiability_loc, h_zero_count = gurobi_instance.runsolver(self.gurobi_thread, presolve_result, 'try_max_h_zero_count' ,adderm, h_zero)

        return target_result, satisfiability_loc
    
    def try_asserted_z3_pysat(self,adderm,h_zero):
        
        self.result_model = None
        self.satisfiability = None
        self.execute_z3_pysat(adderm, h_zero)

        return self.result_model , self.satisfiability




    def find_best_adder_s(self, presolve_result):
        gurobi_instance = FIRFilterGurobi(
            self.filter_type, 
            self.order_upperbound, #you pass upperbound directly to gurobi
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
            self.intW)
        
        
        #iteration search to find the minimum adder count
        
        iteration = presolve_result['min_adderm']
        best_adderm = -1  # Default value if no 'sat' is found
        target_result = None
        target_result_best = None
        print(f"presolve_result min adderm {presolve_result['min_adderm']}")

        while True:            
            print(f"iteration: {iteration}")
            
        
            max_h_zero = presolve_result['max_zero']
            print(f"checking adderm of: {iteration}")
            target_result, satisfiability_loc, _ = gurobi_instance.runsolver(self.gurobi_thread, presolve_result, 'try_max_h_zero_count' ,iteration, max_h_zero)

            if satisfiability_loc == 'unsat':
                iteration += 1
                if iteration > presolve_result['min_adderm'] * 2 * self.order_upperbound: #if iteration bigger than this, something is definitely wrong
                    break

            elif satisfiability_loc == 'sat':
                target_result_best = target_result
                best_adderm = iteration  # Update max_adderm to the current 'sat' index
                break
            else:
                raise TypeError("find_best_adder_s: Problem should be either sat or unsat, this should never happen contact developer")
        
        print(f"max iteration: {iteration}")

        if best_adderm == -1:
            print(f"presolve_result {presolve_result['min_adderm']}")
            raise RuntimeError("Somehow cannot find any solution to all the multiplier adder count: unsat")

        return target_result_best, best_adderm, max_h_zero
    


    
    def find_best_adder_m(self, presolve_result):
        gurobi_instance = FIRFilterGurobi(
            self.filter_type, 
            self.order_upperbound, #you pass upperbound directly to gurobi
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
            self.intW)
        
        
        #iteration search to find the minimum adder count
        
        iteration = presolve_result['min_adderm_without_zero']
        best_adderm = -1  # Default value if no 'sat' is found
        target_result = None
        target_result_best = None
        print(f"presolve_result min adderm {presolve_result['min_adderm_without_zero']}")

        while True:            
            print(f"iteration: {iteration}")
            
        
            max_h_zero = presolve_result['max_zero']
            print(f"checking adderm of: {iteration}")
            target_result, satisfiability_loc ,h_zero_loc = gurobi_instance.runsolver(self.gurobi_thread, presolve_result, 'find_max_zero' ,iteration)

            if satisfiability_loc == 'unsat':
                iteration += 1
                if iteration > presolve_result['min_adderm'] * 2 * self.order_upperbound: #if iteration bigger than this, something is definitely wrong
                    break

            elif satisfiability_loc == 'sat':
                target_result_best = target_result
                best_adderm = iteration  # Update max_adderm to the current 'sat' index
                break
            else:
                raise TypeError("find_best_adder_s: Problem should be either sat or unsat, this should never happen contact developer")
        
        print(f"max iteration: {iteration}")

        if best_adderm == -1:
            print(f"presolve_result {presolve_result['min_adderm']}")
            raise RuntimeError("Somehow cannot find any solution to all the multiplier adder count: unsat")

        return target_result_best, best_adderm, max_h_zero
    
    def deep_search(self,presolve_result ,input_data_dict):
        gurobi_instance = FIRFilterGurobi(
            self.filter_type, 
            self.order_upperbound, #you pass upperbound directly to gurobi
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
            self.intW)
        
        # Unpacking variables from the dictionary
        best_adderm_from_s = input_data_dict['best_adderm_from_s']
        best_adderm_from_m = input_data_dict['best_adderm_from_m']
        total_adder_s = input_data_dict['total_adder_s']
        total_adder_m = input_data_dict['total_adder_m']
        adder_s_h_zero_best = input_data_dict['adder_s_h_zero_best']
        adder_m_h_zero_best = input_data_dict['adder_m_h_zero_best']
        

        h_zero_search_space = [val for val in range(adder_m_h_zero_best+1, adder_s_h_zero_best)]

        # Initialize vars
        min_total_adder = total_adder_s if total_adder_s <= total_adder_m else total_adder_m
        found_min_flag = False
        target_result_best = None
        h_zero_best = None
        best_adderm = None

        for i, h_zero_val in enumerate(h_zero_search_space):
            print(f"value to test is {h_zero_val}")

            while True:
                target_adderm = min_total_adder - self.half_order +  h_zero_val #calculate better adderm from current best -1
                target_result, satisfiability_loc ,h_zero_loc = gurobi_instance.runsolver(self.gurobi_thread, presolve_result, 'try_h_zero_count' ,target_adderm, h_zero_val)
                if satisfiability_loc == 'sat':
                    min_total_adder -= 1
                    found_min_flag = True
                    target_result_best = target_result
                    h_zero_best = h_zero_loc
                    best_adderm = target_adderm
                    print(f"min_total_adder is {min_total_adder}")    

                else: break

        print(min_total_adder)

        return target_result_best, best_adderm, h_zero_best
    
    def find_best_adder_s_z3_paysat(self, presolve_result):
       
        #binary search to find the minimum adder count
        low, high = 1, presolve_result['max_adderm_without_zero']
        iteration = 1
        best_adderm = -1  # Default value if no 'sat' is found
        target_result = None
        target_result_best = None

        while low <= high:            
            print(f"iteration: {iteration}")
            iteration += 1

            mid = (low + high) // 2
            max_h_zero = presolve_result['max_zero']
            print(f"checking mid: {mid}")
            self.execute_z3_pysat(mid, max_h_zero)

            if self.satisfiability == 'unsat':
                low = mid + 1

            elif self.satisfiability == 'sat':
                target_result_best = self.result_model
                best_adderm = mid  # Update max_zero to the current 'sat' index
                high = mid - 1
            else:
                raise TypeError("find_best_adder_s: Problem should be either sat or unsat, this should never happen contact developer")
        
        print(f"max iteration: {iteration}")

        if best_adderm == -1:
            raise RuntimeError("Somehow cannot find any solution to all the multiplier adder count: unsat")

        return target_result_best, best_adderm ,max_h_zero

        

    def z3_run_main(self, seed, adderm, h_zero_count=0):

        result_model , satisfiability = self.z3_instance_creator().runsolver(seed, adderm,h_zero_count)
        
        return result_model , satisfiability
    
    def pysat_run_main(self, solver_id, adderm, h_zero_count=0):

        result_model , satisfiability= self.pysat_instance_creator().runsolver(solver_id,adderm,h_zero_count)
        
        return result_model , satisfiability
    
    
    def execute_z3_pysat(self, adderm , h_zero_count = None):
        pools = []  # To store active pools for cleanup
        futures_z3 = []  # List to store Z3 futures
        futures_pysat = []  # List to store PySAT futures
        

        try:
            # Conditionally create the Z3 pool
            if self.z3_thread > 0:
                pool_z3 = ProcessPool(max_workers=self.z3_thread)
                pools.append(pool_z3)
                for i in range(self.z3_thread):
                    future_single_z3 = pool_z3.schedule(self.z3_run_main, args=(i , adderm, h_zero_count,), timeout=self.timeout)
                    futures_z3.append(future_single_z3)
                    
            else:
                pool_z3 = None

            # Conditionally create the PySAT pool
            if self.pysat_thread > 0:
                pool_pysat = ProcessPool(max_workers=self.pysat_thread)
                pools.append(pool_pysat)
                for i in range(self.pysat_thread):
                    future_single_pysat = pool_pysat.schedule(self.pysat_run_main, args=(i , adderm, h_zero_count,), timeout=self.timeout)
                    futures_pysat.append(future_single_pysat)
                    
            else:
                pool_pysat = None

            solver_pools = {
                'z3': pool_z3,
                'pysat': pool_pysat
            }
            
            if self.z3_thread > 0:
                for future in futures_z3:
                    future.add_done_callback(self.task_done('z3', futures_z3,solver_pools))
            if self.pysat_thread > 0:
                for future in futures_pysat:
                    future.add_done_callback(self.task_done('pysat', futures_pysat,solver_pools))
            
            # Wait for all futures to complete, handling timeouts as well
            all_futures =  futures_z3 + futures_pysat
            done, not_done = wait(all_futures, return_when=ALL_COMPLETED)

            # Iterate over completed futures and handle exceptions
            for future in done:
                try:
                    result_model , satisfiability = future.result()
                except CancelledError:
                    pass
                except TimeoutError:
                    pass
                except Exception as e:
                    # Handle other exceptions if necessary
                    print(f"Task raised an exception: {e}")
                    traceback.print_exc()

        finally:
            # Ensure all pools are properly cleaned up
            for pool in pools:
                pool.stop()
                pool.join()
        
        return
    
    def task_done(self, solver_name, futures, solver_pools):
        def callback(future):
            try:
                h_res = []
                result_model , satisfiability  = future.result()  # blocks until results are ready
                print(f"{solver_name} task done")
                
                # Cancel all other processes for this solver (only within the same group)
                for f in futures:
                    if f is not future and not f.done():  # Check if `f` is a `Future`
                        if not f.cancel():
                        # If cancel() doesn't work, forcefully stop the corresponding pool
                            print(f"{solver_name} process couldn't be cancelled. Force stopping the pool.")
                            solver_pools[solver_name].stop()
                


                self.satisfiability = satisfiability
                self.result_model = result_model
               
            except CancelledError:
                print(f"{solver_name} task was cancelled.")
            except TimeoutError:
                print(f"{solver_name} task timed out.")
            except ProcessExpired as error:
                print(f"{solver_name} process {error.pid} expired.")
            except Exception as error:
                print(f"{solver_name} task raised an exception: {error}")
                # traceback.print_exc()  # Print the full traceback to get more details


        return callback



    def z3_instance_creator(self):
        z3_instance = FIRFilterZ3(
                    self.filter_type, 
                    self.order_upperbound, 
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
    
    def pysat_instance_creator(self):
        pysat_instance = FIRFilterPysat(
                    self.filter_type, 
                    self.order_upperbound, 
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
                    self.intW
                    )
        
        return pysat_instance


if __name__ == "__main__":
    print("MainProblem: can only be run from main backend")
