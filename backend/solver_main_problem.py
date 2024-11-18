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


def try_asserted(config, presolve_result, adderm, h_zero, thread, done_flag=None):
    """
    Tries to solve for the given adderm and h_zero values using a Gurobi instance.
    """
    
    
    
    # Create an instance of FIRFilterGurobi using the configuration dictionary
    gurobi_instance = FIRFilterGurobi(
        config["filter_type"],
        config["half_order"],  # Upper bound is passed directly to Gurobi
        config["xdata"],
        config["upperbound_lin"],
        config["lowerbound_lin"],
        config["ignore_lowerbound"],
        config["adder_count"],
        config["wordlength"],
        config["adder_depth"],
        config["avail_dsp"],
        config["adder_wordlength_ext"],
        config["gain_upperbound"],
        config["gain_lowerbound"],
        config["coef_accuracy"],
        config["intW"]
    )

    if done_flag is not None:
        if done_flag.value:
            time.sleep(5)
            return None, None

    print(f"@MSG@ : Trying to solve for adderm: {adderm} and h_zero: {h_zero} with thread: {thread}")

    # Run the solver
    target_result, satisfiability_loc, h_zero_count = gurobi_instance.runsolver(
        thread, presolve_result, 'try_max_h_zero_count', adderm, h_zero
    )

    return target_result, satisfiability_loc

class MainProblem:
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
        # Dynamically assign values from input_data, skipping any keys that don't have matching attributes
        for key, value in input_data.items():
            if hasattr(self, key):  # Only set attributes that exist in the class
                setattr(self, key, value)

        self.am_start = 0 if self.am_start is None else self.am_start
        self.order_current = None

        self.min_order = None
        self.min_gain = None
        self.h_res = None
        self.gain_res = None

        self.gain_res = None
        self.h_res = None
        self.satisfiability = 'unsat'

        self.max_zero_reduced = 0

        self.result_model = {}

        # self.satisfiability_gurobi = [None for i in range(self.worker)]
        # self.result_model_gurobi = [None for i in range(self.worker)]
        # self.h_zero_count = [None for i in range(self.worker)]

        self.sat_list = [None for i in range(self.worker)]
        self.direct_refine_search = False
        self.skip_as_search = False
        self.skip_deepsearch = False
        self.min_am = 0
        self.futures = []
        self.increased_thread = None
        self.increased_thread_idx = None
        self.all_done = []


    def initialize_json_file(self, deepsearch = False):
        """Initialize the JSON file with default values if it doesn't exist."""
        JSON_FILE = 'problem_description.json'
        LOCK_FILE = JSON_FILE + '.lock'
        lock = FileLock(LOCK_FILE)
        data = None
        with lock:
            with open(JSON_FILE, 'r') as f:
                data = json.load(f)

            problem_id = str(self.problem_id)
            if deepsearch:
                if ('deepsearch_h_zero' not in data or 'deepsearch_am_cost' not in data) or self.continue_solver is False:
                    data[problem_id].update({'deepsearch_h_zero': []})
                    data[problem_id].update({'deepsearch_am_cost': []})
                    data[problem_id].update({'deepsearch_target_result': {}})
                    with open(JSON_FILE, 'w') as f:
                        json.dump(data, f)
                    return None, None
                else:
                    h_zero, am_cost = self.continue_search(data[problem_id], deepsearch=True)
                    return h_zero, am_cost
            else:
                if 'as_results' not in data[problem_id] or self.continue_solver is False:
                    data[problem_id].update({'as_results': {}})
                    data[problem_id].update({'as_target_result': {}})
                    with open(JSON_FILE, 'w') as f:
                        json.dump(data, f)
                    return None, None
                else:
                    max_unsat, min_sat = self.continue_search(data[problem_id]['as_results'])
                    print(f"@MSG@ :Results already exist for problem ID {problem_id}, continuing...")
                    return max_unsat, min_sat
            
    def target_result_update(self, target_result, deepsearch = False):
        JSON_FILE = 'problem_description.json'
        LOCK_FILE = JSON_FILE + '.lock'
        lock = FileLock(LOCK_FILE)

        target_r = copy.deepcopy(target_result)
        target_r.update({
            'problem_id': self.problem_id,
            'avail_dsp' : self.avail_dsp,
            'filter_type' : self.filter_type,
            'adder_wordlength_ext' : self.adder_wordlength_ext,
            'half_order': self.half_order,
        })
       


        with lock:
            with open(JSON_FILE, 'r') as f:
                data = json.load(f)

                problem_id = str(self.problem_id)
                if deepsearch:
                    data[problem_id].update({'deepsearch_target_result': target_r})
                else:
                    data[problem_id].update({'as_target_result': target_r})
                
                with open(JSON_FILE, 'w') as f:
                    json.dump(data, f, indent=4)

                # Save the updated dictionary back to the file

                


    def update_json_file(self, number=None, result=None, deepsearch = False, h_zero = None, am_cost = None):
        """Update the JSON file as_results, keeping keys sorted."""
        JSON_FILE = 'problem_description.json'
        LOCK_FILE = JSON_FILE + '.lock'
        lock = FileLock(LOCK_FILE)

        with lock:
            with open(JSON_FILE, 'r') as f:
                data = json.load(f)

            problem_id = str(self.problem_id)

            if deepsearch:
                if h_zero is not None and am_cost is not None:
                    data[problem_id]['deepsearch_h_zero'] = h_zero
                    data[problem_id]['deepsearch_am_cost'] = am_cost
                else:
                    raise ValueError("h_zero or am_cost is None")
                with open(JSON_FILE, 'w') as f:
                    json.dump(data, f, indent=4)
                return
            
            # Ensure `as_results` are initialized
            if 'as_results' not in data[problem_id]:
                raise ValueError("as_results not found in JSON file")

            # Update the `as_results` dictionary with the new entry
            if number is not None and result is not None:
                data[problem_id]['as_results'][str(number)] = result

            # Sort `as_results` by key
            data[problem_id]['as_results'] = OrderedDict(
                sorted(data[problem_id]['as_results'].items())
            )

            # Write the updated data back to the JSON file
            with open(JSON_FILE, 'w') as f:
                json.dump(data, f, indent=4)

    def continue_search(self, data, deepsearch = False):
        """Continue the search from where it left off."""

        if deepsearch:
            try:
                h_zero = data['deepsearch_h_zero']
                am_cost = data['deepsearch_am_cost']
            except KeyError:
                print(f"@MSG@ : deepsearch_h_zero or deepsearch_am_cost not found in JSON file starting from the beginning...")
                return None, None
            

            print(f"@MSG@ : Results found in the JSON file. Continuing from am: {am_cost} with h_zero search space: {h_zero}")

            return h_zero, am_cost

        sat_numbers = []
        unsat_numbers = []
        canceled_numbers = []
        for key, value in data.items():
            if value == 'sat':
                sat_numbers.append(key)
            if value == 'unsat':
                unsat_numbers.append(key)
            if value == 'canceled':
                canceled_numbers.append(key)
        
        min_sat = min(sat_numbers) if sat_numbers else None
        max_unsat = max(unsat_numbers) if unsat_numbers else None

        try:
            min_sat = int(min_sat) if min_sat is not None else None
            max_unsat = int(max_unsat) if max_unsat is not None else None
        except Exception as e:
            print(f"Failed to convert sat_numbers and unsat_numbers to integers: {e}")

        
        
        if min_sat is None and max_unsat is None:
            print(f"@MSG@ : No results found in the JSON file. Starting from the beginning...")
            return None, None
        elif min_sat is None and max_unsat is not None:
            print(f"@MSG@ : No sat results found in the JSON file. Starting search step from max unsat...")
            self.am_start = max_unsat+1 if max_unsat+1 > self.am_start else self.am_start
            return None, None
        elif min_sat is not None and max_unsat is None:
            print(f"@MSG@ : No unsat results found in the JSON file. Starting refine search up to min sat...")
            self.direct_refine_search = True
            return 0, min_sat-1
        elif min_sat is not None and max_unsat is not None:
            if min_sat - 1 <= max_unsat:
                self.skip_as_search = True
                return max_unsat, min_sat
            print(f"@MSG@ : Results found in the JSON file. Continuing from am: {max_unsat} to {min_sat} with refine search")
            self.direct_refine_search = True
            return max_unsat+1, min_sat-1
        else:
            raise ValueError("Invalid state reached in continue_search")
        
    def unload_json_file(self, am_count = None, sat_list = None, find_max_unsat = None, find_min_sat = None):
        JSON_FILE = 'problem_description.json'
        LOCK_FILE = JSON_FILE + '.lock'
        lock = FileLock(LOCK_FILE)

        with lock:
            with open(JSON_FILE, 'r') as f:
                data = json.load(f)

        problem_id = str(self.problem_id)

        data_to_check = data[problem_id]['as_results']
        if am_count is not None and sat_list is not None:
            if len(am_count) != len(sat_list):
                raise ValueError("am_count and sat_list should have the same length")
            for idx, am in enumerate(am_count):
                if str(am) in data_to_check:
                    sat_list[idx] = data_to_check[str(am)]

            return sat_list
        
        if find_max_unsat is not None:
            max_unsat = max([int(key) for key, value in data_to_check.items() if value == 'unsat'])
            return max_unsat
        
        if find_min_sat is not None:
            min_sat = min([int(key) for key, value in data_to_check.items() if value == 'sat'])
            return min_sat
        
    def unload_target_result(self, deepsearch = False):
        JSON_FILE = 'problem_description.json'
        LOCK_FILE = JSON_FILE + '.lock'
        lock = FileLock(LOCK_FILE)
        target_result = {}
        with lock:
            with open(JSON_FILE, 'r') as f:
                data = json.load(f)
        key = None
        if deepsearch:
            if 'deepsearch_target_result' in data[str(self.problem_id)] and self.skip_deepsearch is False:
                for key, value in data[str(self.problem_id)]['deepsearch_target_result'].items():
                    target_result[key] = self.convert_nested_list(value)                
                return target_result
            
            if 'optimal_leak' in data[str(self.problem_id)]:
                raise RuntimeError("This solver is done and should not be called again")
            elif 'optimal_valid' in data[str(self.problem_id)]:
                raise RuntimeError("This solver is done and should not be called again")
            else:
                print(f"@MSG@ :Deepsearch is complete, But no solution is found in result file for problem {self.problem_id} cannot continue.")
                raise ValueError("No key found in JSON file")
        else:    
            if 'as_target_result' in data[str(self.problem_id)] and self.skip_as_search is False:
                for key, value in data[str(self.problem_id)]['as_target_result'].items():
                    target_result[key] = self.convert_nested_list(value)
                return target_result
               
            if 'min(AS)_leak' in data[str(self.problem_id)]:
                key = data[str(self.problem_id)]['min(AS)_leak']
                JSON_FILE = 'result_leak.json'
            elif 'min(AS)_valid' in data[str(self.problem_id)]:
                key = data[str(self.problem_id)]['min(AS)_valid']
                JSON_FILE = 'result_valid.json'
            else:
                print(f"@MSG@ :AS search is complete. No further search needed. But no solution is found in result file for problem {self.problem_id} please delete the file and start again.")
                raise ValueError("No key found in JSON file")
        
        LOCK_FILE = JSON_FILE + '.lock'
        lock = FileLock(LOCK_FILE)

        with lock:
            with open(JSON_FILE, 'r') as f:
                data = json.load(f)

        for key, value in data[key].items():
            target_result[key] = self.convert_nested_list(value)

        return target_result
    
    def convert_nested_list(self, lst):
        """Recursively convert elements in nested lists to int or float where possible."""
        if isinstance(lst, str):
            try:
                lst = ast.literal_eval(lst)
            except Exception as e:
                print(f"Error parsing string to list: {e}")
                return lst  # Return the string if it can't be parsed

        if isinstance(lst, list):
            if not lst:  # Return immediately if the list is empty
                return lst


            for i in range(len(lst)):
                lst[i] = self.convert_nested_list(lst[i])  # Recursively process each element
            return lst
        else:
            # lst is not a list; try to convert it to int or float
            if isinstance(lst, str):
                try:
                    if '.' in lst:
                        return float(lst)
                    else:
                        return int(lst)
                except ValueError:
                    return lst  # Return the string if it can't be converted
            else:
                return lst  # Return the element as is if it's not a string or list

    def try_asserted(self, presolve_result , adderm,h_zero, thread = None):
        print(f"@MSG@ :Trying to solve for adderm: {adderm} and h_zero: {h_zero}")
        gurobi_instance = FIRFilterGurobi(
            self.filter_type, 
            self.half_order, #you pass upperbound directly to gurobi
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
        
        thread = thread if thread is not None else self.gurobi_thread
        target_result, satisfiability_loc, h_zero_count = gurobi_instance.runsolver(thread, presolve_result, 'try_max_h_zero_count' ,adderm, h_zero)
        print(f"@MSG@ :Result for adderm: {adderm} and h_zero: {h_zero} is {satisfiability_loc}")
        return target_result, satisfiability_loc
    
    def try_asserted_z3_pysat(self,adderm,h_zero):
        
        self.result_model = None
        self.satisfiability = None
        self.execute_z3_pysat(adderm, h_zero)

        return self.result_model , self.satisfiability




    def find_best_adder_s2(self, presolve_result):
        gurobi_instance = FIRFilterGurobi(
            self.filter_type, 
            self.half_order, #you pass upperbound directly to gurobi
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
        

        iteration = self.am_start if self.am_start != None else presolve_result['min_adderm']
        best_adderm = -1  # Default value if no 'sat' is found
        target_result = None
        target_result_best = None
        print(f"presolve_result min adderm {presolve_result['min_adderm']}")

        while True:            
            print(f"iteration: {iteration}")
            print(f"@MSG@ : finding am for min(AS), am: {iteration}")
            max_h_zero = presolve_result['max_zero']
            print(f"checking adderm of: {iteration}")
            start_time = time.time()
            target_result, satisfiability_loc, _ = gurobi_instance.runsolver(self.gurobi_thread, presolve_result, 'try_max_h_zero_count' ,iteration, max_h_zero)
            end_time = time.time()
            duration = end_time - start_time
            print(f"@MSG@ : iteration {iteration} took {duration} seconds and result is {satisfiability_loc}")

            if satisfiability_loc == 'unsat':
                iteration += 1

            
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
            self.half_order, #you pass upperbound directly to gurobi
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
        
        max_iter = presolve_result['min_adderm'] * 4 * self.order_upperbound if presolve_result['min_adderm'] != 0 else 5 * self.order_upperbound
        #iteration search to find the minimum adder count
        
        iteration = presolve_result['min_adderm_without_zero']
        best_adderm = -1  # Default value if no 'sat' is found
        target_result = None
        target_result_best = None
        # print(f"presolve_result min adderm {presolve_result['min_adderm_without_zero']}")
        max_h_zero = None

        while True:            
            print(f"iteration: {iteration}")
            
        
            print(f"checking adderm of: {iteration}")
            target_result, satisfiability_loc ,h_zero_loc = gurobi_instance.runsolver(self.gurobi_thread, presolve_result, 'find_max_zero' ,iteration)

            if satisfiability_loc == 'unsat':
                iteration += 1
                if iteration > max_iter: #if iteration bigger than this, something is definitely wrong
                    break

            elif satisfiability_loc == 'sat':
                target_result_best = target_result
                best_adderm = iteration # Update max_adderm to the current 'sat' index
                max_h_zero = h_zero_loc
                break
            else:
                raise TypeError("find_best_adder_s: Problem should be either sat or unsat, this should never happen contact developer")
        
        print(f"max iteration: {iteration}")

        if best_adderm == -1:
            print(f"presolve_result {presolve_result['min_adderm']}")
            raise RuntimeError("Somehow cannot find any solution to all the multiplier adder count: unsat")

        return target_result_best, best_adderm, max_h_zero
    
    def deep_search2(self,presolve_result ,input_data_dict):
        gurobi_instance = FIRFilterGurobi(
            self.filter_type, 
            self.half_order, #you pass upperbound directly to gurobi
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
        total_adder_s = input_data_dict['total_adder_s']
        adder_s_h_zero_best = input_data_dict['adder_s_h_zero_best']

        print(f"presolveresult {presolve_result}")
        print(f"input data dict {input_data_dict}")
        

        h_zero_search_space = [val for val in range(adder_s_h_zero_best)]
        adder_m_cost = [None for i in range(len(h_zero_search_space))]
        for i, val in enumerate(h_zero_search_space):
            adder_m_cost[i] = total_adder_s - self.order_upperbound + (2*val) - 1


        print(f"h_zero_search_space is {h_zero_search_space}")
        print(f"adder_m_cost is {adder_m_cost}")

        # Initialize vars
        target_result_best = None
        h_zero_best = None
        best_adderm = None
        done = False
    
        while True:
            # print(f"im running")
            for i, h_zero_val in enumerate(h_zero_search_space):
                target_adderm = adder_m_cost[i] #calculate better adderm from current best -1
                if all(val < 0 for val in adder_m_cost):
                    done = True
                    break

                if target_adderm < 0:
                    continue
                

                print(f"testing adderm {target_adderm} and h_zero {h_zero_val}")
                target_result, satisfiability_loc ,h_zero_loc = gurobi_instance.runsolver(self.gurobi_thread, presolve_result, 'try_h_zero_count' ,target_adderm, h_zero_val)
                if satisfiability_loc == 'sat':
                    adder_m_cost = [val - 1 for val in adder_m_cost] #decrement all values by 1
                    target_result_best = target_result
                    h_zero_best = h_zero_loc
                    best_adderm = target_adderm
                    print(f"adder m cost best is {adder_m_cost[i]}")
                else:
                    adder_m_cost[i] = -1
                
            if done:
                break
        
        if target_result_best == None:
            print(f"No solution from deep search")
        print(f"min_total_adder is {target_result_best}")
        print(f"best adderm is {best_adderm}")
        print(f"h_zero_best is {h_zero_best}")
        return target_result_best, best_adderm, h_zero_best
    
    
    
    
    def find_best_adder_s_z3_paysat(self, presolve_result):
       
        #iteration search to find the minimum adder count
        max_iter = presolve_result['min_adderm_without_zero'] * 4 * self.order_upperbound if presolve_result['min_adderm_without_zero'] != 0 else 5 * self.order_upperbound
        iteration = presolve_result['min_adderm_without_zero']
        best_adderm = -1  # Default value if no 'sat' is found
        target_result = None
        target_result_best = None

        while True:            
            print(f"iteration: {iteration}")
            
        
            max_h_zero = presolve_result['max_zero']
            print(f"checking adderm of: {iteration}")

            self.execute_z3_pysat(iteration, max_h_zero)


            if self.satisfiability == 'unsat':
                iteration += 1
                if iteration > max_iter: #if iteration bigger than this, something is definitely wrong
                    break

            elif self.satisfiability == 'sat':
                target_result_best = self.result_model
                best_adderm = iteration  # Update max_adderm to the current 'sat' index
                break
            else:
                raise TypeError("find_best_adder_s: Problem should be either sat or unsat, this should never happen contact developer")
        
        print(f"max iteration: {iteration}")

        if best_adderm == -1:
            raise RuntimeError("Somehow cannot find any solution to all the multiplier adder count: unsat")

        return target_result_best, best_adderm ,max_h_zero

        

    def z3_run_main(self, adderm, h_zero_count=0):

        result_model , satisfiability = self.z3_instance_creator().runsolver(self.z3_thread, adderm,h_zero_count)
        
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
                pool_z3 = ProcessPool(max_workers=1)
                pools.append(pool_z3)
                future_single_z3 = pool_z3.schedule(self.z3_run_main, args=(adderm, h_zero_count,), timeout=self.timeout)
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
                    future.add_done_callback(self.task_done_z3_pysat('z3', futures_z3,solver_pools))
            if self.pysat_thread > 0:
                for future in futures_pysat:
                    future.add_done_callback(self.task_done_z3_pysat('pysat', futures_pysat,solver_pools))
            
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
    
    def task_done_z3_pysat(self, solver_name, futures, solver_pools):
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
    


    def deep_search(self, presolve_result, input_data_dict):
        sat_found_event = threading.Event()
        task_lock = threading.RLock()  # Use RLock to prevent deadlocks
        failed_cancel = threading.Event()
        
        self.min_am = 0
        # Unpacking variables from the dictionary
        total_adder_s = input_data_dict['total_adder_s']
        adder_s_h_zero_best = input_data_dict['adder_s_h_zero_best']

        # Initialize vars
        self.target_result_best = None
        self.h_zero_best = None
        self.best_adderm = None
        done = False


        if self.continue_solver is False:
            self.initialize_json_file(deepsearch=True)
            # Initialize h_zero_search_space and adder_m_cost
            self.h_zero_search_space = [val for val in range(adder_s_h_zero_best)]
            self.adder_m_cost = [
                total_adder_s - self.order_upperbound + (2 * val) - 1
                for val in self.h_zero_search_space
            ]
            print(f"@MSG@ : h_zero starting search space: {self.h_zero_search_space}")
            print(f"@MSG@ : adder_m starting cost for h_zero: {self.adder_m_cost}")

        else:
            self.h_zero_search_space, self.adder_m_cost = self.initialize_json_file(deepsearch=True)
            if self.h_zero_search_space is None or self.adder_m_cost is None:
                self.h_zero_search_space = [val for val in range(adder_s_h_zero_best)]
                self.adder_m_cost = [
                    total_adder_s - self.order_upperbound + (2 * val) - 1
                    for val in self.h_zero_search_space
                ]
                print(f"@MSG@ : h_zero starting search space: {self.h_zero_search_space}")
                print(f"@MSG@ : adder_m starting cost for h_zero: {self.adder_m_cost}")
            else:
                self.target_result_best = self.unload_target_result(deepsearch=True)
                print(f"@MSG@ : target result best is {self.target_result_best}")
                if self.target_result_best is not None and len(self.target_result_best) > 0:
                    self.best_adderm = self.target_result_best['am_count']
                    self.h_zero_best = self.target_result_best['h_zero_count']
                    print(f"@MSG@ : Loaded h_zero starting search space: {self.h_zero_search_space}")
                    print(f"@MSG@ : Loaded adder_m starting cost for h_zero: {self.adder_m_cost}")
        
        

        if all(val < self.min_am for val in self.adder_m_cost):
            print("@MSG@ : All adder_m_cost values are smaller than min_am. No Better solution possible.")
            return None, None, None

        
        # Create pools
        pools = [ProcessPool(max_workers=1) for _ in range(self.worker)]

        try:
            while not done:
                if failed_cancel.is_set():
                    # Cancel failed, need to restart the loop wait a bit
                    print("Failed cancel, waiting before restarting loop.")
                    print(f"pool status: {pools}")
                    time.sleep(5)
                    print(f"pool status: {pools}")
                    failed_cancel.clear()

                print("\n\nStarting new iteration\n\n")
                # Check if all adder_m_cost values are negative
                if all(val < self.min_am for val in self.adder_m_cost):
                    print(f"@MSG@ : Deep search complete.")
                    break

                print("Searching for adder_m_cost >= 0")

                # Get indices where adder_m_cost >= 0
                indices_to_process = [
                    i for i, val in enumerate(self.adder_m_cost) if val >= self.min_am
                ]
                print(f"@MSG@ : h_zero to solve: {[self.h_zero_search_space[i] for i in indices_to_process]}")
                print(f"@MSG@ : adder_m_cost to solve: {[self.adder_m_cost[i] for i in indices_to_process]}")   
                if not indices_to_process:
                    print("No adder_m_cost values >= 0. Ending search.")
                    break

                # Dictionary to keep track of running tasks
                futures_dict = {}
                futures = [None for _ in range(min(self.worker, len(indices_to_process)))]
                config = self.config_gurobi()
                # Function to submit new tasks
                def submit_task(idx):
                    with task_lock:
                        h_zero_val = self.h_zero_search_space[idx]
                        target_adderm = self.adder_m_cost[idx]

                        if target_adderm < self.min_am:
                            return  # Skip if adder_m_cost is less than self.min_am
                        
                        future_index = idx % len(futures)

                        if all(future is not None for future in futures):
                            for future in futures:
                                if future.done():
                                    future_index = futures.index(future)
                                    break

                        # Initialize variables for task management
                        threads_per_worker = self.gurobi_thread // self.worker
                        threads_per_worker +=1 if idx % self.worker==self.worker-1 else 0

                        print(f"pool_index: {future_index}, idx: {idx}, h_zero: {h_zero_val}, adder_m_cost: {target_adderm}")
                        future = pools[future_index].schedule(
                            try_asserted,
                            args=(config, presolve_result, target_adderm, h_zero_val, threads_per_worker,),
                            timeout=self.timeout
                        )
                        futures_dict[idx] = future
                        futures[future_index] = future
                        future.add_done_callback(
                            self.task_done_deep_search(
                                idx, futures_dict, indices_to_process, task_lock, submit_task, sat_found_event, pools, failed_cancel, futures
                            )
                        )


                # Submit initial tasks to fill up the worker slots
                with task_lock:
                    for _ in range(min(self.worker, len(indices_to_process))):
                        idx = indices_to_process.pop(0)
                        submit_task(idx)

                # Wait for sat_found_event or until all tasks are done
                while not sat_found_event.is_set() and futures_dict:
                    time.sleep(0.1)  # Sleep briefly to avoid busy waiting

                if sat_found_event.is_set():
                    # Reduce adder_m_cost by 1 for all h_zero's
                    self.adder_m_cost = [
                        val - 1 for val in self.adder_m_cost
                    ]
                    print(f"Reduced adder_m_cost: {self.adder_m_cost}")
                    sat_found_event.clear()
                    # Need to restart the loop
                    continue
                else:
                    # No 'sat' found, and all tasks are done
                    done = True
                    break

        except Exception as e:
            print(f"Exception occurred: {e}")

        finally:
            for pool in pools:
                pool.stop()
                pool.join()

        if self.target_result_best is None:
            print("No solution from deep search")

        else:
            print(f"min_total_adder is {self.target_result_best}")
            print(f"best adderm is {self.best_adderm}")
            print(f"h_zero_best is {self.h_zero_best}")


        return self.target_result_best, self.best_adderm, self.h_zero_best

    def task_done_deep_search(self, idx, futures_dict, indices_to_process, task_lock, submit_task, sat_found_event, pools, failed_cancel, futures):
        def callback(future):
            try:
                target_result, satisfiability_loc = future.result()
                h_zero_val = self.h_zero_search_space[idx]
                adder_m_cost_result = self.adder_m_cost[idx]
                future_index = futures.index(future)
                with task_lock:
                    # Remove the completed task from futures_dict
                    del futures_dict[idx]

                if satisfiability_loc == 'sat':
                    # Found a satisfiable solution
                    print(f"@MSG@ : worker {future_index}: h_zero count {h_zero_val} with AM cost {adder_m_cost_result} is sat")
                    self.target_result_best = target_result
                    self.target_result_update(target_result,deepsearch=True)
                    self.h_zero_best = h_zero_val
                    self.best_adderm = adder_m_cost_result
                    sat_found_event.set()

                    

                    # Cancel other running tasks
                    with task_lock:
                        for other_idx, other_future in list(futures_dict.items()):
                            if not other_future.done():
                                if not other_future.cancel():
                                    failed_cancel.set()
                                    print(f"Cancelling task for index {other_idx}.")
                                    pools[other_idx].stop()
                                    pools[other_idx].join()
                                    pools[other_idx] = ProcessPool(max_workers=1)
                                            
                        futures_dict.clear()

                else:
                    # 'unsat', set adder_m_cost[idx] = -1
                    print(f"@MSG@ : worker {future_index}: h_zero count {h_zero_val} with AM cost {adder_m_cost_result} is unsat")
                    self.adder_m_cost[idx] = -1

                    with task_lock:
                        # If there are more indices to process, submit a new task
                        if indices_to_process:
                            next_idx = indices_to_process.pop(0)
                            submit_task(next_idx)
                        elif not futures_dict:
                            # No more tasks running and no more indices, proceed to next iteration
                            sat_found_event.set()
                    
                adder_m_cost_copy = copy.deepcopy(self.adder_m_cost)
                h_zero_search_space_copy = copy.deepcopy(self.h_zero_search_space)
                if sat_found_event.is_set():
                    adder_m_cost_copy = [val - 1 for val in self.adder_m_cost]
                self.update_json_file(am_cost=adder_m_cost_copy, h_zero=h_zero_search_space_copy, deepsearch=True)
            
            except CancelledError:
                print(f"Task at index {idx} was cancelled.")
            except TimeoutError:
                print(f"Task at index {idx} timed out.")
            except ProcessExpired as error:
                print(f"Task at index {idx} raised a ProcessExpired error: {error}")
            except Exception as error:
                print(f"Task raised an exception: {error}")
                traceback.print_exc()  # Print the full traceback to get more details

        return callback




    def find_best_adder_s(self, presolve_result):
        """Find the best adder count for the smallest SAT number."""
        adder_s_h_zero_best = presolve_result['max_zero']

        lower_bound, upper_bound = self.initialize_json_file()
        if self.skip_as_search:
            print(f"Skipping AS search")
            target_result = self.unload_target_result()
            print(f"@MSG@ : Target result is loaded from result file")
            print(f"@MSG@ : Best AM for min(AS) was {upper_bound}")
            return target_result, upper_bound, adder_s_h_zero_best

        if self.direct_refine_search == False:
            target_result = None
            print(f"@MSG@ : Starting AM search for min(AS) with step size {self.search_step}")
            lower_bound, upper_bound, found_sat, target_result_best_as = self.min_as_finder_gurobi(presolve_result)
            print(f"@MSG@ : upper {upper_bound} and lower at {lower_bound} for min(AS)")
            #this is just to print 
            low = lower_bound-1 if lower_bound != 0 else 0
            print(f"@MSG@ : Found SAT at {found_sat} and unsat at {low} for min(AS)")

            target_result = target_result_best_as
            best_adderm = found_sat

        # Refinement step
        if lower_bound is not None or upper_bound is not None:
            print(f"@MSG@ : Refining search for min(AS) from am: {lower_bound} to {upper_bound}")
            am_count_best, target_result_best_ref = self.refine_search(lower_bound, upper_bound, presolve_result)
            if am_count_best is not None:
                target_result = target_result_best_ref
                best_adderm = am_count_best

        print(f"@MSG@ : Best AM for min(AS) is {best_adderm}")
        return target_result, best_adderm, adder_s_h_zero_best
        
    
    
    def task_done_gurobi_step(self, future, idx, am_count_list, sat_found, manager_sat_found):
        try:
            target_result, satisfiability_loc = future.result()
            am_count = am_count_list[idx]
            print(f"@MSG@: Task done for adderm {am_count} with result {satisfiability_loc}")
            self.update_json_file(number=am_count, result=satisfiability_loc)
            if self.increased_thread_idx == idx:
                self.increased_thread.clear()
            if satisfiability_loc == "sat":
                manager_sat_found.value = True
                if sat_found is not None:
                    sat_found.set()
                self.sat_list[idx] = "sat"
                sat = [i for i, x in enumerate(self.sat_list) if x == "sat"]
                min_sat = min(sat, default=None)
                self.target_result_update(target_result)
                if min_sat == idx:
                    if sat_found is not None:
                        sat_found.set()
                print(f"@MSG@: Adderm {am_count} is SAT")
                # If the previous adderm was UNSAT, we can stop further processing
                # Cancel all futures beyond the completed one
                for i in range(idx + 1, len(self.futures)):
                    future_to_cancel = self.futures[i]
                    if future_to_cancel is not None and not future_to_cancel.done():
                        if not future_to_cancel.cancel():
                            print(f"@MSG@: Could not cancel future for adderm {am_count_list[i]}, This is weird, you might have to wait.")

                if idx > 0: 
                    if self.sat_list[idx - 1] == "unsat":
                        print(f"@MSG@: Transition from UNSAT to SAT found, stopping remaining tasks.")
                        # Cancel remaining futures
                        for i in range(0, len(self.futures)):
                            future_to_cancel = self.futures[i]
                            if future_to_cancel is not None and not future_to_cancel.done():
                                if not future_to_cancel.cancel():
                                    print(f"@MSG@: Could not cancel future for adderm {am_count_list[i]}, This is weird, you might have to wait.")
            else:
                self.sat_list[idx] = "unsat"
                print(f"@MSG@: Adderm {am_count} is UNSAT")

                for i in range(0, idx-1):
                   future_to_cancel = self.futures[i]
                   if future_to_cancel is not None and not future_to_cancel.done():
                       if not future_to_cancel.cancel():
                           print(f"@MSG@: Could not cancel future for adderm {am_count_list[i]}, This is weird, you might have to wait.")

                if idx < len(self.sat_list) - 1: 
                    if self.sat_list[idx + 1] == "sat":
                        print(f"@MSG@: Transition from UNSAT to SAT found, stopping step search.")
                        # Cancel remaining futures
                        for i in range(0, len(self.futures)):
                            future_to_cancel = self.futures[i]
                            if future_to_cancel is not None and not future_to_cancel.done():
                                if not future_to_cancel.cancel():
                                    print(f"@MSG@: Could not cancel future for adderm {am_count_list[i]}, This is weird, you might have to wait.")

            self.all_done[idx].set()  
            print(f"@MSG@: Done for adderm {am_count}")
                
        except CancelledError:
            self.update_json_file(number=am_count_list[idx], result='Cancelled')
            # print(f"Task at AM {am_count_list[idx]} was cancelled.")
            self.all_done[idx].set()
        except TimeoutError:
            print(f"Task at AM {am_count_list[idx]} timed out.")
            self.all_done[idx].set()

        except ProcessExpired as error:
            print(f"Task at AM {am_count_list[idx]} raised a ProcessExpired error: {error}")
            self.all_done[idx].set()
        except Exception as e:
            print(f"@MSG@: Exception in callback for AM {am_count_list[idx]}: {e}")
            self.all_done[idx].set()
            traceback.print_exc()
        
    def config_gurobi(self):
        """Configure the Gurobi solver."""
        config = {
            "filter_type": self.filter_type,
            "half_order": self.half_order,
            "xdata": self.xdata,
            "upperbound_lin": self.upperbound_lin,
            "lowerbound_lin": self.lowerbound_lin,
            "ignore_lowerbound": self.ignore_lowerbound,
            "adder_count": self.adder_count,
            "wordlength": self.wordlength,
            "adder_depth": self.adder_depth,
            "avail_dsp": self.avail_dsp,
            "adder_wordlength_ext": self.adder_wordlength_ext,
            "gain_upperbound": self.gain_upperbound,
            "gain_lowerbound": self.gain_lowerbound,
            "coef_accuracy": self.coef_accuracy,
            "intW": self.intW,
            "gurobi_thread": self.gurobi_thread,
            "worker": self.worker
        }
        return config
    
    def min_as_finder_gurobi(self, presolve_result):
        """Main function to search for the smallest SAT number."""
        # Determine the starting point for the adder count
        if self.am_start is not None:
            am_start = max(self.am_start, presolve_result['min_adderm'])
        else:
            am_start = presolve_result['min_adderm']

        lower_bound = None
        upper_bound = None

        h_zero_count = presolve_result['max_zero']
        target_result_best = self.unload_target_result(deepsearch=False)

        # Configuration for the solver
        config = self.config_gurobi()

        sat_found = multiprocessing.Event()
        self.increased_thread = multiprocessing.Event()
        max_am_count_mult = 20  # Adjust as needed
        am_to_check = list(range(am_start, max_am_count_mult * self.search_step, self.search_step))
        self.sat_list = [None] * len(am_to_check)
        self.sat_list = self.unload_json_file(am_to_check, self.sat_list)
        self.increased_thread_idx = 0
        self.all_done = [multiprocessing.Event() for _ in range(len(am_to_check))]
        manager = multiprocessing.Manager()
        sat_found_flag = manager.Value('b', False)

        with ProcessPool(max_workers=self.worker) as pool:
            self.futures = [None] * len(am_to_check)
            for idx, number in enumerate(am_to_check):
                thread = self.gurobi_thread // self.worker
                if not self.increased_thread.is_set():
                    self.increased_thread.set()
                    self.increased_thread_idx = idx
                    thread += self.gurobi_thread % self.worker
                if self.sat_list[idx] is None:
                    # Schedule the task
                    future = pool.schedule(
                        try_asserted,
                        args=(config, presolve_result, number, h_zero_count, thread, sat_found_flag),
                        timeout=self.timeout
                    )
                    self.futures[idx] = future

                    # Add the callback using functools.partial to pass extra arguments
                    callback_func = partial(
                        self.task_done_gurobi_step,
                        idx=idx,
                        am_count_list=am_to_check,
                        sat_found=sat_found,
                        manager_sat_found=sat_found_flag
                    )
                    future.add_done_callback(callback_func)
                else:
                    self.futures[idx] = None

            # Wait for all scheduled futures to complete
            done_futures = [f for f in self.futures if f is not None]
            wait(done_futures, return_when=ALL_COMPLETED)

            for i, done in enumerate(self.all_done):
                done.wait()

            sat_list = []
            unsat_list = []
            for i, sat_val in enumerate(self.sat_list):
                if sat_val == "sat":
                    sat_list.append(am_to_check[i])
                elif sat_val == "unsat":
                    unsat_list.append(am_to_check[i])


        # Refinement step if SAT was found
        smallest_sat = None
        target_result_best = None
        lower_bound = None
        upper_bound = None
        if sat_found.is_set():
            if unsat_list:
                lower_bound = (max(unsat_list) + 1)
            else:
                lower_bound = 0
            
            upper_bound = min(sat_list)-1
            smallest_sat = min(sat_list)
            target_result_best = self.unload_target_result()
        else:
            raise RuntimeError("No SAT found in the AS step Search.")

        print(f"@MSG@: The smallest SAT adderm is: {smallest_sat}")

        return lower_bound, upper_bound, smallest_sat, target_result_best


    def task_done_refine(self, futures, pools, am_count_list, sat_found= None):
        def callback(future):
            try:
                target_result, satisfiability_loc = future.result()  # blocks until results are ready
                # Find the index of the current future
                completed_index = futures.index(future)
                am_count = am_count_list[completed_index]
                print(f"@MSG@ : Task done for number {am_count} with result {satisfiability_loc}")

                self.update_json_file(number=am_count, result=satisfiability_loc)
                if satisfiability_loc == "sat":
                    if sat_found is not None:
                        sat_found.set()
                    self.sat_list[completed_index] = "sat"
                    
                    print(f"sat_list: {self.sat_list}")
                    print(f"@MSG@ : {am_count} is SAT") 
                    # If the task is satisfiable, check if the previous task was unsatisfiable
                    if completed_index != 0:
                        # If the previous task was unsatisfiable, stop all pools
                        if self.sat_list[completed_index - 1] == "unsat":
                            print(f"Stopping all pools")
                            for idx, f in enumerate(futures):
                                if f is not None:
                                    if not f.done():
                                        if not f.cancel():
                                            # Stop the pool associated with this future
                                            print(f"Stopping pool for future at index {idx}.")
                                            if pools is not None:
                                                pools[idx].stop()

                    # Cancel all futures beyond the completed one
                    for idx, f in enumerate(futures[completed_index + 1:], start=completed_index + 1):
                        if f is not None:
                            if not f.done():
                                if not f.cancel():
                                    # Stop the pool associated with this future
                                    print(f"Stopping pool for future at index {idx}.")
                                    if pools is not None:
                                        pools[idx].stop()
                else:
                    # If the task is unsatisfiable, update the previous unsat am_count
                    print(f"@MSG@ : sat_list: {self.sat_list}, completed_index: {completed_index}")
                    self.sat_list[completed_index] = "unsat"
                    print(f"@MSG@ : {am_count} is UNSAT") 
                    if completed_index != len(self.sat_list) - 1:
                        if self.sat_list[completed_index + 1] == "sat":
                            print(f"Stopping all pools")
                            for idx, f in enumerate(futures):
                                if f is not None:
                                    if not f.done():
                                        if not f.cancel():
                                            # Stop the pool associated with this future
                                            print(f"Stopping pool for future at index {idx}.")
                                            if pools is not None:
                                                pools[idx].stop()
                        
                    # Cancel all futures before the completed one
                    for idx, f in enumerate(futures[:completed_index - 1], start=0):
                        if f is not None:
                            if not f.done():
                                if not f.cancel():
                                    # Stop the pool associated with this future
                                    print(f"Stopping pool for future at index {idx}.")
                                    if pools is not None:
                                                pools[idx].stop()

                self.all_done[completed_index].set()

            except CancelledError:
                completed_index = futures.index(future)
                am_count = am_count_list[completed_index]
                self.update_json_file(number=am_count, result='Cancelled')
                print(f"Task at index {completed_index} was cancelled.")
                self.all_done[completed_index].set()
            except TimeoutError:
                completed_index = futures.index(future)
                print(f"Task at index {completed_index} timed out.")
                self.all_done[completed_index].set()

            except ProcessExpired as error:
                completed_index = futures.index(future)
                print(f"Task at index {completed_index} raised a ProcessExpired error: {error}")
                self.all_done[completed_index].set()
            except Exception as error:
                print(f"Task raised an exception: {error}")
                traceback.print_exc()  # Print the full traceback to get more details
                self.all_done[completed_index].set()

        return callback

    def refine_search(self, lower_bound, upper_bound, presolve_result):
        """Refine the search between lower_bound and upper_bound using binary search with n parallel workers."""
        am_count_best = None
        target_result_best = None

        if lower_bound > upper_bound:
            print(f"@MSG@ : No Search Space to refine, returning...")
            return None, None

        # Use a single pool with max_workers=self.worker
        pools = [ProcessPool(max_workers=1) for _ in range(self.worker)]
        res = [None for i in range(lower_bound, upper_bound+1)]
        original_lower_bound = lower_bound

        h_zero_count = presolve_result['max_zero']
        target_result = None
        config = self.config_gurobi()

        
        try:
            while lower_bound <= upper_bound:
                used_worker = 0
                # Compute midpoints
                midpoints = [lower_bound + (upper_bound - lower_bound) * (i + 1) / (self.worker + 1) for i in range(self.worker)]
                midpoints = list(map(math.floor, midpoints))                
                # Remove duplicates and ensure midpoints are within bounds
                midpoints = sorted(set(midpoints))
                midpoints = [m for m in midpoints if lower_bound <= m <= upper_bound]
                print(f"Testing midpoints: {midpoints}")
                self.all_done = None
                self.all_done = [multiprocessing.Event() for _ in range(len(midpoints))]


                if not midpoints:
                    break  # No midpoints to test

                for idx, _ in enumerate(midpoints):
                    used_worker += 1
                print(f"Used worker: {used_worker}")
                self.sat_list = [None for _ in range(len(midpoints))]
                self.sat_list = self.unload_json_file(midpoints, self.sat_list)

                # Print the midpoints that need to be checked
                midpoints_to_print = [mid for i, mid in enumerate(midpoints) if self.sat_list[i] is None]
                print(f"@MSG@ : Refine Search midpoints: {midpoints_to_print}")

                # Schedule tasks
                futures = []
                for idx, am_count_loc in enumerate(midpoints):
                    if self.sat_list[idx] is not None:
                        continue
                    thread = self.gurobi_thread // used_worker
                    # If the last id is reached, use the remaining threads
                    if idx == len(midpoints):
                        thread += self.gurobi_thread % used_worker

                    future = pools[idx].schedule(try_asserted, args=(config, presolve_result, am_count_loc,h_zero_count, thread,), timeout=self.timeout)
                    futures.append((future))


                for future in futures:
                    future.add_done_callback(self.task_done_refine(futures, pools, midpoints))

                # Wait for all futures to complete
                done, not_done = wait(futures, return_when=ALL_COMPLETED)

                for all_done in self.all_done:
                    all_done.wait()
                
            

                # Collect results
                sat_am_counts = []
                unsat_am_counts = []
                min_sat = min([i for i, x in enumerate(self.sat_list) if x == "sat"], default=None)
                for future in futures:
                    try:
                        target_result, satisfiability_loc = future.result()
                        am_count = midpoints[futures.index(future)]
                        am_count_idx = futures.index(future)
                        res_index = am_count - original_lower_bound
                        res[res_index] = satisfiability_loc
                        if satisfiability_loc == "sat":
                            if am_count_idx == min_sat:
                                target_result_best = target_result
                                self.target_result_update(target_result_best)
                                am_count_best = am_count
                            sat_am_counts.append(am_count)
                        else:
                            unsat_am_counts.append(am_count)
                            
                    except CancelledError:
                        am_count = midpoints[futures.index(future)]
                        res_index = am_count - original_lower_bound
                        res[res_index] = "Cancelled"

                    except Exception as e:
                        print(f"Task raised an exception: {e}")
                        traceback.print_exc()

                if sat_am_counts:
                    # Found SAT at midpoints
                    min_sat = min(sat_am_counts)
                    # Adjust upper_bound
                    upper_bound = min_sat - 1

                if unsat_am_counts:
                    max_unsat = max(unsat_am_counts)
                    lower_bound = max_unsat + 1

                if not sat_am_counts and not unsat_am_counts:
                    # No results found, break the loop
                    raise Exception("Somehow no results found in the search space.")
                
                for res_idx, res_val in enumerate(res):
                    print(f"res_idx: {res_idx}, res_val: {res_val}")
                    if res_val == True:
                        if res[res_idx-1] == False:
                            found_sat = res_idx + original_lower_bound
                            upper_bound = lower_bound - 1
                            print(f"Found SAT at {found_sat}, breaking the loop")
                            break
                        break
                        

                # Check if lower_bound > upper_bound
                if lower_bound > upper_bound:
                    break

        finally:
            # Ensure the pool is properly closed
            for pool in pools:
                pool.close()
                pool.join()

        return am_count_best, target_result_best


    


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
    
    def pysat_instance_creator(self):
        pysat_instance = FIRFilterPysat(
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
                    self.intW
                    )
        
        return pysat_instance


if __name__ == "__main__":
    print("MainProblem: can only be run from main backend")
