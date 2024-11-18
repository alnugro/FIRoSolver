from pebble import ProcessPool, ProcessExpired, ThreadPool
from concurrent.futures import TimeoutError, CancelledError, wait, ALL_COMPLETED
import traceback
import time
import copy
import numpy as np
import math
import json
from filelock import FileLock
import ast

try:
    from .formulation_pysat import FIRFilterPysat
    from .formulation_z3_pbsat import FIRFilterZ3
    from .formulation_gurobi import FIRFilterGurobi
    from .solver_func import SolverFunc
    from .rat2bool import Rat2bool

except:
    from formulation_pysat import FIRFilterPysat
    from formulation_z3_pbsat import FIRFilterZ3
    from formulation_gurobi import FIRFilterGurobi
    from solver_func import SolverFunc
    from rat2bool import Rat2bool


class Presolver:
    def __init__(self, input_data):
        # Explicit declaration of instance variables with default values (if applicable)
        self.filter_type = None
        self.order_upperbound = None

        self.continue_solver = None

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

        self.xdata_presolve = None
        self.upperbound_lin_presolve = None
        self.lowerbound_lin_presolve = None

        self.cutoffs_upper_ydata_lin = None
        self.cutoffs_lower_ydata_lin = None

        self.half_order = None
        self.problem_id = None


        # Dynamically assign values from input_data, skipping any keys that don't have matching attributes
        for key, value in input_data.items():
            if hasattr(self, key):  # Only set attributes that exist in the class
                setattr(self, key, value)

        self.xdata = self.xdata_presolve
        self.upperbound_lin = self.upperbound_lin_presolve
        self.lowerbound_lin = self.lowerbound_lin_presolve

        self.order_current = None

        self.min_order = None
        self.min_gain = None
        self.h_res = None
        self.gain_res = None

        self.gain_res = None
        self.h_res = None
        self.satisfiability = 'unsat'

        self.max_zero_reduced = 0

    def presolve_update(self, presolve_result):
        JSON_FILE = 'problem_description.json'
        LOCK_FILE = JSON_FILE + '.lock'
        lock = FileLock(LOCK_FILE)

        target_pre = copy.deepcopy(presolve_result)


        with lock:
            with open(JSON_FILE, 'r') as f:
                data = json.load(f)

                problem_id = str(self.problem_id)
                data[problem_id].update(
                    {'presolve_result': target_pre}
                )
                with open(JSON_FILE, 'w') as f:
                    json.dump(data, f, indent=4, sort_keys=True)

    def load_presolve(self):
        JSON_FILE = 'problem_description.json'
        LOCK_FILE = JSON_FILE + '.lock'
        lock = FileLock(LOCK_FILE)

        with lock:
            with open(JSON_FILE, 'r') as f:
                data = json.load(f)
        try:
            problem_id = str(self.problem_id)
            presolve_result = {}
            for key, value in data[problem_id]['presolve_result'].items():
                presolve_result[key] = self.convert_nested_list(value)
        except KeyError:
            print(f"@MSG@ : Presolve result not found in JSON file for problem ID {self.problem_id}, running presolve again.")
            presolve_result = None

        return presolve_result
    
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

    def minmax_h_zero_worker_func(self,input_m):
        """Function to be executed in parallel."""
        print(f"running with input :{input_m}")

        gurobi_instance = FIRFilterGurobi(
            self.filter_type, 
            self.half_order, #you pass upperbound directly to gurobi
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
        
        target_result_max = gurobi_instance.run_barebone_real(self.gurobi_thread//2, 'maximize_h',self.max_h_zero_for_minmax, input_m)
        target_result_min = gurobi_instance.run_barebone_real(self.gurobi_thread//2, 'minimize_h',self.max_h_zero_for_minmax, input_m)
        
        print(f"input is done:{input_m}")

        return target_result_max, target_result_min,input_m
    
    def minmax_h_worker_func(self,input_m):
        """Function to be executed in parallel."""
        print(f"running with input :{input_m}")

        gurobi_instance = FIRFilterGurobi(
            self.filter_type, 
            self.half_order, #you pass upperbound directly to gurobi
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
        
        target_result_max = gurobi_instance.run_barebone_real(self.gurobi_thread//2, 'maximize_h_without_zero',None, input_m)
        target_result_min = gurobi_instance.run_barebone_real(self.gurobi_thread//2, 'minimize_h_without_zero',None, input_m)
        
        print(f"input is done:{input_m}")

        return target_result_max, target_result_min,input_m

    def run_minmax_h_zero_threadpool(self, half_order_list):
        with ThreadPool(2) as pool:
            future = pool.map(self.minmax_h_zero_worker_func, half_order_list)
            iterator = future.result()
            try:
                for res in iterator:
                    target_result_max, target_result_min, input_m = res
                    self.h_max[input_m]= (target_result_max['target_h_res'])
                    self.h_min[input_m]= (target_result_min['target_h_res'])
            except Exception as error:
                print("Error:", error)

    def run_minmax_h_threadpool(self, half_order_list):
        with ThreadPool(2) as pool:
            future = pool.map(self.minmax_h_worker_func, half_order_list)
            iterator = future.result()
            try:
                for res in iterator:
                    target_result_max, target_result_min, input_m = res
                    self.h_max_without_zero[input_m]= (target_result_max['target_h_res'])
                    self.h_min_without_zero[input_m]= (target_result_min['target_h_res'])
            except Exception as error:
                print("Error:", error)



    def run_presolve_gurobi(self, h_zero_input = None):
        presolve_result = {}

        if self.continue_solver:
            presolve_result = self.load_presolve()
            if presolve_result:
                print(f"@MSG@ : Presolve result loaded from JSON file for problem ID {self.problem_id}.")
                max_zero = presolve_result['max_zero']
                min_adderm = presolve_result['min_adderm']
                print(f"@MSG@ : Loaded max zero: {max_zero}, min adder count: {min_adderm}")
                return presolve_result
            else:
                presolve_result = {}


        gurobi_instance = FIRFilterGurobi(
            self.filter_type, 
            self.half_order, #you pass upperbound directly to gurobi
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
            target_result = gurobi_instance.run_barebone(self.gurobi_thread, 'find_max_zero')

            #get result data
            if target_result['satisfiability'] == 'unsat':
                raise ValueError("problem is unsat")
            
            max_h_zero = target_result['max_h_zero']
            
            print("max h zero: ", max_h_zero)

            #decrease the h_zero by one if its not satisfiable
            solve_with_h_zero_sum_sat_flag = False
            while solve_with_h_zero_sum_sat_flag == False:
                target_result_temp = gurobi_instance.run_barebone(self.gurobi_thread,'try_h_zero_count' ,max_h_zero)
                if target_result_temp['satisfiability'] == 'unsat':
                    max_h_zero -= 1
                    print("\n.......calculated h_zero was not satisfiable......\n")
                    self.max_zero_reduced +=1
                    if max_h_zero < 0:
                        raise ValueError("problem is unsat")

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
        target_result = []



        print("\nFinding Minimum and Maximum For each Filter Coefficients......\n")
        half_order_list = [m for m in range(self.half_order)]
        self.h_max = []
        self.h_min = []

        self.max_h_zero_for_minmax = max_h_zero

        self.h_max = [None for m in range(self.half_order)] 
        self.h_min = [None for m in range(self.half_order)]
        #run h_zero minmax finder using threadpool
        self.run_minmax_h_zero_threadpool( half_order_list)
        min_adderm = self.min_adderm_finder(self.h_max,self.h_min)

        print(f"@MSG@ : Found max zero: {max_h_zero}, min adder count: {min_adderm}")
        presolve_result.update({
            'max_zero' : max_h_zero,
            'min_gain' : min_gain,
            'hmax' : self.h_max,
            'hmin' : self.h_min,
            'max_zero_reduced' : self.max_zero_reduced,
            'h_res':self.h_res,
            'gain_res': self.gain_res,
            'min_adderm': min_adderm,

        })
        # print(presolve_result)

        self.max_h_zero_for_minmax = None
        self.presolve_update(presolve_result)

        return presolve_result
    
    def run_presolve_z3_pysat(self):
        presolve_result = {}


        if self.start_with_error_prediction == False:
            #try satisfiability first if error pred havent run
            print("run_presolve_z3_pysat")
            #check first if the problem is even satisfiable
            self.execute_z3_pysat()

            if self.satisfiability == 'unsat':
                raise ValueError("problem is unsat")
        
      
        #binary search to find the h_zero count sat transition point
        low, high = 0, self.half_order
        iteration = 1
        max_h_zero = -1  # Default value if no 'sat' is found
        
        while low <= high:            
            #reset all inputs first
            self.satisfiability = None
            self.gain_res = None
            self.h_res = None

            print(f"iteration: {iteration}")
            iteration += 1

            mid = (low + high) // 2

            print(f"checking mid: {mid}")
            self.execute_z3_pysat('try_h_zero_count' ,mid)

            if self.satisfiability == 'sat':
                max_h_zero = mid  # Update max_zero to the current 'sat' index
                gain_res_at_max = self.gain_res
                h_res_at_max = self.h_res
                low = mid + 1

            elif self.satisfiability == 'unsat':
                high = mid - 1
            else:
                raise TypeError("run_presolve_z3_pysat: Problem should be either sat or unsat, this should never happen contact developer")

        presolve_result.update({
            'max_zero' : max_h_zero,
            'min_gain' : None,
            'hmax' : None,
            'hmin' : None,
            'max_zero_reduced' : None,
            'h_res': h_res_at_max,
            'gain_res': gain_res_at_max,

            'min_gain_without_zero' : None,
            'hmax_without_zero' : None,
            'hmin_without_zero' : None,
            'h_res_without_zero': None,
        })

        print(presolve_result)

        return presolve_result
        

    def z3_presolve(self, solver_options=None, h_zero_count= None):

        satisfiability, h_res ,gain= self.z3_instance_creator().run_barebone(self.z3_thread, solver_options,h_zero_count)
        
        return satisfiability, h_res ,gain
    
    def pysat_presolve(self, solver_id, solver_options=None, h_zero_count= None):
        if h_zero_count == None:
            h_zero_count_loc = 0
        else: h_zero_count_loc = h_zero_count
        satisfiability, h_res ,gain= self.pysat_instance_creator().run_barebone(solver_id,h_zero_count_loc)
        
        return satisfiability, h_res ,gain
    
    
    def execute_z3_pysat(self, solver_options=None,h_zero_count = None):
        pools = []  # To store active pools for cleanup
        futures_z3 = []  # List to store Z3 futures
        futures_pysat = []  # List to store PySAT futures
        

        try:
            # Conditionally create the Z3 pool
            if self.z3_thread > 0:
                pool_z3 = ProcessPool(max_workers=1)
                pools.append(pool_z3)
                future_single_z3 = pool_z3.schedule(self.z3_presolve, args=(solver_options, h_zero_count,), timeout=self.timeout)
                futures_z3.append(future_single_z3)
                    
            else:
                pool_z3 = None

            # Conditionally create the PySAT pool
            if self.pysat_thread > 0:
                pool_pysat = ProcessPool(max_workers=self.pysat_thread)
                pools.append(pool_pysat)
                for i in range(self.pysat_thread):
                    future_single_pysat = pool_pysat.schedule(self.pysat_presolve, args=(i , solver_options, h_zero_count,), timeout=self.timeout)
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
                    satisfiability, h_res, gain_res = future.result()
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
                satisfiability, h_res, gain_res  = future.result()  # blocks until results are ready
                print(f"{solver_name} task done")
                
                # Cancel all other processes for this solver (only within the same group)
                for f in futures:
                    if f is not future and not f.done():  # Check if `f` is a `Future`
                        if not f.cancel():
                        # If cancel() doesn't work, forcefully stop the corresponding pool
                            print(f"{solver_name} process couldn't be cancelled. Force stopping the pool.")
                            solver_pools[solver_name].stop()
                


                self.satisfiability = satisfiability
                self.gain_res = gain_res
                self.h_res = h_res

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
                    self.half_order, 
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
    
    def pysat_instance_creator(self):
        pysat_instance = FIRFilterPysat(
                    self.filter_type, 
                    self.half_order, 
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
                    self.intW
                    )
        
        return pysat_instance
    
    def min_adderm_finder(self, h_res_max ,h_res_min = None):
        wordlength = self.wordlength
        fracW = wordlength - self.intW
        r2b = Rat2bool()
        h_res_max_array = np.array(h_res_max)
        h_res_min_array = np.array(h_res_min)
        h_res_max_array *= 2**fracW
        h_res_min_array *= 2**fracW
        print(f"h_res_max: {h_res_max_array}")
        print(f"h_res_min: {h_res_min_array}")
        h_res_max_round = []
        h_res_min_round = []
        for i in range(len(h_res_max_array)):
            h_res_max_round.append(math.floor(h_res_max_array[i]))
            h_res_min_round.append(math.ceil(h_res_min_array[i]))

        
        for i in range(len(h_res_max_round)):
            if h_res_max_round[i] < 0 and h_res_min_round[i] < 0:
                min = h_res_max_round[i]
                max = h_res_min_round[i]
                h_res_max_round[i] = abs(max)
                h_res_min_round[i] = abs(min)
            elif h_res_max_round[i] >= 0 and h_res_min_round[i] < 0:
                if abs(h_res_max_round[i]) > abs(h_res_min_round[i]):
                    h_res_max_round[i] = abs(h_res_max_round[i])
                    h_res_min_round[i] = 0
                else:
                    h_res_max_round[i] = 0
                    h_res_min_round[i] = abs(h_res_min_round[i])
            else:
                pass
        print(f"max h_res_round: {h_res_max_round}")
        print(f"min h_res_round: {h_res_min_round}")     

        
        o_list = [[] for i in range(len(h_res_max))]
        a_cost = [None for i in range(len(h_res_max))]
        one_included = False
        for i in range(len(h_res_max)):
            for j in range(h_res_min_round[i], h_res_max_round[i]+1):
                if j == 0:
                    a_cost[i] = 0
                    o_list[i].append(0)
                    continue
                odd = 0
                while True:
                    if j % 2 == 0:
                        j = j // 2
                    else:
                        odd = j
                        break
                if odd == 1:
                    a_cost[i] = 0
                    o_list[i].append(1)
                    one_included = True
                    continue
                
                o_list[i].append(odd)

        print(f"o_list: {o_list}")


        for i in range(len(o_list)):
            if a_cost[i] == 0:
                continue
            if o_list[i] == []:
                a_cost[i] = 0
                continue
            a_cost[i] = 1
            for j in range(len(o_list)):
                if i == j:
                    continue
                common = [item for item in o_list[i] if item in o_list[j]]
                if common:
                    # print(f"Common elements between o_list[{i}] and o_list[{j}]: {common}")
                    a_cost[i] = 0
                    break

        print(f"a_cost: {a_cost}")
        muq_sum = sum(a_cost)
        print(f"muq_sum: {muq_sum}")

        if one_included:
            print(f"am: {muq_sum}")
            return muq_sum
            
        csd_min = 100000000

        for i in range(len(o_list)):
            for j in range(len(o_list[i])):
                csd = r2b.frac2csd([o_list[i][j]], self.wordlength, 0)[0]
                # print(f"csd of {o_list[i][j]}: {csd}")
                csd_sum = 0
                for csd_bin in csd:
                    if csd_bin != 0:
                        csd_sum += 1
                if csd_sum < csd_min:
                    csd_min = csd_sum

        if csd_min-1 < 0 or csd_min == 100000000:
            csd_min = 0
        else:
            csd_min -= 1

        am = csd_min + muq_sum
        return am




if __name__ == "__main__":
    # Test inputs
    filter_type = 0
    order_current = 7
    accuracy = 4
    wordlength = 17
    gain_upperbound = 1
    gain_lowerbound = 1
    coef_accuracy = 5
    intW = 6

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


    passband_error = 0.1
    stopband_error = 0.1
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
    }

    # Create an instance of SolverBackend
    presolver = Presolver(input_data)
    # presolver.run_presolve_gurobi()
    presolve_result= presolver.run_presolve_gurobi()
    min_adderm = presolver.min_adderm_finder(presolve_result['hmax'],presolve_result['hmin'])
