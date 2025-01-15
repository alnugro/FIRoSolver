# solver_runner.py
import sys
import json
import os
import traceback
import numpy as np
import json
import numpy as np
import os
import signal
import time
from filelock import FileLock




try:
    from backend_main import SolverBackend
except:
    from backend.backend_main import SolverBackend

class BackendRunner:
    def __init__(self, solver_input):
        # Explicit declaration of instance variables with default values (if applicable)
        self.filter_type = None
        self.order_upperbound = None
        self.continue_solver = None

        self.ignore_lowerbound = None
        self.wordlength = None
        self.adder_depth = None
        self.avail_dsp = None
        self.adder_wordlength_ext = None
        self.gain_upperbound = None
        self.gain_lowerbound = None
        self.coef_accuracy = None
        self.intW = None

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

        self.deepsearch = None
        self.problem_id = None
        


        # Dynamically assign values from input_data, skipping any keys that don't have matching attributes
        for key, value in solver_input.items():
            if hasattr(self, key):  # Only set attributes that exist in the class
                setattr(self, key, value)
        


        self.solver_input = solver_input

        self.solver_name = None

        if self.gurobi_thread > 0:
            self.solver_name = 'Gurobi'
        if self.pysat_thread > 0:
            self.solver_name = 'pysat'
        if self.z3_thread > 0:
            self.solver_name = 'z3'

        self.result_id = None
        self.continue_status = None

        

    def run_solvers(self):
        start_time = time.time()

        print(f"@MSG@ : Running solver")
        # time.sleep(40)
        #initiate backend

        if self.continue_solver:
            print(f"@MSG@ : Continue solver is True, checking if problem is already solved")
            self.continue_status = self.check_solved()
            if self.continue_status == 0:
                print(f"@MSG@ : Problem already solved with min(AS) as the best solution and no leaks")
            elif self.continue_status == 1:
                print(f"@MSG@ : Problem already solved with min(AS) as the best solution with leaks, trying to patch leaks")
            elif self.continue_status == 2:
                print(f"@MSG@ : Problem already solved with optimal as the best solution and no leaks")
                return
            elif self.continue_status == 3:
                print(f"@MSG@ : Problem already solved with optimal as the best solution with leaks, trying to patch leaks")
            else:
                print(f"@MSG@ : Problem not solved yet, continuing solver")


        backend = SolverBackend(self.solver_input)
        if self.gurobi_thread > 1:
            print(f"@MSG@ : Gurobi is chosen, running compatibility test")            
            backend.gurobi_test()
            print(f"@MSG@ : Gurobi compatibility test done")

        if self.start_with_error_prediction:
            print(f"@MSG@ : Running error prediction")
            backend.error_prediction()

        print(f"@MSG@ : Running presolve")
        
        # start presolve
        presolve_result = backend.solver_presolve()
        min_am = presolve_result['min_adderm']
        h_zero_max = presolve_result['max_zero']
        print(f"@MSG@ : Presolve done, min(AM) = {min_am}, h_zero_max = {h_zero_max}")

    
        print(f"@MSG@ : Presolve done")
        
        result1_valid = False
        
        print(f"@MSG@ : Finding best min(AS)")
        target_result, best_adderm ,total_adder, adder_s_h_zero_best = backend.find_best_adder_s(presolve_result)
        
        if self.continue_status == 1 or self.continue_status == -1 or self.continue_solver == False:
            target_result.update({
            'option' : 'min(AS)'
            })
            while True:
                leaks, leaks_mag = backend.result_validator(target_result['h_res'],target_result['gain'])
                if leaks:

                    self.update_json_with_lock(target_result , 'result_leak.json')

                    print(f"@MSG@ : While solving for min(AS): leak found, patching leak...")
                    backend.patch_leak(leaks, leaks_mag)

                    target_result_temp, satisfiability = backend.solving_result_barebone(presolve_result,best_adderm,adder_s_h_zero_best)
                    if satisfiability == 'unsat':
                        print(f"@MSG@ : While solving for min(AS): problem is unsat from asserting the leak to the problem")
                        backend.revert_patch()
                        break
                    else:
                        target_result = target_result_temp
                        target_result.update({
                            'option' : 'min(AS)'
                        })
                else:
                    print(f"@MSG@ : min(AS): Result Validated, no leaks found")
                    result1_valid = True
                    self.update_json_with_lock( target_result , 'result_valid.json')
                    break
        
        print(f"@MSG@ : min(AS) found")
        
        if self.deepsearch and self.gurobi_thread > 0:
            print(f"@MSG@ : Starting deep search")

            # Packing variables into a dictionary 
            data_dict = {
            'best_adderm_from_s': int(best_adderm),
            'total_adder_s': int(total_adder),
            'adder_s_h_zero_best': int(adder_s_h_zero_best),
            }

            if int(adder_s_h_zero_best) == 0:
                print(f"@MSG@ : Deep Search canceled, h_zero_count is 0, no search space available")
                end_time = time.time()
                duration = end_time - start_time
                print(f"@MSG@ : Solver finished in {duration} seconds")
                self.solver_done_json()
                return

           
            target_result3, best_adderm3, total_adder3, h_zero_best3 = backend.deep_search_adder_total(presolve_result, data_dict)
            

            if h_zero_best3 == None: #if all unsat
                print(f"@MSG@ : Deep Search finished, all search space unsat: Taking min(AS) as the best solution")
                best_adderm3 = best_adderm
                total_adder3 = total_adder
                target_result3 = target_result  

                target_result3.update({
                                'option' : 'optimal'
                            })
                
                if result1_valid:
                    self.update_json_with_lock( target_result3 , 'result_valid.json')
                else:
                    self.update_json_with_lock( target_result3 , 'result_leak.json')

                end_time = time.time()
                duration = end_time - start_time
                print(f"@MSG@ : Solver finished in {duration} seconds")
                self.solver_done_json()
                return
            else:
                #if its sat then take it as optimal
                target_result3.update({
                    'option' : 'optimal'
                })
                
                while True:
                    leaks, leaks_mag = backend.result_validator(target_result3['h_res'],target_result3['gain'])
                    
                    if leaks:
                        self.update_json_with_lock(target_result3 , 'result_leak.json')

                        print(f"@MSG@ : While solving the optimal result: leak found, patching leak...")
                        backend.patch_leak(leaks, leaks_mag)

                        target_result_temp3, satisfiability = backend.solving_result_barebone(presolve_result,best_adderm3,h_zero_best3)
                        if satisfiability == 'unsat':
                            print(f"@MSG@ : While solving the optimal result: problem is unsat from asserting the leak to the problem")
                            backend.revert_patch()
                            break
                        else:
                            target_result3 = target_result_temp3
                            target_result3.update({
                                'option' : 'optimal'
                            })
                    else:
                        print(f"@MSG@ :optimal result: Result Validated, no leaks found")
                        self.update_json_with_lock( target_result3 , 'result_valid.json')
                        break
        end_time = time.time()
        self.solver_done_json()
        duration = end_time - start_time
        print(f"@MSG@ : Solver finished in {duration} seconds")

        
        
    def solver_done_json(self):
        JSON_FILE = 'problem_description.json'
        LOCK_FILE = JSON_FILE + '.lock'
        lock = FileLock(LOCK_FILE)
        data = None
        try:
            with lock:
                with open(JSON_FILE, 'r') as f:
                    data = json.load(f)
                
                problem_id = str(self.problem_id)
                if problem_id not in data:
                    raise ValueError(f"Problem ID {problem_id} not found in {JSON_FILE}")
                
                data[problem_id].update({
                    'done': True
                })

                with open(JSON_FILE, 'w') as f:
                    json.dump(data, f, indent=4)

        except:
            raise ValueError(f"No problem_description.json found, this is impossible")


        
            
    def update_json_with_lock(self, new_data, filename):
        
        new_data.update({
                'problem_id': self.problem_id,
                'solver' : self.solver_name,
                'avail_dsp' : self.avail_dsp,
                'filter_type' : self.filter_type,
                'adder_wordlength_ext' : self.adder_wordlength_ext,
                        })
        lock_filename = filename + '.lock'
        lock = FileLock(lock_filename)
        key = None

        with lock:  # Acquire lock before file access
            # Check if the file exists before reading
            if os.path.exists(filename):
                # Load current data from the file
                with open(filename, 'r') as json_file:
                    current_data = json.load(json_file)

                # Find the largest key and set the new key as largest_key + 1
                if current_data:
                    largest_key = max(map(int, current_data.keys()))  # Convert keys to integers
                    key = largest_key + 1
                else:
                    key = 0  # If the file is empty, start with key 0
            else:
                # If the file does not exist, start with an empty dictionary and key 0
                current_data = {}
                key = 0

            # Add new data under the determined key
            current_data[str(key)] = new_data  # Ensure the key is a string for JSON compatibility


            if filename == 'result_valid.json':
                if new_data['option'] == 'min(AS)':
                    result_key = 'min(AS)'
                elif new_data['option'] == 'optimal':
                    result_key = 'optimal'
                else:
                    result_key = 'unknown'
                result_key += '_valid'
                self.update_problem_json_file(key, result_key)
            elif filename == 'result_leak.json':
                if new_data['option'] == 'min(AS)':
                    result_key = 'min(AS)'
                elif new_data['option'] == 'optimal':
                    result_key = 'optimal'
                else:
                    result_key = 'unknown'
                result_key += '_leak'
                self.update_problem_json_file(key, result_key)

            # Save the updated dictionary back to the file
            with open(filename, 'w') as json_file:
                json.dump(current_data, json_file, indent=4)
    
    def update_problem_json_file(self, result_id, result_key):
        JSON_FILE = 'problem_description.json'
        LOCK_FILE = JSON_FILE + '.lock'
        lock = FileLock(LOCK_FILE)

        with lock:
            with open(JSON_FILE, 'r') as f:
                data = json.load(f)

            problem_id = str(self.problem_id)

            # Ensure `problem_id` and `as_results` are initialized
            if problem_id not in data:
                raise ValueError(f"Problem ID {problem_id} not found in {JSON_FILE}")
            
            data[problem_id].update({
                str(result_key): str(result_id)
            })

            # Write the updated data back to the JSON file
            with open(JSON_FILE, 'w') as f:
                json.dump(data, f, indent=4)

    def check_solved(self):
        JSON_FILE = 'problem_description.json'
        LOCK_FILE = JSON_FILE + '.lock'
        lock = FileLock(LOCK_FILE)
        data = None
        try:
            with lock:
                with open(JSON_FILE, 'r') as f:
                    data = json.load(f)
        except:
            raise ValueError(f"You have not run any problem yet, Nothing to continue")

        problem_id = str(self.problem_id)

        # Ensure `problem_id` and `as_results` are initialized
        if problem_id not in data:
            raise ValueError(f"Problem ID {problem_id} not found in {JSON_FILE}")
        
        if 'min(AS)_valid' in data[problem_id]:
            return 0
        elif 'min(AS)_leak' in data[problem_id]:
            return 1
        elif 'optimal_valid' in data[problem_id]:
            return 2
        elif 'optimal_leak' in data[problem_id]:
            return 3
        else:
            return -1




def main():

    try:
        # Read the input file path from command-line arguments
        input_file = sys.argv[1]

        # Load the initial_solver_input from the JSON file
        with open(input_file, 'r') as f:
            initial_solver_input = json.load(f)

        # Convert lists back to NumPy arrays if necessary
        for key in ['original_xdata', 'original_upperbound_lin', 'original_lowerbound_lin',
                    'cutoffs_x', 'cutoffs_upper_ydata_lin', 'cutoffs_lower_ydata_lin']:
            if key in initial_solver_input:
                initial_solver_input[key] = np.array(initial_solver_input[key])

        # Create an instance of BackendRunner and run solvers
        solver_runner = BackendRunner(initial_solver_input)
        solver_runner.run_solvers()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        if str(e) == "problem is unsat":
            print(f"@MSG@ problem is unsat")
                       
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()