# solver_runner.py
import sys
import json
import os
import traceback
import numpy as np
import json
import numpy as np
import os
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

        self.deepsearch = None
        self.problem_id = None


        # Dynamically assign values from input_data, skipping any keys that don't have matching attributes
        for key, value in solver_input.items():
            if hasattr(self, key):  # Only set attributes that exist in the class
                setattr(self, key, value)


        self.solver_input = solver_input
        # self.file_lock = file_lock  # Ensure this is pickleable

        self.solver_name = None
        # self.problem_id = problem_id

        if self.gurobi_thread > 0:
            self.solver_name = 'Gurobi'
        if self.pysat_thread > 0:
            self.solver_name = 'pysat'
        if self.z3_thread > 0:
            self.solver_name = 'z3'

        

    def run_solvers(self):
        print(f"@MSG@ : Running solver")
        # time.sleep(40)
        #initiate backend
        backend = SolverBackend(self.solver_input)
        if self.gurobi_thread > 1:
            print(f"@MSG@ : Gurobi is chosen, running compatibility test")            
            backend.gurobi_test()
            print(f"@MSG@ : Gurobi compatibility test done")

        if self.start_with_error_prediction:
            backend.error_prediction()

        print(f"@MSG@ : Running presolve")
        
        # print(f"self.solver_name {self.solver_name}")
        # print(f"self.gurobi_thread {self.gurobi_thread}")
        # print(f"self.pysat_thread {self.pysat_thread}")
        # print(f"self.z3_thread {self.z3_thread}")
         
        # start presolve
        presolve_result = backend.solver_presolve()
        print(presolve_result)
        

        print(f"@MSG@ : Presolve done")
        
        result1_valid = False
        result2_valid = False
        
        print(f"@MSG@ : Finding best A_S(h_zero_max)")
        target_result, best_adderm ,total_adder, adder_s_h_zero_best = backend.find_best_adder_s(presolve_result)
            # target_result2, best_adderm2, total_adder2, adderm_h_zero_best = backend.find_best_adder_m(presolve_result)
        
        target_result.update({
           'option' : 'A_S(h_zero_max)'
        })

        
        while True:
            leaks, leaks_mag = backend.result_validator(target_result['h_res'],target_result['gain'])
            if leaks:

                self.update_json_with_lock(target_result , 'result_leak.json')

                print(f"@MSG@ : While solving for A_S(h_zero_max): leak found, patching leak...")
                backend.patch_leak(leaks, leaks_mag)

                target_result_temp, satisfiability = backend.solving_result_barebone(presolve_result,best_adderm,adder_s_h_zero_best)
                if satisfiability == 'unsat':
                    print(f"@MSG@ : While solving for A_S(h_zero_max): problem is unsat from asserting the leak to the problem")
                    backend.revert_patch()
                    break
                else:
                    target_result = target_result_temp
                    target_result.update({
                        'option' : 'A_S(h_zero_max)'
                    })
            else:
                print(f"@MSG@ :A_S(h_zero_max): Result Validated, no leaks found")
                result1_valid = True
                self.update_json_with_lock( target_result , 'result_valid.json')
                break
        
        print(f"@MSG@ : A_S(h_zero_max) found")
        
        if self.deepsearch and self.gurobi_thread > 0:
            print(f"@MSG@ : Finding best A_S(A_M_Max(h_zero_max))")
            target_result2, best_adderm2 ,total_adder2, adder_m_h_zero_best2 = backend.find_best_adder_m(presolve_result)
            target_result2.update({
                'option' : 'A_S(A_M_Max(h_zero_max))'
            })
            while True:
                leaks, leaks_mag = backend.result_validator(target_result2['h_res'],target_result2['gain'])
                
                if leaks:
                    self.update_json_with_lock(target_result2 , 'result_leak.json')

                    print(f"@MSG@ : While solving for A_S(A_M_Max(h_zero_max)): leak found, patching leak...")
                    backend.patch_leak(leaks, leaks_mag)

                    target_result_temp2, satisfiability = backend.solving_result_barebone(presolve_result,best_adderm2,adder_m_h_zero_best2)
                    if satisfiability == 'unsat':
                        print(f"@MSG@ : While solving A_S(A_M_Max(h_zero_max)): problem is unsat from asserting the leak to the problem")
                        backend.revert_patch()
                        break
                    else:
                        target_result2 = target_result_temp2
                        target_result2.update({
                            'option' : 'A_S(A_M_Max(h_zero_max))'
                        })
                else:
                    print(f"@MSG@ :A_S(A_M_Max(h_zero_max)): Result Validated, no leaks found")
                    result2_valid = True
                    self.update_json_with_lock( target_result2 , 'result_valid.json')
                    break
            
            print(f"@MSG@ : A_S(A_M_Max(h_zero_max)) found")

            print(f"@MSG@ : Starting deep search")

            # Packing variables into a dictionary 
            data_dict = {
            'best_adderm_from_s': int(best_adderm),
            'best_adderm_from_m': int(best_adderm2),
            'total_adder_s': int(total_adder),
            'total_adder_m': int(total_adder2),
            'adder_s_h_zero_best': int(adder_s_h_zero_best),
            'adder_m_h_zero_best': int(adder_m_h_zero_best2),
            }

            print(data_dict)

            if adder_m_h_zero_best2 + 1 >= presolve_result['max_zero']:
                print(f"@MSG@ : Deep Search canceled, no search space for h_zero: Taking either A_S(h_zero_max) or A_S(A_M_Min(h_zero_max)) as the best solution")
                best_adderm3 = best_adderm if total_adder >= total_adder2 else best_adderm2
                total_adder3 = total_adder if total_adder >= total_adder2 else total_adder2
                if total_adder <= total_adder2 and result1_valid:
                    target_result3 = target_result  

                elif total_adder2 <= total_adder and result2_valid:
                    target_result3 = target_result2
                
                else: target_result3 = {}

                target_result3.update({
                                'option' : 'optimal'
                            })
                
                self.update_json_with_lock( target_result3 , 'result_valid.json')
            else:
                target_result3, best_adderm3, total_adder3, h_zero_best3 = backend.deep_search_adder_total(presolve_result, data_dict)
                

                if h_zero_best3 == None: #if unsat
                    print(f"@MSG@ : Deep Search canceled, all search space unsat: Taking either A_S(h_zero_max) or A_S(A_M_Min(h_zero_max)) as the best solution")
                    best_adderm3 = best_adderm if total_adder >= total_adder2 else best_adderm2
                    total_adder3 = total_adder if total_adder >= total_adder2 else total_adder2
                    if total_adder <= total_adder2 and result1_valid:
                        target_result3 = target_result  

                    elif total_adder2 <= total_adder and result2_valid:
                        target_result3 = target_result2
                    
                    else: target_result3 = {}

                    target_result3.update({
                                    'option' : 'optimal'
                                })
                    
                    self.update_json_with_lock( target_result3 , 'result_valid.json')

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

            
        # if result1_valid and not self.deepsearch:
        #     target_result.update({
        #             'option' : 'optimal'
        #          })
        #     self.update_json_with_lock( target_result , 'result_valid.json')
        # else:
        #     target_result = {}
        #     target_result.update({
        #             'option' : 'optimal'
        #          })
        #     self.update_json_with_lock( target_result , 'result_valid.json')
        
        

    def update_json_with_lock(self, new_data, filename):
        
        new_data.update({
                'problem_id': self.problem_id,
                'solver' : self.solver_name
        })
        lock_filename = filename + '.lock'
        lock = FileLock(lock_filename)


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

            # Save the updated dictionary back to the file
            with open(filename, 'w') as json_file:
                json.dump(current_data, json_file, indent=4)




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