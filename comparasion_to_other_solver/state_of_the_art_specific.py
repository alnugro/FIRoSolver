import numpy as np

import sys
import os
import json
import traceback
import time


try:
    from backend.backend_main import SolverBackend

except:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from backend.backend_main import SolverBackend



class TestBench():
    # presolve_done = pyqtSignal(tuple)

    def __init__(self ,initial_solver_input, test_key):
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

        # Dynamically assign values from input_data, skipping any keys that don't have matching attributes
        for key, value in initial_solver_input.items():
            if hasattr(self, key):  # Only set attributes that exist in the class
                setattr(self, key, value)
            
        self.key = test_key
        self.initial_solver_input = initial_solver_input
        self.result_valid = {}
        self.result_leaked = {}
        self.bound_key = None
        self.duration = 0
        self.start_time = time.time()
        self.init_txt()
        self.txt_iter
    
    def run_solver(self):
        try:
            #initiate backend
            backend = SolverBackend(self.initial_solver_input)
            
            print(f"Running presolve")
           
           #start presolve
            presolve_result = backend.solver_presolve()

            print(f"Presolve done")
            

            result1_valid = False
            result2_valid = False
            
            print(f"Finding best A_S(h_zero_max)")
            target_result, best_adderm ,total_adder, adder_s_h_zero_best = backend.find_best_adder_s(presolve_result)
                # target_result2, best_adderm2, total_adder2, adderm_h_zero_best = backend.find_best_adder_m(presolve_result)

            target_result.update({
               'option' : 'A_S(h_zero_max)'
           })

            
            while True:

                leaks, leaks_mag = backend.result_validator(target_result['h_res'],target_result['gain'])
                if leaks:
                    self.print_txt_file('sat',target_result, best_adderm ,total_adder, True)
                    print(f"While solving for A_S(h_zero_max): leak found, patching leak...")
                    backend.patch_leak(leaks, leaks_mag)

                    target_result_temp, satisfiability = backend.solving_result_barebone(presolve_result,best_adderm,adder_s_h_zero_best)
                    if satisfiability == 'unsat':
                        print(f"While solving for A_S(h_zero_max): problem is unsat from asserting the leak to the problem")
                        self.print_txt_file('unsat',target_result_temp, None ,None, True)

                        backend.revert_patch()
                        break
                    else:

                        target_result = target_result_temp
                        target_result.update({
                            'option' : 'A_S(h_zero_max)'
                        })
                        self.print_txt_file('sat',target_result, best_adderm ,total_adder, True)

                else:
                    result1_valid = True
                    self.print_txt_file('sat',target_result, best_adderm ,total_adder, False)
                    break
            
            print(f"A_S(h_zero_max) found")
            
            if self.deepsearch and self.gurobi_thread > 0:
                print(f"Finding best A_S(A_M_Max(h_zero_max))")
                target_result2, best_adderm2 ,total_adder2, adder_m_h_zero_best2 = backend.find_best_adder_m(presolve_result)
                target_result2.update({
                    'option' : 'A_S(A_M_Max(h_zero_max))'
                })
                while True:
                    leaks, leaks_mag = backend.result_validator(target_result2['h_res'],target_result2['gain'])
                    
                    if leaks:
                        self.print_txt_file('sat',target_result2, best_adderm2 ,total_adder2, True)

                        print(f"While solving for A_S(A_M_Max(h_zero_max)): leak found, patching leak...")
                        backend.patch_leak(leaks, leaks_mag)

                        target_result_temp2, satisfiability = backend.solving_result_barebone(presolve_result,best_adderm2,adder_m_h_zero_best2)
                        if satisfiability == 'unsat':
                            print(f"While solving A_S(A_M_Max(h_zero_max)): problem is unsat from asserting the leak to the problem")
                            backend.revert_patch()
                            self.print_txt_file('unsat',target_result_temp2, None ,None, True)
                            break
                        else:
                            target_result2 = target_result_temp2
                            target_result2.update({
                                'option' : 'A_S(A_M_Max(h_zero_max))'
                            })
                    else:
                        result2_valid = True
                        self.print_txt_file('sat',target_result2, best_adderm2 ,total_adder2, False)
                        break
                
                print(f"A_S(A_M_Max(h_zero_max)) found")

                print(f"Starting deep search")

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
                    print(f"Deep Search canceled, no search space for h_zero: Taking either A_S(h_zero_max) or A_S(A_M_Min(h_zero_max)) as the best solution")
                    best_adderm3 = best_adderm if total_adder >= total_adder2 else best_adderm2
                    total_adder3 = total_adder if total_adder >= total_adder2 else total_adder2
                    if total_adder >= total_adder2 and result1_valid:
                        target_result3 = target_result  

                    elif total_adder2 >= total_adder and result2_valid:
                        target_result3 = target_result2
                    
                    else: 
                        target_result3 = {
                            'option' : 'optimal',
                            'h_res' : '',
                            'gain' : ''
                        }

                    target_result3.update({
                                    'option' : 'optimal'
                                })
                    
                    self.print_txt_file('no search space',target_result3, best_adderm3 ,total_adder3, False)

                else:
                    target_result3, best_adderm3, total_adder3, h_zero_best3 = backend.deep_search_adder_total(presolve_result, data_dict)
                    

                    if h_zero_best3 == None: #if unsat
                        print(f"Deep Search canceled, all search space unsat: Taking either A_S(h_zero_max) or A_S(A_M_Min(h_zero_max)) as the best solution")
                        best_adderm3 = best_adderm if total_adder >= total_adder2 else best_adderm2
                        total_adder3 = total_adder if total_adder >= total_adder2 else total_adder2
                        if total_adder >= total_adder2 and result1_valid:
                            target_result3 = target_result  

                        elif total_adder2 >= total_adder and result2_valid:
                            target_result3 = target_result2
                        
                        else: target_result3 = {}

                        target_result3.update({
                                        'option' : 'optimal'
                                    })
                        
                        self.print_txt_file('unsat',target_result3, best_adderm3 ,total_adder3, False)

                        return
                    else:
                        #if its sat then take it as optimal
                        target_result3.update({
                            'option' : 'optimal'
                        })
                        
                    while True:
                        leaks, leaks_mag = backend.result_validator(target_result3['h_res'],target_result3['gain'])
                        
                        if leaks:
                            self.print_txt_file('sat',target_result3, best_adderm3 ,total_adder3, True)

                            print(f"While solving the optimal result: leak found, patching leak...")
                            backend.patch_leak(leaks, leaks_mag)

                            target_result_temp3, satisfiability = backend.solving_result_barebone(presolve_result,best_adderm3,h_zero_best3)
                            if satisfiability == 'unsat':
                                print(f"While solving the optimal result: problem is unsat from asserting the leak to the problem")
                                backend.revert_patch()
                                self.print_txt_file('unsat',target_result_temp2, None ,None, True)
                                break
                            else:
                                target_result3 = target_result_temp3
                                target_result3.update({
                                    'option' : 'optimal'
                                })
                        else:
                            self.print_txt_file('sat',target_result3, best_adderm3 ,total_adder3, False)
                            break

            else:
                if result1_valid:
                    target_result.update({
                            'option' : 'optimal'
                        })
                    self.print_txt_file('sat',target_result, best_adderm ,total_adder, False)
                else:
                    target_result = {}
                    target_result.update({
                            'option' : 'optimal'
                        })
                    self.print_txt_file('sat',target_result, best_adderm ,total_adder, True)
            
        except ValueError as e:
                if str(e) == "problem is unsat":
                    print(f"Given problem is unsat")
                    self.print_txt_file('problem is unsat',0, 0 ,0, False)
        except Exception as e:
            print(f"{e}")
            self.print_txt_file(f"Exception: {e}",0, 0 ,0, False)
            traceback.print_exc()

    def init_txt(self):
        self.txt_iter = self.get_iterator()
        self.print_txt_file('init',0,0,0,False)

    def get_iterator(self):
        i = 0  # Start with i = 0
        
        while True:
            filename = f"state_of_the_art_basic{i}.txt"
            
            # Check if the file exists
            if not(os.path.exists(filename)):
                # if the file doesnt exists create one
                with open(filename, "w") as file:
                    # Write header
                    file.write("Key ;satisfiability;Leak flag; option; h_res;Gain;Total_adder ; Adder_m; Adder_s ;Duration;Filter Type;\n")
                    print(f"{filename} created.")

                    break
            else:                 
                print(f"{filename} already exists. Breaking the loop.")
                i += 1
            
            # If the file doesn't exist, create it
            
            
              # Increment i for the next potential file
        
        return i  # Return the value of i where the file was found to exist

       
    def print_txt_file(self, satisfiability,target_result, best_adderm ,total_adder, leak_flag):
        iter = self.txt_iter
        
        end_time = time.time()
        
        h_res = 0
        gain = 0
        adder_s = 0
        option = 0

        if satisfiability == 'sat':
            adder_s = total_adder - best_adderm
            h_res = target_result['h_res']
            gain = target_result['gain']
            option = target_result['option']

        
        duration = end_time - self.start_time
        
        with open(f"state_of_the_art_basic{iter}.txt", "a") as file:
                file.write(f"{self.key};{satisfiability};{leak_flag};{option}; {h_res};{gain};{total_adder};{best_adderm};{adder_s};{duration};{self.filter_type+1}\n")
            
        print(f"File Written to state_of_the_art_basic{iter}.txt")


    


    
            

if __name__ == '__main__':
    # Unpack the dictionary to corresponding variables
    test_run = {
        8: {#L3
            'filter_type': 1,
            'order_current': 45,
            'accuracy': 3,
            'wordlength': 8,
            'gain_upperbound': 2.63,
            'gain_lowerbound': 1,
            'coef_accuracy': 5,
            'intW': 2,
            'adder_count': None,
            'adder_depth': 0,
            'avail_dsp': 0,
            'adder_wordlength_ext': 4,
            'gain_wordlength': 6,
            'gain_intW': 2,
            'gurobi_thread': 16,
            'pysat_thread': 0,
            'z3_thread': 0,
            'timeout': 0,
        }
    }



    test_key = 8
    print(test_run[test_key])
    # Accessing the dictionary for the test_key 1 and assigning variables
    filter_type = test_run[test_key]['filter_type']
    order_current = test_run[test_key]['order_current']
    accuracy = test_run[test_key]['accuracy']
    wordlength = test_run[test_key]['wordlength']
    gain_upperbound = test_run[test_key]['gain_upperbound']
    gain_lowerbound = test_run[test_key]['gain_lowerbound']
    coef_accuracy = test_run[test_key]['coef_accuracy']
    intW = test_run[test_key]['intW']
    adder_count = test_run[test_key]['adder_count']
    adder_depth = test_run[test_key]['adder_depth']
    avail_dsp = test_run[test_key]['avail_dsp']
    adder_wordlength_ext = test_run[test_key]['adder_wordlength_ext']
    gain_wordlength = test_run[test_key]['gain_wordlength']
    gain_intW = test_run[test_key]['gain_intW']
    gurobi_thread = test_run[test_key]['gurobi_thread']
    pysat_thread = test_run[test_key]['pysat_thread']
    z3_thread = test_run[test_key]['z3_thread']
    timeout = test_run[test_key]['timeout']
    



    space = order_current * accuracy * 20 #original accuracy
    # Initialize freq_upper and freq_lower with NaN values
    freqx_axis = np.linspace(0, 1, space)
    freq_upper = np.full(space, np.nan)
    freq_lower = np.full(space, np.nan)


    # Manually set specific values for the elements of freq_upper and freq_lower in dB
    point1 = int(0.15*(space))
    point2 = int(0.1875*(space))
    point3 = int(0.2125*(space))
    point4 = int(0.2875*(space))
    end_point = space

    freq_upper[0:point1] = 1 + 0.0165
    freq_lower[0:point1] = 1 - 0.0165

    freq_upper[point1:point2] = 1 + 0.0296
    freq_lower[point1:point2] = 1 - 0.0296

    freq_upper[point2:point3] = 1 + 0.0546
    freq_lower[point2:point3] = 1 - 0.0546

    freq_upper[point4:end_point] = 0 + 0.0316
    freq_lower[point4:end_point] = 0 #will be ignored


    cutoffs_x = []
    cutoffs_upper_ydata = []
    cutoffs_lower_ydata = []

    cutoffs_x.append(0)
    cutoffs_x.append(0.15)
    cutoffs_x.append(0.1875)
    cutoffs_x.append(0.2125)
    cutoffs_x.append(0.2875)
    cutoffs_x.append(1)

    cutoffs_upper_ydata.append(1 + 0.0165)
    cutoffs_upper_ydata.append(1 + 0.0296)
    cutoffs_upper_ydata.append(1 + 0.0296)
    cutoffs_upper_ydata.append(1 + 0.0546)
    cutoffs_upper_ydata.append(0 + 0.0316)
    cutoffs_upper_ydata.append(0 + 0.0316)
    
    cutoffs_lower_ydata.append(1 - 0.0165)
    cutoffs_lower_ydata.append(1 - 0.0296)
    cutoffs_lower_ydata.append(1 - 0.0296)
    cutoffs_lower_ydata.append(1 - 0.0546)
    cutoffs_lower_ydata.append(0)
    cutoffs_lower_ydata.append(0)

    # print(cutoffs_x)
    # print(len(freqx_axis))


    #beyond this bound lowerbound will be ignored
    ignore_lowerbound = -100

    #linearize the bound
    upperbound_lin = np.copy(freq_upper)
    lowerbound_lin = np.copy(freq_lower)
    ignore_lowerbound_lin = 10 ** (ignore_lowerbound / 20)

    cutoffs_upper_ydata_lin = np.copy(cutoffs_upper_ydata)
    cutoffs_lower_ydata_lin = np.copy(cutoffs_lower_ydata)

    # print(np.array(upperbound_lin).tolist())
    # print(np.array(lowerbound_lin).tolist())
    # print(ignore_lowerbound)
    
    


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
        'wordlength': wordlength,
        'adder_count': adder_count,
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
        'solver_accuracy_multiplier': accuracy,
        'deepsearch': False,
        'patch_multiplier' : 1,
        'gurobi_auto_thread': False
    }
    

    # Instantiate the BackendMediator
    tester = TestBench(input_data, test_key)
    tester.run_solver()
    
    
        
    print("Test ", test_key, " is completed")

    print("Benchmark completed and results saved to state_of_the_art_basic.txt")


