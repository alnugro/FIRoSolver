import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QComboBox, QSpinBox, QTextEdit, QTableWidget, QTableWidgetItem, QWidget, QFrame
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from concurrent.futures import TimeoutError  # Correct import for TimeoutError
import multiprocessing
import sys
import os
import json
import traceback


try:
    from .live_logger import LiveLogger
    from backend.backend_main import SolverBackend

except:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from live_logger import LiveLogger
    from backend.backend_main import SolverBackend



class BackendMediator(QThread):
    log_message = pyqtSignal(str)
    exception_message = pyqtSignal(str)
    # presolve_done = pyqtSignal(tuple)

    def __init__(self ,initial_solver_input):
        super().__init__()

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
            

        self.initial_solver_input = initial_solver_input

        self.file_lock = multiprocessing.Lock()
        self.result_valid = {}
        self.result_leaked = {}
        self.bound_key = None

       



    


    
    def run(self):
        try:
            #initiate backend
            backend = SolverBackend(self.initial_solver_input)
            self.log_message.emit(f"Gurobi is chosen, running compatibility test")            
            backend.gurobi_test()
            self.log_message.emit(f"Gurobi compatibility test done")
            if self.start_with_error_prediction:
                backend.error_prediction()

            self.log_message.emit(f"Running presolve")
           
           #start presolve
            presolve_result = backend.solver_presolve()

            self.log_message.emit(f"Presolve done")
            
            #generate bound description for result handler
            self.generate_bound_description()

            result1_valid = False
            result2_valid = False
            
            self.log_message.emit(f"Finding best A_S(h_zero_max)")
            target_result, best_adderm ,total_adder, adder_s_h_zero_best = backend.find_best_adder_s(presolve_result)
                # target_result2, best_adderm2, total_adder2, adderm_h_zero_best = backend.find_best_adder_m(presolve_result)

            target_result.update({
               'option' : 'A_S(h_zero_max)'
           })

            
            while True:
                print(f"i ran here after true")

                leaks, leaks_mag = backend.result_validator(target_result['h_res'],target_result['gain'])
                
                if leaks:
                    print(f"i ran here leaks")

                    self.update_json_with_lock(target_result , 'result_leak.json')

                    self.log_message.emit(f"While solving for A_S(h_zero_max): leak found, patching leak...")
                    backend.patch_leak(leaks, leaks_mag)

                    target_result_temp, satisfiability = backend.solving_result_barebone(presolve_result,best_adderm,adder_s_h_zero_best)
                    if satisfiability == 'unsat':
                        self.log_message.emit(f"While solving for A_S(h_zero_max): problem is unsat from asserting the leak to the problem")
                        backend.revert_patch()
                        break
                    else:
                        print(f"i ran here else")

                        target_result = target_result_temp
                        target_result.update({
                            'option' : 'A_S(h_zero_max)'
                        })
                else:
                    result1_valid = True
                    print(f"i ran here2")
                    self.update_json_with_lock( target_result , 'result_valid.json')
                    print(f"i ran here3")
                    break
            
            self.log_message.emit(f"A_S(h_zero_max) found")
            
            if self.deepsearch and self.gurobi_thread > 0:
                self.log_message.emit(f"Finding best A_S(A_M_Max(h_zero_max))")
                target_result2, best_adderm2 ,total_adder2, adder_m_h_zero_best2 = backend.find_best_adder_m(presolve_result)
                target_result2.update({
                    'option' : 'A_S(A_M_Max(h_zero_max))'
                })
                while True:
                    leaks, leaks_mag = backend.result_validator(target_result2['h_res'],target_result2['gain'])
                    
                    if leaks:
                        self.update_json_with_lock(target_result2 , 'result_leak.json')

                        self.log_message.emit(f"While solving for A_S(A_M_Max(h_zero_max)): leak found, patching leak...")
                        backend.patch_leak(leaks, leaks_mag)

                        target_result_temp2, satisfiability = backend.solving_result_barebone(presolve_result,best_adderm2,adder_m_h_zero_best2)
                        if satisfiability == 'unsat':
                            self.log_message.emit(f"While solving A_S(A_M_Max(h_zero_max)): problem is unsat from asserting the leak to the problem")
                            backend.revert_patch()
                            break
                        else:
                            target_result2 = target_result_temp2
                            target_result2.update({
                                'option' : 'A_S(A_M_Max(h_zero_max))'
                            })
                    else:
                        result2_valid = True
                        self.update_json_with_lock( target_result2 , 'result_valid.json')
                        break
                
                self.log_message.emit(f"A_S(A_M_Max(h_zero_max)) found")

                self.log_message.emit(f"Starting deep search")

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
                    self.log_message.emit(f"Deep Search canceled, no search space for h_zero: Taking either A_S(h_zero_max) or A_S(A_M_Min(h_zero_max)) as the best solution")
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
                    
                    self.update_json_with_lock( target_result3 , 'result_valid.json')
                else:
                    target_result3, best_adderm3, total_adder3, h_zero_best3 = backend.deep_search_adder_total(presolve_result, data_dict)
                    

                    if h_zero_best3 == None: #if unsat
                        self.log_message.emit(f"Deep Search canceled, all search space unsat: Taking either A_S(h_zero_max) or A_S(A_M_Min(h_zero_max)) as the best solution")
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

                            self.log_message.emit(f"While solving the optimal result: leak found, patching leak...")
                            backend.patch_leak(leaks, leaks_mag)

                            target_result_temp3, satisfiability = backend.solving_result_barebone(presolve_result,best_adderm3,h_zero_best3)
                            if satisfiability == 'unsat':
                                self.log_message.emit(f"While solving the optimal result: problem is unsat from asserting the leak to the problem")
                                backend.revert_patch()
                                break
                            else:
                                target_result3 = target_result_temp3
                                target_result3.update({
                                    'option' : 'optimal'
                                })
                        else:
                            self.update_json_with_lock( target_result3 , 'result_valid.json')
                            break

                
            if result1_valid:
                target_result.update({
                        'option' : 'optimal'
                     })
                self.update_json_with_lock( target_result , 'result_valid.json')
            else:
                target_result = {}
                target_result.update({
                        'option' : 'optimal'
                     })
                self.update_json_with_lock( target_result , 'result_valid.json')
            
            
            self.load_and_print_with_lock(True)


        except ValueError as e:
                if str(e) == "problem is unsat":
                     self.log_message.emit(f"Given problem is unsat")
        except Exception as e:
            print(f"{e}")
            self.exception_message.emit(f"{e}")
            traceback.print_exc()


    def generate_bound_description(self):
        bound_description = {
                'original_xdata' : np.array(self.original_xdata).tolist(),
                'original_upperbound_lin': np.array(self.original_upperbound_lin).tolist(),
                'original_lowerbound_lin': np.array(self.original_lowerbound_lin).tolist(),
            }

        self.update_json_with_lock(bound_description , 'bound_description.json', True)
         
    # Step 2.5: Function to update the dictionary with new data
    def update_json_with_lock(self, new_data, filename, collect_key = False):
        
        if collect_key == False:
            new_data.update({
                 'bound_key': self.bound_key
            })

        with self.file_lock:  # Acquire lock before file access
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

            if collect_key:
                self.bound_key = key

            # Add new data under the determined key
            current_data[str(key)] = new_data  # Ensure the key is a string for JSON compatibility

            # Save the updated dictionary back to the file
            with open(filename, 'w') as json_file:
                json.dump(current_data, json_file, indent=4)


    


    # Step 3: Function to load and print data with locking
    def load_and_print_with_lock(self,valid_flag = True):
        if valid_flag:
            filename='result_valid.json'
        else:
            filename='result_leak.json'

        with self.file_lock:  # Acquire lock before file access
            with open(filename, 'r') as json_file:
                loaded_data = json.load(json_file)

            # print(loaded_data)
            # Unpacking key-value pairs in a loop
            for key, value in loaded_data.items():
                print(f"Key: {key}, Value: {value}")
            

if __name__ == '__main__':
    from PyQt6.QtWidgets import QApplication

      # Test inputs
    filter_type = 0
    order_current = 16
    accuracy = 4
    wordlength = 12
    gain_upperbound = 2
    gain_lowerbound = 1
    coef_accuracy = 4
    intW = 4

    adder_count = 4
    adder_depth = 0
    avail_dsp = 0
    adder_wordlength_ext = 2
    intW = 4

    gain_wordlength = 6
    gain_intW = 2

    gurobi_thread = 16
    pysat_thread = 0
    z3_thread = 0

    timeout = 0

    space = order_current * accuracy * 50
    # Initialize freq_upper and freq_lower with NaN values
    freqx_axis = np.linspace(0, 1, space) #according to Mr. Kumms paper
    freq_upper = np.full(space, np.nan)
    freq_lower = np.full(space, np.nan)

    # Manually set specific values for the elements of freq_upper and freq_lower in dB
    lower_half_point = int(0.3*(space))
    upper_half_point = int(0.6*(space))
    end_point = space

    freq_upper[0:lower_half_point] = 3
    freq_lower[0:lower_half_point] = 0

    freq_upper[upper_half_point:end_point] = -15
    freq_lower[upper_half_point:end_point] = -1000


    cutoffs_x = []
    cutoffs_upper_ydata = []
    cutoffs_lower_ydata = []

    cutoffs_x.append(freqx_axis[0])
    cutoffs_x.append(freqx_axis[lower_half_point-1])
    cutoffs_x.append(freqx_axis[upper_half_point])
    cutoffs_x.append(freqx_axis[end_point-1])

    cutoffs_upper_ydata.append(freq_upper[0])
    cutoffs_upper_ydata.append(freq_upper[lower_half_point-1])
    cutoffs_upper_ydata.append(freq_upper[upper_half_point])
    cutoffs_upper_ydata.append(freq_upper[end_point-1])

    cutoffs_lower_ydata.append(freq_lower[0])
    cutoffs_lower_ydata.append(freq_lower[lower_half_point-1])
    cutoffs_lower_ydata.append(freq_lower[upper_half_point])
    cutoffs_lower_ydata.append(freq_lower[end_point-1])


    #beyond this bound lowerbound will be ignored
    ignore_lowerbound = -40

    #linearize the bound
    upperbound_lin = [10 ** (f / 20) if not np.isnan(f) else np.nan for f in freq_upper]
    lowerbound_lin = [10 ** (f / 20) if not np.isnan(f) else np.nan for f in freq_lower]
    ignore_lowerbound_lin = 10 ** (ignore_lowerbound / 20)

    cutoffs_upper_ydata_lin = [10 ** (f / 20) if not np.isnan(f) else np.nan for f in cutoffs_upper_ydata]
    cutoffs_lower_ydata_lin = [10 ** (f / 20) if not np.isnan(f) else np.nan for f in cutoffs_lower_ydata]


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
        'deepsearch': True,
        'patch_multiplier' : 1,
        'gurobi_auto_thread': False
    }
    

    # Initialize the QApplication
    app = QApplication(sys.argv)

    # Instantiate the BackendMediator
    mediator = BackendMediator(input_data)

    # Define slots for signals
    def log_slot(message):
        print(f"Log: {message}")

    def exception_slot(message):
        print(f"Exception: {message}")

    # Connect signals to slots
    mediator.log_message.connect(log_slot)
    mediator.exception_message.connect(exception_slot)

    # Connect the thread's finished signal to the application's quit method
    mediator.finished.connect(app.quit)

    # Start the mediator thread
    mediator.start()

    # Start the event loop
    sys.exit(app.exec())


