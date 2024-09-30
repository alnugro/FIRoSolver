import numpy as np
from pebble import ProcessPool, ProcessExpired
from concurrent.futures import TimeoutError, CancelledError, wait, ALL_COMPLETED
import traceback
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QComboBox, QSpinBox, QTextEdit, QTableWidget, QTableWidgetItem, QWidget, QFrame
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from concurrent.futures import TimeoutError  # Correct import for TimeoutError
import multiprocessing
from multiprocessing import Manager

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

        manager = Manager()
        self.file_lock = manager.Lock()


        self.queue = multiprocessing.Queue()
        self.timer = QTimer()  # Timer to check the queue periodically
        self.timer.timeout.connect(self.check_queue)
        self.pools = []

    def run(self):
        self.execute_parallel_solve()


    
    def run_solvers(self, solver_input, queue):
         # Explicit declaration of instance variables with default values (if applicable)
        filter_type = None
        order_upperbound = None
        ignore_lowerbound = None
        wordlength = None
        adder_depth = None
        avail_dsp = None
        adder_wordlength_ext = None
        gain_upperbound = None
        gain_lowerbound = None
        coef_accuracy = None
        intW = None
        gain_wordlength = None
        gain_intW = None
        gurobi_thread = None
        pysat_thread = None
        z3_thread = None
        timeout = None
        start_with_error_prediction = None
        original_xdata = None
        original_upperbound_lin = None
        original_lowerbound_lin = None
        cutoffs_x = None
        cutoffs_upper_ydata_lin = None
        cutoffs_lower_ydata_lin = None
        solver_accuracy_multiplier = None
        deepsearch = None
        bound_key = None


       # Dynamically assign values from solver_input to local variables (in the dictionary)
        for key, value in solver_input.items():
            if key in solver_input:  # Check if the key exists in local_vars
                solver_input[key] = value



        #initiate backend
        backend = SolverBackend(solver_input)
        queue.put(f"Gurobi is chosen, running compatibility test")            
        backend.gurobi_test()
        queue.put(f"Gurobi compatibility test done")
        if start_with_error_prediction:
            backend.error_prediction()

        queue.put(f"Running presolve")
        
        # start presolve
        presolve_result = backend.solver_presolve()

        queue.put(f"Presolve done")
        
        result1_valid = False
        result2_valid = False
        
        queue.put(f"Finding best A_S(h_zero_max)")
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

                self.update_json_with_lock(target_result , 'result_leak.json', bound_key)

                queue.put(f"While solving for A_S(h_zero_max): leak found, patching leak...")
                backend.patch_leak(leaks, leaks_mag)

                target_result_temp, satisfiability = backend.solving_result_barebone(presolve_result,best_adderm,adder_s_h_zero_best)
                if satisfiability == 'unsat':
                    queue.put(f"While solving for A_S(h_zero_max): problem is unsat from asserting the leak to the problem")
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
                self.update_json_with_lock( target_result , 'result_valid.json', bound_key)
                print(f"i ran here3")
                break
        
        queue.put(f"A_S(h_zero_max) found")
        
        if deepsearch and gurobi_thread > 0:
            queue.put(f"Finding best A_S(A_M_Max(h_zero_max))")
            target_result2, best_adderm2 ,total_adder2, adder_m_h_zero_best2 = backend.find_best_adder_m(presolve_result)
            target_result2.update({
                'option' : 'A_S(A_M_Max(h_zero_max))'
            })
            while True:
                leaks, leaks_mag = backend.result_validator(target_result2['h_res'],target_result2['gain'])
                
                if leaks:
                    self.update_json_with_lock(target_result2 , 'result_leak.json', bound_key)

                    queue.put(f"While solving for A_S(A_M_Max(h_zero_max)): leak found, patching leak...")
                    backend.patch_leak(leaks, leaks_mag)

                    target_result_temp2, satisfiability = backend.solving_result_barebone(presolve_result,best_adderm2,adder_m_h_zero_best2)
                    if satisfiability == 'unsat':
                        queue.put(f"While solving A_S(A_M_Max(h_zero_max)): problem is unsat from asserting the leak to the problem")
                        backend.revert_patch()
                        break
                    else:
                        target_result2 = target_result_temp2
                        target_result2.update({
                            'option' : 'A_S(A_M_Max(h_zero_max))'
                        })
                else:
                    result2_valid = True
                    self.update_json_with_lock( target_result2 , 'result_valid.json', bound_key)
                    break
            
            queue.put(f"A_S(A_M_Max(h_zero_max)) found")

            queue.put(f"Starting deep search")

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
                queue.put(f"Deep Search canceled, no search space for h_zero: Taking either A_S(h_zero_max) or A_S(A_M_Min(h_zero_max)) as the best solution")
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
                
                self.update_json_with_lock( target_result3 , 'result_valid.json', bound_key)
            else:
                target_result3, best_adderm3, total_adder3, h_zero_best3 = backend.deep_search_adder_total(presolve_result, data_dict)
                

                if h_zero_best3 == None: #if unsat
                    queue.put(f"Deep Search canceled, all search space unsat: Taking either A_S(h_zero_max) or A_S(A_M_Min(h_zero_max)) as the best solution")
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
                    
                    self.update_json_with_lock( target_result3 , 'result_valid.json', bound_key)

                    return
                else:
                    #if its sat then take it as optimal
                    target_result3.update({
                        'option' : 'optimal'
                    })
                    
                while True:
                    leaks, leaks_mag = backend.result_validator(target_result3['h_res'],target_result3['gain'])
                    
                    if leaks:
                        self.update_json_with_lock(target_result3 , 'result_leak.json', bound_key)

                        queue.put(f"While solving the optimal result: leak found, patching leak...")
                        backend.patch_leak(leaks, leaks_mag)

                        target_result_temp3, satisfiability = backend.solving_result_barebone(presolve_result,best_adderm3,h_zero_best3)
                        if satisfiability == 'unsat':
                            queue.put(f"While solving the optimal result: problem is unsat from asserting the leak to the problem")
                            backend.revert_patch()
                            break
                        else:
                            target_result3 = target_result_temp3
                            target_result3.update({
                                'option' : 'optimal'
                            })
                    else:
                        self.update_json_with_lock( target_result3 , 'result_valid.json', bound_key)
                        break

            
        if result1_valid:
            target_result.update({
                    'option' : 'optimal'
                 })
            self.update_json_with_lock( target_result , 'result_valid.json', bound_key)
        else:
            target_result = {}
            target_result.update({
                    'option' : 'optimal'
                 })
            self.update_json_with_lock( target_result , 'result_valid.json', bound_key)
        
        
        self.load_and_print_with_lock(True)


        
            


    def generate_bound_description(self):
        bound_description = {
                'original_xdata' : np.array(self.original_xdata).tolist(),
                'original_upperbound_lin': np.array(self.original_upperbound_lin).tolist(),
                'original_lowerbound_lin': np.array(self.original_lowerbound_lin).tolist(),
            }

        self.update_json_with_lock(bound_description , 'bound_description.json',0 , True)
         
    # Step 2.5: Function to update the dictionary with new data
    def update_json_with_lock(self, new_data, filename, bound_key = 0, collect_key = False ):
        
        if collect_key == False:
            new_data.update({
                 'bound_key': bound_key
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


    def execute_parallel_solve(self):#with queue
        futures_gurobi = []
        futures_z3 = []
        futures_pysat = []
        
        #generate bound description for result handler
        self.generate_bound_description()

        gurobi_solver_input = self.initial_solver_input
        pysat_solver_input = self.initial_solver_input
        z3_solver_input = self.initial_solver_input

        try:
            # Conditionally create the Gurobi pool
            if self.gurobi_thread > 0:
                gurobi_solver_input.update({
                    'bound_key': self.bound_key,
                    'pysat_thread': 0,
                    'z3_thread': 0,
                })
                pool_gurobi = ProcessPool(max_workers=1)
                self.pools.append(pool_gurobi)
                future_single_gurobi = pool_gurobi.schedule(
                    self.run_solvers, args=(gurobi_solver_input, self.queue, ), timeout=self.timeout
                )
                futures_gurobi.append(future_single_gurobi)
            else:
                pool_gurobi = None


            # Conditionally create the Z3 pool
            if self.z3_thread > 0:
                z3_solver_input.update({
                    'bound_key': self.bound_key,
                    'gurobi_thread': 0,
                    'pysat_thread': 0,
                })
                pool_z3 = ProcessPool(max_workers=self.z3_thread)
                self.pools.append(pool_z3)
                for i in range(self.z3_thread):
                    future_single_z3 = pool_z3.schedule(
                        self.run_solvers, args=(z3_solver_input, self.queue, ), timeout=self.timeout
                    )
                    futures_z3.append(future_single_z3)

            else:
                pool_z3 = None

            # Conditionally create the PySAT pool
            if self.pysat_thread > 0:
                pysat_solver_input.update({
                    'bound_key': self.bound_key,
                    'gurobi_thread': 0,
                    'z3_thread': 0,
                })
                pool_pysat = ProcessPool(max_workers=self.pysat_thread)
                self.pools.append(pool_pysat)
                for i in range(self.pysat_thread):
                    future_single_pysat = pool_pysat.schedule(
                        self.run_solvers, args=(pysat_solver_input, self.queue, ), timeout=self.timeout
                    )
                    futures_pysat.append(future_single_pysat)
            else:
                pool_pysat = None

            

            if self.gurobi_thread > 0:
                future_single_gurobi.add_done_callback(self.task_done('gurobi'))
            if self.z3_thread > 0:
                for future in futures_z3:
                    future.add_done_callback(self.task_done('z3', futures_z3))
            if self.pysat_thread > 0:
                for future in futures_pysat:
                    future.add_done_callback(self.task_done('pysat', futures_pysat))

            # Start the QTimer to check the queue every 100ms
            self.timer.start(100)
            
            # Wait for all futures to complete, handling timeouts as well
            all_futures = futures_gurobi + futures_z3 + futures_pysat
            done, not_done = wait(all_futures, return_when=ALL_COMPLETED)

            


        finally:
            # Ensure all pools are properly cleaned up
            for pool in self.pools:
                pool.stop()
                pool.join()
        
        return

    def check_queue(self):
        """
        Periodically checks the multiprocessing Queue for messages.
        Emits PyQt signals based on the messages.
        """
        while not self.queue.empty():
            message = self.queue.get()
            if "Error" in message:
                self.exception_message.emit(message)
            else:
                self.log_message.emit(message)

        
    def task_done(self, solver_name):
        def callback(future):
            try:
                future.result()  # blocks until results are ready
                print(f"{solver_name} task done")


            except ValueError as e:
                if str(e) == "problem is unsat":
                    self.log_message.emit(f"problem is unsat for solver {solver_name}")

            except CancelledError:
                self.log_message.emit(f"{solver_name} task was cancelled.")
            except TimeoutError:
                self.log_message.emit(f"{solver_name} task Timed Out.")
            except ProcessExpired as error:
                self.log_message.emit(f"{solver_name} process {error.pid} expired.")
            except Exception as error:
                print(f"{solver_name} task raised an exception: {error}")
                self.exception_message.emit(f"{solver_name} task raised an exception: {error}")
                traceback.print_exc()

        return callback


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


