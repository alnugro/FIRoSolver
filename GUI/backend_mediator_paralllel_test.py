import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QComboBox, QSpinBox, QTextEdit, QTableWidget, QTableWidgetItem, QWidget, QFrame
from PyQt6.QtCore import QThread, pyqtSignal, QTimer
import sys
import os
import time
import string
import json
import traceback



from multiprocessing import Process, Manager
from concurrent.futures import TimeoutError  # Correct import for TimeoutError
from pebble import ProcessPool, ProcessExpired
from concurrent.futures import TimeoutError, CancelledError, wait, ALL_COMPLETED


try:
    from .live_logger import LiveLogger
    from backend.backend_runner import BackendRunner
    from backend.backend_main import SolverBackend

except:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from live_logger import LiveLogger
    from backend.backend_runner import BackendRunner
    from backend.backend_main import SolverBackend


class BackendMediator():

    def __init__(self, initial_solver_input):

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
        self.queue = manager.Queue()
        
        self.pools = []
        self.all_futures = None
        self.problem_id = None

    def run(self):
        # self.execute_parallel_solve()
        solver_runner = BackendRunner(self.initial_solver_input, self.file_lock, self.problem_id)
        solver_runner.run_solvers(self.queue)

    def execute_parallel_solve(self):
        
        #generate the problem description
        self.generate_bound_description()

        gurobi_solver_input = self.initial_solver_input.copy()
        pysat_solver_input = self.initial_solver_input.copy()
        z3_solver_input = self.initial_solver_input.copy()
        problem_id_loc = self.problem_id
        
        print("execute_parallel_solve")
        futures_gurobi = []
        futures_z3 = []
        futures_pysat = []

        try:
            if self.gurobi_thread > 0:
                gurobi_solver_input.update({
                    'pysat_thread': 0,
                    'z3_thread': 0,
                })
                # Create an instance of SolverRunner
                solver_runner = BackendRunner(gurobi_solver_input, self.file_lock, problem_id_loc)

                pool_gurobi = ProcessPool(max_workers=1)
                self.pools.append(pool_gurobi)
                future_single_gurobi = pool_gurobi.schedule(
                    solver_runner.run_solvers, args=(self.queue,), timeout=self.timeout
                )
                futures_gurobi.append(future_single_gurobi)
            else:
                pool_gurobi = None

             # Conditionally create the Z3 pool
            if self.z3_thread > 0:
                z3_solver_input.update({
                    'gurobi_thread': 0,
                    'pysat_thread': 0,
                })
                # Create an instance of SolverRunner
                solver_runner = BackendRunner(z3_solver_input, self.file_lock, problem_id_loc)

                pool_z3 = ProcessPool(max_workers=1)  # Use the non-daemonic pool
                             
                self.pools.append(pool_z3)
                future_single_z3 = pool_z3.schedule(
                    solver_runner.run_solvers, args=(self.queue,), timeout=self.timeout
                )
                futures_z3.append(future_single_z3)

            else:
                pool_z3 = None

            # Conditionally create the PySAT pool
            if self.pysat_thread > 0:
                pysat_solver_input.update({
                    'gurobi_thread': 0,
                    'z3_thread': 0,
                })
                # Create an instance of SolverRunner
                solver_runner = BackendRunner(pysat_solver_input, self.file_lock, problem_id_loc)

                pool_pysat = ProcessPool(max_workers=1)
                self.pools.append(pool_pysat)
                future_single_pysat = pool_pysat.schedule(
                    solver_runner.run_solvers, args=(self.queue,), timeout=self.timeout
                )
                futures_pysat.append(future_single_pysat)
            else:
                pool_pysat = None


            # Wait for all futures to complete
            self.all_futures = futures_gurobi + futures_z3 + futures_pysat
            done, not_done = wait(self.all_futures, return_when=ALL_COMPLETED)

            # Iterate over completed futures and handle exceptions
            for future in done:
                try:
                    # Determine which solver the future corresponds to
                    if future in futures_gurobi:
                        solver_name = 'Gurobi'
                    elif future in futures_z3:
                        solver_name = 'Z3'
                    elif future in futures_pysat:
                        solver_name = 'PySAT'
                    else:
                        solver_name = 'Unknown Solver'

                    future.result()
                except ValueError as e:
                    if str(e) == "problem is unsat":
                        self.log_message.emit(f"problem is unsat for solver: {solver_name}")
                        print("problem unsat")

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

        finally:
            # Clean up pools
            for pool in self.pools:
                pool.stop()
                pool.join()

    def check_queue(self):
        while not self.queue.empty():
            message = self.queue.get()
            if "Error" in message:
                self.exception_message.emit(message)
            else:
                self.log_message.emit(message)
    
    def generate_bound_description(self):
        problem_dict = self.initial_solver_input
        backend = SolverBackend(self.initial_solver_input)
        problem_dict.update ({
                'xdata' :  np.array(backend.xdata).tolist(),
                'upperbound_lin':  np.array(backend.upperbound_lin).tolist(),
                'lowerbound_lin': np.array(backend.lowerbound_lin).tolist(),
                'original_xdata': np.array(self.original_xdata).tolist(), #convert them to list, ndarray is not supported with json
                'original_upperbound_lin': np.array(self.original_upperbound_lin).tolist(),
                'original_lowerbound_lin': np.array(self.original_lowerbound_lin).tolist(),
            })
        
        filename = 'problem_description.json'

        # Check if the file exists before reading
        if os.path.exists(filename):
            # Load current data from the file
            with open(filename, 'r') as json_file:
                current_data = json.load(json_file)

            # Find the largest key and set the new key as largest_key + 1
            if current_data:
                largest_key = max(map(int, current_data.keys()))  # Convert keys to integers
                self.problem_id = largest_key + 1
            else:
                self.problem_id = 0  # If the file is empty, start with key 0
        else:
            # If the file does not exist, start with an empty dictionary and key 0
            current_data = {}
            self.problem_id = 0
        
        self.log_message.emit(f"Generated Problem id:{self.problem_id}")
        
        #Add new data under the determined key
        current_data[str(self.problem_id)] = problem_dict  # Ensure the key is a string for JSON compatibility

        with open(filename, 'w') as json_file:
                json.dump(current_data, json_file, indent=4)

    def kill_process(self):
        for future in self.all_futures:
            if not future.done():  # Only cancel if not already finished
                future.cancel()
    
    def stop(self):
        # First, cancel any running futures
        self.kill_process()

        # Then, terminate the thread
        self.terminate()

        # Optionally, wait for the thread to terminate
        self.wait()




if __name__ == '__main__':

      # Test inputs
    filter_type = 0
    order_current = 8
    accuracy = 1
    wordlength = 14
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

    gurobi_thread = 0
    pysat_thread = 0
    z3_thread = 1

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

    freq_upper[upper_half_point:end_point] = -10
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
        'gurobi_auto_thread': False,
        'seed': 0
    }
    
    
    # Instantiate the BackendMediator
    mediator = BackendMediator(input_data)

   
   


    mediator.run()

    # Create a QTimer in the main thread to poll the queue
    
    # Schedule the killing of the process after 5 seconds
    # def delayed_kill():
    #     print("killing process")
    #     mediator.stop()

    # # Use a QTimer to call `kill_process` after 5 seconds
    # kill_timer = QTimer()
    # kill_timer.singleShot(5000, delayed_kill)  # 5000 milliseconds = 5 seconds



    # Start the event loop
