import sys
import os
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton
from PyQt6.QtCore import QProcess, pyqtSignal, QObject
import json
import tempfile
import numpy as np
import copy

try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from backend.backend_main import SolverBackend
except:
    from backend.backend_main import SolverBackend

# Custom JSON encoder to handle NumPy arrays
class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class BackendMediator(QObject):
    log_message = pyqtSignal(str)
    exception_message = pyqtSignal(str)
    finished = pyqtSignal()
    all_finished = pyqtSignal()  # Signal emitted when all solvers have finished

    def __init__(self, initial_solver_input):
        super().__init__()
        self.initial_solver_input = initial_solver_input
        self.processes = {}  # Dictionary to store QProcess instances
        self.temp_input_files = {}  # Dictionary to store temp file paths
        self.stdout_buffers = {}  # Buffers for each process's stdout
        self.solver_count = 0  # Total number of solvers running
        self.problem_id = 0
        
        self.generate_bound_description()

        self.initial_solver_input.update({
            'problem_id' : self.problem_id
        })

        self.verbose = False


    def generate_bound_description(self):
        problem_dict = copy.deepcopy(self.initial_solver_input)
        backend = SolverBackend(self.initial_solver_input)
        problem_dict.update ({
                'xdata' :  np.array(backend.xdata).tolist(),
                'upperbound_lin':  np.array(backend.upperbound_lin).tolist(),
                'lowerbound_lin': np.array(backend.lowerbound_lin).tolist(),
                'original_xdata': np.array(self.initial_solver_input['original_xdata']).tolist(), #convert them to list, ndarray is not supported with json
                'original_upperbound_lin': np.array(self.initial_solver_input['original_upperbound_lin']).tolist(),
                'original_lowerbound_lin': np.array(self.initial_solver_input['original_lowerbound_lin']).tolist(),
                'cutoffs_x': np.array(self.initial_solver_input['cutoffs_x']).tolist(),
                'cutoffs_upper_ydata_lin': np.array(self.initial_solver_input['cutoffs_upper_ydata_lin']).tolist(),
                'cutoffs_lower_ydata_lin': np.array(self.initial_solver_input['cutoffs_lower_ydata_lin']).tolist(),
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

    def run(self):
        solvers_to_run = []

        if self.initial_solver_input['gurobi_thread'] > 0:
            solvers_to_run.append('gurobi')
        if self.initial_solver_input['pysat_thread'] > 0:
            solvers_to_run.append('pysat')
        if self.initial_solver_input['z3_thread'] > 0:
            solvers_to_run.append('z3')

        self.solver_count = len(solvers_to_run)

        for solver_name in solvers_to_run:
            self.start_solver_process(solver_name)

    def start_solver_process(self, solver_name):
        process = QProcess()
        process.readyReadStandardOutput.connect(lambda sn=solver_name: self.handle_stdout(sn))
        process.readyReadStandardError.connect(lambda sn=solver_name: self.handle_stderr(sn))
        process.finished.connect(lambda exitCode, exitStatus, sn=solver_name: self.process_finished(sn))
        self.processes[solver_name] = process
        self.stdout_buffers[solver_name] = ''

        print(f"solver name {solver_name}")
        #rework the solver input
        solver_input = copy.deepcopy(self.initial_solver_input)
        if solver_name == 'gurobi':
            solver_input.update({
                'gurobi_thread': self.initial_solver_input['gurobi_thread'],
                'pysat_thread': 0,
                'z3_thread': 0,
            })
        elif solver_name == 'pysat':
             solver_input.update({
                'gurobi_thread': 0,
                'pysat_thread': self.initial_solver_input['pysat_thread'],
                'z3_thread': 0,
            })
        elif solver_name == 'z3':
             solver_input.update({
                'gurobi_thread': 0,
                'pysat_thread': 0,
                'z3_thread': self.initial_solver_input['z3_thread'],
            })
        else:
            raise NotImplementedError(f"This solver: {solver_name} is not implemented")
        

        # Create temp input file for this solver
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
            self.temp_input_files[solver_name] = temp_file.name
            json.dump(solver_input, temp_file, cls=NumpyJSONEncoder)

        # Adjust the script path according to the solver
        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend', 'backend_runner.py'))

        # Start the solver script via QProcess
        process.start(sys.executable, ["-u", script_path, self.temp_input_files[solver_name]])

    def handle_stdout(self, solver_name):
        process = self.processes[solver_name]
        data = process.readAllStandardOutput()
        stdout = bytes(data).decode('utf-8')
        self.stdout_buffers[solver_name] += stdout
        if self.verbose:
            print(stdout)
        # Process complete lines
        while '\n' in self.stdout_buffers[solver_name]:
            line, self.stdout_buffers[solver_name] = self.stdout_buffers[solver_name].split('\n', 1)
            # Emit only lines that start with "@MSG@"
            if line.startswith('@MSG@'):
                clean_line = line[len('@MSG@'):]  # Remove the "@MSG@" prefix
                self.log_message.emit(f"[{solver_name}] {clean_line.strip()}")

    def handle_stderr(self, solver_name):
        process = self.processes[solver_name]
        data = process.readAllStandardError()
        stderr = bytes(data).decode('utf-8')
        self.exception_message.emit(f"[{solver_name}] {stderr}")

    def process_finished(self, solver_name):
        print(f"{solver_name} Runner Script Finished.")
        self.log_message.emit(f"#### {solver_name} is done! ####\n Check result data.")

        # Clean up temporary input file
        if solver_name in self.temp_input_files:
            try:
                os.remove(self.temp_input_files[solver_name])
            except Exception as e:
                print(f"Error deleting temp file for {solver_name}: {e}")
            del self.temp_input_files[solver_name]

        # Remove the process from the dictionary
        if solver_name in self.processes:
            self.processes[solver_name].deleteLater()
            del self.processes[solver_name]

        # Check if all processes have finished
        if not self.processes:
            self.all_finished.emit()
            self.finished.emit()

    def stop(self):
        # Create a copy of the keys to iterate over
        solver_names = list(self.processes.keys())

        for solver_name in solver_names:
            process = self.processes[solver_name]
            if process and process.state() == QProcess.ProcessState.Running:
                process.kill()
                process.waitForFinished()

if __name__ == '__main__':
    # Test inputs
    filter_type = 0
    order_current = 10
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

    gain_wordlength = 14
    gain_intW = 4

    gurobi_thread = 5
    pysat_thread = 5
    z3_thread = 5

    timeout = 0

    space = order_current * accuracy * 50
    # Initialize freq_upper and freq_lower with NaN values
    freqx_axis = np.linspace(0, 1, space)
    freq_upper = np.full(space, np.nan)
    freq_lower = np.full(space, np.nan)

    # Manually set specific values for the elements of freq_upper and freq_lower in dB
    lower_half_point = int(0.3 * space)
    upper_half_point = int(0.5 * space)
    end_point = space

    freq_upper[0:lower_half_point] = 5
    freq_lower[0:lower_half_point] = 0

    freq_upper[upper_half_point:end_point] = -15
    freq_lower[upper_half_point:end_point] = -1000

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
    ignore_lowerbound = -40

    # Linearize the bounds
    upperbound_lin = [10 ** (f / 20) if not np.isnan(f) else np.nan for f in freq_upper]
    lowerbound_lin = [10 ** (f / 20) if not np.isnan(f) else np.nan for f in freq_lower]
    ignore_lowerbound_lin = 10 ** (ignore_lowerbound / 20)

    cutoffs_upper_ydata_lin = [
        10 ** (f / 20) if not np.isnan(f) else np.nan for f in cutoffs_upper_ydata
    ]
    cutoffs_lower_ydata_lin = [
        10 ** (f / 20) if not np.isnan(f) else np.nan for f in cutoffs_lower_ydata
    ]

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
        'adder_wordlength_ext': adder_wordlength_ext,  # This is extension, not the adder wordlength
        'gain_wordlength': gain_wordlength,
        'gain_intW': gain_intW,
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
        'patch_multiplier': 1,
        'gurobi_auto_thread': False,
        'seed': 0
    }
    from PyQt6.QtCore import QTimer

    # Initialize the QApplication
    app = QApplication(sys.argv)

    # Instantiate the BackendMediator
    mediator = BackendMediator(input_data)

    # Connect signals to print output and errors
    mediator.log_message.connect(lambda msg: print("Log:", msg))
    mediator.exception_message.connect(lambda msg: print("Error:", msg))
    mediator.finished.connect(app.quit)

    # Start the mediator
    mediator.run()

    # Create a QTimer in the main thread to poll the queue
    
    # # Schedule the killing of the process after 5 seconds
    # def delayed_kill():
    #     print("killing process")
    #     mediator.stop()

    # # Use a QTimer to call `kill_process` after 5 seconds
    # kill_timer = QTimer()
    # kill_timer.singleShot(5000, delayed_kill)  # 5000 milliseconds = 5 seconds

    # Start the event loop
    sys.exit(app.exec())
