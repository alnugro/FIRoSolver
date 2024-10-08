import sys
import os
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton
from PyQt6.QtCore import QProcess, pyqtSignal, QObject
import json
import tempfile
import numpy as np

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

    def __init__(self, initial_solver_input):
        super().__init__()
        self.initial_solver_input = initial_solver_input
        self.process = QProcess()
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stderr)
        self.process.finished.connect(self.process_finished)
        self.temp_input_file_name = None

    def run(self):
        # Write the initial_solver_input to a temporary JSON file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
            self.temp_input_file_name = temp_file.name
            json.dump(self.initial_solver_input, temp_file, cls=NumpyJSONEncoder)

        print("Running Backend Runner Script...")

        # Determine the path to the backend runner script
        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend', 'backend_runner.py'))

        # Start the backend runner script via QProcess, passing the temp input file
        self.process.start(sys.executable, ["-u",script_path, self.temp_input_file_name])

    def handle_stdout(self):
        data = self.process.readAllStandardOutput()
        stdout = bytes(data).decode('utf-8')

        print(stdout)

         # Split stdout into individual lines
        lines = stdout.splitlines()
        # Loop through lines and filter those that start with "@MSG@"
        for line in lines:
            if line.startswith('@MSG@'):
                self.log_message.emit(line)
    
   

    def handle_stderr(self):
        data = self.process.readAllStandardError()
        stderr = bytes(data).decode('utf-8')
        print(f"stderr {stderr}")
        if stderr == "problem is unsat"
        self.exception_message.emit(stderr)

    def process_finished(self):
        print("Backend Runner Script Finished.")
        self.finished.emit()
        # Clean up temporary input file
        if self.temp_input_file_name:
            try:
                os.remove(self.temp_input_file_name)
            except Exception as e:
                print(f"Error deleting temp file: {e}")

    def stop(self):
        if self.process.state() == QProcess.ProcessState.Running:
            self.process.kill()
            self.process.waitForFinished()

if __name__ == '__main__':
    # Test inputs
    filter_type = 0
    order_current = 6
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

    gain_wordlength = 6
    gain_intW = 2

    gurobi_thread = 0
    pysat_thread = 0
    z3_thread = 10

    timeout = 0

    space = order_current * accuracy * 50
    # Initialize freq_upper and freq_lower with NaN values
    freqx_axis = np.linspace(0, 1, space)
    freq_upper = np.full(space, np.nan)
    freq_lower = np.full(space, np.nan)

    # Manually set specific values for the elements of freq_upper and freq_lower in dB
    lower_half_point = int(0.3 * space)
    upper_half_point = int(0.6 * space)
    end_point = space

    freq_upper[0:lower_half_point] = 0
    freq_lower[0:lower_half_point] = 0

    freq_upper[upper_half_point:end_point] = -80
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

    # Start the event loop
    sys.exit(app.exec())
