import sys
import os
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QThread, pyqtSignal
import numpy as np
import copy

try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from backend.backend_main import SolverBackend
except:
    from backend.backend_main import SolverBackend


class AdditionalFuncMediator(QThread):
    # Define a signal to send results back to the main thread
    result_signal = pyqtSignal(object, int, int)
    exception_message = pyqtSignal(str)

    def __init__(self, initial_solver_input, option=None):
        super().__init__()
        self.initial_solver_input = copy.deepcopy(initial_solver_input)
        self.option = option

    def run(self):
        backend = SolverBackend(self.initial_solver_input)
        if self.option == 'automatic':
            # Run the automatic parameter search
            try:
                best_target_result, best_filter_type, wordlength = backend.automatic_param_search()
                # Emit the results through the signal
                self.result_signal.emit(best_target_result, best_filter_type, wordlength)
            except Exception as e:
                self.exception_message.emit(str(e))
                self.result_signal.emit(None, 0, 0)
        elif self.option == 'quick_check_sat':
            try:
                # Run the asserted parameter search
                target_result = backend.asserted_param_quick_check()
                # Emit the results through the signal
                self.result_signal.emit(target_result, 0, 0)
            except Exception as e:
                self.exception_message.emit(str(e))
                self.result_signal.emit(None, 0, 0)
            

#AdditionalFuncMediator Test
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
    mediator = AdditionalFuncMediator(input_data)

    # Start the mediator
    mediator.run()

    
    # Start the event loop
    sys.exit(app.exec())
