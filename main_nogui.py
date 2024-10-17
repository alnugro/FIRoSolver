import numpy as np
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QObject
import sys
import os
import copy

''''this is the main file to run the solver without GUI, the result is saved in the result_valid.json file'''

try:
    from GUI.backend_mediator import BackendMediator
except:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from GUI.backend_mediator import BackendMediator

class IterationManager(QObject):
    def __init__(self, total_iterations):
        super().__init__()
        self.total_iterations = total_iterations
        self.current_iteration = 0
        self.delta = 0.1
        self.order_current = 16

    def start(self):
        self.start_next_iteration()

    def start_next_iteration(self):
        if self.current_iteration < self.total_iterations:
            # Prepare data for this iteration
            i = self.current_iteration
            delta = self.delta
            order_current = self.order_current

            # Test inputs (use your actual data preparation logic)
            filter_type = 0
            accuracy = 2
            wordlength = 10
            gain_upperbound = 1
            gain_lowerbound = 1
            coef_accuracy = 6
            intW = 1

            adder_count = None
            adder_depth = 0
            avail_dsp = 0
            adder_wordlength_ext = 2

            gain_intW = 4       
            gain_wordlength = wordlength + gain_intW - intW

            gurobi_thread = 10
            pysat_thread = 0
            z3_thread = 0

            timeout = 0

            passband_error = delta
            stopband_error = delta
            space = order_current * accuracy * 20

            freqx_axis = np.linspace(0, 1, space)
            freq_upper = np.full(space, np.nan)
            freq_lower = np.full(space, np.nan)

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

            ignore_lowerbound = -60

            upperbound_lin = np.copy(freq_upper)
            lowerbound_lin = np.copy(freq_lower)
            ignore_lowerbound_lin = 10 ** (ignore_lowerbound / 20)

            cutoffs_upper_ydata_lin = np.copy(cutoffs_upper_ydata)
            cutoffs_lower_ydata_lin = np.copy(cutoffs_lower_ydata)

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
                'adder_wordlength_ext': adder_wordlength_ext,
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
            }

            # Instantiate the BackendMediator
            self.mediator = BackendMediator(input_data)

            # Connect signals
            self.mediator.log_message.connect(lambda msg: print("Log:", msg))
            self.mediator.exception_message.connect(lambda msg: print("Error:", msg))
            self.mediator.finished.connect(self.on_mediator_finished)

            # Start the mediator
            self.mediator.run()

            self.current_iteration += 1
            self.delta *= 0.75
            self.order_current += 2
        else:
            # All iterations done
            print("All iterations completed.")
            app.quit()

    def on_mediator_finished(self):
        # Start next iteration
        self.start_next_iteration()

if __name__ == "__main__":
    # Initialize the QApplication
    app = QApplication(sys.argv)

    # Create and start the iteration manager
    manager = IterationManager(total_iterations=1)
    manager.start()

    sys.exit(app.exec())
