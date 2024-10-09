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




if __name__ == "__main__":
    # Test inputs
    accuracy = 3
    wordlength = 11
    gain_upperbound = 1
    gain_lowerbound = 1
    coef_accuracy = 10
    intW = 2

    adder_count = None
    adder_depth = 0
    avail_dsp = 0
    adder_wordlength_ext = 2

    gain_intW = 4       
    gain_wordlength = wordlength + gain_intW - intW
    

    gurobi_thread = 1
    pysat_thread = 0
    z3_thread = 0

    timeout = 1200

    with open("gurobi_bench.txt", "w") as file:
            file.write("satisfiability;filter_order;number_of_adders;number_of_mult_adders;number_of_struct_adders;h;solver_duration;delta;leak\n")
    filter_type = 0
    order_current = 6
    delta = 0.4


    for i in range(10):
        passband_error = delta
        stopband_error = delta
        space = order_current * accuracy * 50
        # Initialize freq_upper and freq_lower with NaN values
        freqx_axis = np.linspace(0, 1, space)
        freq_upper = np.full(space, np.nan)
        freq_lower = np.full(space, np.nan)

        # Manually set specific values for the elements of freq_upper and freq_lower in dB
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

        # Beyond this bound, lowerbound will be ignored
        ignore_lowerbound = -60

        # Linearize the bounds
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

        # Create an instance of SolverBackend
        backend = SolverBackend(input_data)

        
        start_time = time.time()
        # start presolve
        satisfiability = 'sat'
        try:
            presolve_result = backend.solver_presolve()
        except ValueError as e:
            if str(e) == "problem is unsat":
                print("problem is unsat")
                satisfiability = 'unsat'
                with open("gurobi_bench.txt", "a") as file:
                    file.write(f"{satisfiability};{order_current};{0};{0};{0};{0};{0};{delta}\n")
                continue
        except Exception as e:
            print("error in presolve")
            print(e)
            traceback.print_exc()
            with open("gurobi_bench.txt", "a") as file:
                file.write(f"error;{order_current};{0};{0};{0};{0};{0};{delta}\n")
            continue

        target_result, best_adderm ,total_adder,adder_s_h_zero_best = backend.find_best_adder_s(presolve_result)
            # target_result2, best_adderm2, total_adder2, adderm_h_zero_best = backend.find_best_adder_m(presolve_result)
        leaked = 'no'
        while True:
            leaks, leaks_mag = backend.result_validator(target_result['h_res'],target_result['gain'])
            
            if leaks:
                print("leak_flag")
                leaked = 'yes'
                target_result, satisfiability = backend.solving_result_barebone(presolve_result,best_adderm,adder_s_h_zero_best)
                if satisfiability == 'unsat':
                    print("problem is unsat from asserting the leak to the problem")
                    break
            else:
                if leaked == 'no':
                    print("no leak")
                else:
                    print("leak fixed")
                    leaked = 'fixed'
                break
        
        h_res = [res * 2**(wordlength-intW) for res in target_result['h_res']]
        end_time = time.time()
        duration = end_time - start_time
        adder_s = (total_adder- best_adderm)/2 if filter_type == 0 or filter_type == 2 else (total_adder - best_adderm - 1)/2
        with open("gurobi_bench.txt", "a") as file:
            file.write(f"{satisfiability};{order_current};{total_adder};{best_adderm};{adder_s};{h_res};{duration};{delta};{leaked}\n")
        order_current += 2
        delta = round(delta * 0.75, 6)
        backend = None
