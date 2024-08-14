import random
import time
import numpy as np
from pysat_formulation import FIRFilterPysat
from z3_sat_formulation import FIRFilterZ3
from kumm_formulation import FIRFilterKumm
from pebble import ProcessPool
from concurrent.futures import TimeoutError  # Correct import for TimeoutError

# Initialize global variable
it = 4
timeout = 4  # Timeout in seconds (5 minutes)
random_seed = 1
random.seed(random_seed)

def generate_random_filter_params():
    global it
    filter_type = 0
    order_upper = it
    accuracy = random.choice([1, 2, 3, 4, 5])
    adder_count = np.abs(it - (random.choice([1, 2, 3, 4, it - 4])))
    wordlength = random.choice([10, 12, 14, 16])
    upper_cutoff = random.choice([0.6, 0.7, 0.8, 0.9])
    lower_cutoff = random.choice([0.2, 0.3, 0.4, 0.5])
    lower_half_point = int(lower_cutoff * (accuracy * order_upper))
    upper_half_point = int(upper_cutoff * (accuracy * order_upper))
    end_point = accuracy * order_upper
    freqx_axis = np.linspace(0, 1, accuracy * order_upper)
    freq_upper = np.full(accuracy * order_upper, np.nan)
    freq_lower = np.full(accuracy * order_upper, np.nan)
    passband_upperbound = random.choice([0, 1, 2, 3, 4, 5])
    passband_lowerbound = random.choice([0, -1, -2])
    stopband_upperbound = random.choice([-10, -20, -30, -40, -50])
    stopband_lowerbound = -1000
    freq_upper[0:lower_half_point] = passband_upperbound
    freq_lower[0:lower_half_point] = passband_lowerbound
    freq_upper[upper_half_point:end_point] = stopband_upperbound
    freq_lower[upper_half_point:end_point] = stopband_lowerbound
    ignore_lowerbound_lin = -20
    it += 1
    return (filter_type, order_upper, freqx_axis, freq_upper, freq_lower, ignore_lowerbound_lin, adder_count, wordlength, accuracy, upper_cutoff, lower_cutoff, passband_upperbound, passband_lowerbound, stopband_upperbound, stopband_lowerbound)

def run_solver(solver_instance):
    return solver_instance.runsolver()

if __name__ == "__main__":
    # Write header
    with open("res_z3_vs_naive_vs_pysat.txt", "w") as file:
        file.write("time_smt, result_smt, time_sat, result_sat, time_naive, result_naive, filter_type, order_upper, accuracy, adder_count, wordlength, upper_cutoff, lower_cutoff, passband_upperbound, passband_lowerbound, stopband_upperbound, stopband_lowerbound\n")

    results = []

    with ProcessPool(max_workers=3) as pool:  # Using a pool of 3 workers for parallel processing
        for i in range(50):
            print("Running test: ", i)
            params = generate_random_filter_params()
            filter_type, order_upper, freqx_axis, freq_upper, freq_lower, ignore_lowerbound_lin, adder_count, wordlength, accuracy, upper_cutoff, lower_cutoff, passband_upperbound, passband_lowerbound, stopband_upperbound, stopband_lowerbound = params
            
            # Creating solver instances
            pysat_instance = FIRFilterPysat(filter_type, order_upper, freqx_axis, freq_upper, freq_lower, ignore_lowerbound_lin, adder_count, wordlength)
            z3_instance = FIRFilterZ3(filter_type, order_upper, freqx_axis, freq_upper, freq_lower, ignore_lowerbound_lin, adder_count, wordlength)
            naive_instance = FIRFilterKumm(filter_type, order_upper, freqx_axis, freq_upper, freq_lower, ignore_lowerbound_lin, adder_count, wordlength)
            
            # Run the solvers in parallel with a timeout
            future_z3 = pool.schedule(run_solver, args=(z3_instance,), timeout=timeout)
            future_pysat = pool.schedule(run_solver, args=(pysat_instance,), timeout=timeout)
            future_naive = pool.schedule(run_solver, args=(naive_instance,), timeout=timeout)
            
            try:
                time1, result1 = future_z3.result()
            except TimeoutError:
                time1, result1 = timeout, "Timeout"
            
            try:
                time2, result2 = future_pysat.result()
            except TimeoutError:
                time2, result2 = timeout, "Timeout"
            
            try:
                time3, result3 = future_naive.result()
            except TimeoutError:
                time3, result3 = timeout, "Timeout"
            
            results.append((time1, result1, time2, result2, time3, result3, *params))
            
            with open("res_z3_vs_naive_vs_pysat.txt", "a") as file:
                file.write(f"{time1}, {result1}, {time2}, {result2}, {time3}, {result3}, {filter_type}, {order_upper}, {accuracy}, {adder_count}, {wordlength}, {upper_cutoff}, {lower_cutoff}, {passband_upperbound}, {passband_lowerbound}, {stopband_upperbound}, {stopband_lowerbound}\n")
            
            print("Test ", i, " is completed")

    print("Benchmark completed and results saved to res_z3_vs_naive_vs_pysat.txt")
