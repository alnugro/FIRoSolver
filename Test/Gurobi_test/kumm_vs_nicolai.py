import random
import time
import numpy as np
from gurobi_baseline import FIRFilterGurobi
from gurobi_kumm_true import FIRFilterKumm
from pebble import ProcessPool
from concurrent.futures import TimeoutError  # Correct import for TimeoutError
import multiprocessing

# Initialize global variable
it = 6
timeout = 60  # Timeout in seconds (5 minutes)
random_seed = 1
random.seed(random_seed)

def generate_random_filter_params():
    global it
    iter = int(it)
    filter_type = 0
    order_upper = iter
    if order_upper%2 != 0:
        order_upper+=1
    accuracy = 1
    adder_count = int(order_upper//2)
    wordlength = random.choice([10, 12, 14, 16])
    upper_cutoff = random.choice([0.6, 0.7])
    lower_cutoff = random.choice([0.4, 0.5])
    lower_half_point = int(lower_cutoff * (accuracy * order_upper))
    upper_half_point = int(upper_cutoff * (accuracy * order_upper))
    end_point = accuracy * order_upper
    freqx_axis = np.linspace(0, 1, accuracy * order_upper)
    freq_upper = np.full(accuracy * order_upper, np.nan)
    freq_lower = np.full(accuracy * order_upper, np.nan)
    passband_upperbound = random.choice([0, 1, 2, 3, 4, 5])
    passband_lowerbound = random.choice([0, -1, -2])
    stopband_upperbound = random.choice([-10, -20, -30])
    stopband_lowerbound = -1000
    freq_upper[0:lower_half_point] = passband_upperbound
    freq_lower[0:lower_half_point] = passband_lowerbound
    freq_upper[upper_half_point:end_point] = stopband_upperbound
    freq_lower[upper_half_point:end_point] = stopband_lowerbound
    ignore_lowerbound_lin = -20
    it += 0.3
    return (filter_type, order_upper, freqx_axis, freq_upper, freq_lower, ignore_lowerbound_lin, adder_count, wordlength, accuracy, upper_cutoff, lower_cutoff, passband_upperbound, passband_lowerbound, stopband_upperbound, stopband_lowerbound)



if __name__ == "__main__":
    multiprocessing.freeze_support()

    # Write header
    with open("kumm_vs_prop.txt", "w") as file:
        file.write("order;satisfiability Baseline; duration Baseline ;satisfiability Kumm; duration Kumm;Bas_faster_flag;Kum_faster_flag\n")

    results = []
    timeout = 300 #5 mins timeout

    
    for i in range(1000000):
        print("Running test: ", i)
        params = generate_random_filter_params()
        filter_type, order_upper, freqx_axis, freq_upper, freq_lower, ignore_lowerbound_lin, adder_count, wordlength, accuracy, upper_cutoff, lower_cutoff, passband_upperbound, passband_lowerbound, stopband_upperbound, stopband_lowerbound = params
        
        # Creating solver instances
        baseline_instance = FIRFilterGurobi(filter_type, order_upper, freqx_axis, freq_upper, freq_lower, ignore_lowerbound_lin, adder_count, wordlength,timeout)
        kumm_instance = FIRFilterKumm(filter_type, order_upper, freqx_axis, freq_upper, freq_lower, ignore_lowerbound_lin, adder_count, wordlength,timeout)
        
        duration,satisfiability= baseline_instance.runsolver()
        duration_kumm,satisfiability_kumm = kumm_instance.runsolver()
             

        Bas_faster_flag = 0
        Kum_faster_flag = 0
        if duration_kumm < duration:
            Kum_faster_flag = 1
        else: Bas_faster_flag = 1

        
        
        
        with open("kumm_vs_prop.txt", "a") as file:
                file.write(f"{order_upper};{satisfiability}; {duration};{satisfiability_kumm};{duration_kumm};{Bas_faster_flag};{Kum_faster_flag}\n")
              
        print("Test ", i, " is completed")

    print("Benchmark completed and results saved to kumm_vs_prop.txt")