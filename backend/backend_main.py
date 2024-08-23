import random
import time
import numpy as np
from backend.formulation_pysat import FIRFilterPysat
from backend.formulation_z3_pbsat_ import FIRFilterZ3
from backend.formulation_gurobi import FIRFilterGurobi
from pebble import ProcessPool
from concurrent.futures import TimeoutError  # Correct import for TimeoutError
import multiprocessing

class SolverBackend():
    def __init__(self, input):
        self.filter_type = input['filter_type']
        self.order_upper = input['order_upper']
        self.order_lower = input['order_lower']
        self.freqx_axis = input['freqx_axis']
        self.freq_upper = input['freq_upper']
        self.freq_lower = input['freq_lower']
        self.ignore_lowerbound = input['ignore_lowerbound']
        self.adder_count = input['adder_count']
        self.wordlength = input['wordlength']
        self.adder_depth = input['adder_depth']
        self.avail_dsp = input['avail_dsp']
        self.adder_wordlength_ext = input['adder_wordlength_ext']
        self.gain_upperbound = input['gain_upperbound']
        self.gain_lowerbound = input['gain_lowerbound']
        self.intW = input['intW']
        
        self.gurobi_thread = input['gurobi_thread']
        self.pysat_thread = input['pysat_thread']
        self.z3_thread = input['z3_thread']

        self.timeout = input['timeout']
        self.max_iteration = input['max_iteration']
        self.start_with_worst_error = input['start_with_worst_error']

    def no_gurobi_presolve(self):
        pass

    def gurobi_presolve(self):
        pass


    def run_solver(self,solver_instance):
        return solver_instance.runsolver()

    def solver_executor(self):
        multiprocessing.freeze_support()

        # Write header
        with open("res_z3_vs_naive_vs_pysat.txt", "w") as file:
            file.write("time_smt, result_smt, time_sat, result_sat, time_naive, result_naive, filter_type, order_upper, accuracy, adder_count, wordlength, upper_cutoff, lower_cutoff, passband_upperbound, passband_lowerbound, stopband_upperbound, stopband_lowerbound\n")

        results = []

        with ProcessPool(max_workers=3) as pool:  # Using a pool of 3 workers for parallel processing
            for i in range(1000000):
                print("Running test: ", i)
                # Creating solver instances
                pysat_instance = FIRFilterPysat(
                    self.filter_type, 
                    self.order_upper, 
                    self.freqx_axis, 
                    self.freq_upper, 
                    self.freq_lower, 
                    self.ignore_lowerbound, 
                    self.adder_count, 
                    self.wordlength, 
                    self.adder_depth,
                    self.avail_dsp,
                    self.adder_wordlength_ext,
                    self.gain_upperbound,
                    self.gain_lowerbound,
                    self.intW,
                    )
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
