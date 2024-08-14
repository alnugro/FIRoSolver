import numpy as np
from z3 import *
import matplotlib.pyplot as plt
from concurrent.futures import TimeoutError, CancelledError
from pebble import ProcessPool, ProcessExpired
import time
import traceback
from copy import deepcopy



class SolverFunc():
    def __init__(self, filter_type, order):
        self.filter_type = filter_type
        self.half_order = (order // 2)

    def db_to_linear(self, db_arr):
        # Create a mask for NaN values
        nan_mask = np.isnan(db_arr)

        # Apply the conversion to non-NaN values (magnitude)
        linear_array = np.zeros_like(db_arr)
        linear_array[~nan_mask] = 10 ** (db_arr[~nan_mask] / 20)

        # Preserve NaN values
        linear_array[nan_mask] = np.nan
        return linear_array

    def cm_handler(self, m, omega):
        if self.filter_type == 0:
            if m == 0:
                return 1
            return 2 * np.cos(np.pi * omega * m)

        # Ignore the rest, its for later use if type 1 works
        if self.filter_type == 1:
            return 2 * np.cos(omega * np.pi * (m + 0.5))

        if self.filter_type == 2:
            return 2 * np.sin(omega * np.pi * (m - 1))

        if self.filter_type == 3:
            return 2 * np.sin(omega * np.pi * (m + 0.5))


class FIRFilter:
    def __init__(self, filter_type, order_upper, freqx_axis, freq_upper, freq_lower, ignore_lowerbound_lin, app=None):
        self.filter_type = filter_type
        self.order_upper = order_upper
        self.freqx_axis = freqx_axis
        self.freq_upper = freq_upper
        self.freq_lower = freq_lower
        self.ignore_lowerbound_lin = ignore_lowerbound_lin
        self.h_int_res = []
        self.app = app
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1)
        self.freq_upper_lin = 0
        self.freq_lower_lin = 0

        self.solver_timeout = 1


        self.order_current = int(self.order_upper)

        # Number of parallel solver instances
        self.num_instances = 8
        self.half_order = (self.order_current // 2)

        self.sf = SolverFunc(self.filter_type, self.order_current)
        self.freq_upper_lin = [self.sf.db_to_linear(f) for f in self.freq_upper]
        self.freq_lower_lin = [self.sf.db_to_linear(f) for f in self.freq_lower]



    def runsolver(self):
        print("solver Running")
        random_seeds = [i for i in range(self.num_instances)]
        self.parallel_solve(random_seeds)
        print("solver stopped")  

    def solve_with_seed(self, seed):
        ctx = z3.Context()
        h_int = [Int(f'h_int_{i}', ctx=ctx) for i in range(self.order_current + 1)]

        problem = list()

        print("solver called with seed: ", seed)
        
        sf = SolverFunc(self.filter_type, self.order_current)
        # Linearize the bounds
        freq_upper_lin = deepcopy(self.freq_upper_lin)
        freq_lower_lin = deepcopy(self.freq_lower_lin)

        

        # Create the sum constraints
        for i in range(len(self.freqx_axis)):
            term_sum_exprs = 0

            if np.isnan(freq_upper_lin[i]) or np.isnan(freq_lower_lin[i]):
                continue

            for j in range(self.half_order + 1):
                cm_const = sf.cm_handler(j, self.freqx_axis[i])
                term_sum_exprs += h_int[j] * cm_const
            problem.append(term_sum_exprs <= freq_upper_lin[i])

            if freq_lower_lin[i] < self.ignore_lowerbound_lin:
                problem.append(term_sum_exprs >= -freq_upper_lin[i])
                continue
            problem.append(term_sum_exprs >= freq_lower_lin[i])

        for i in range(self.half_order + 1):
            problem.append(h_int[i] <= 2**12)
            problem.append(h_int[i] >= -2**12)

        solver = z3.Solver(ctx=ctx)
        solver.set("random_seed", seed)
        solver.add(problem)
        result = []

        if solver.check() == z3.sat:
            print("solver sat with seed", seed)
            model = solver.model()
            
            for i in range(self.half_order + 1):
                result.append(model[h_int[i]].as_long())
            print("solver is returning")
            return result

        elif solver.check() == z3.unsat:
            result.append("unsat")
            print("solver unsat")
            return result
        else:
            print("Something weird came out")
            result.append("unknown")
            return result

    def parallel_solve(self, random_seeds):
        with ProcessPool(max_workers=self.num_instances) as pool:
            futures = {}
            for seed in random_seeds:
                future = pool.schedule(self.solve_with_seed, args=(seed,), timeout=self.solver_timeout)
                futures[future] = seed
                future.add_done_callback(self.task_done(seed, futures))



    def task_done(self, seed, futures):
        def callback(future):
            try:
                result = future.result()  # blocks until results are ready
                print(f"Task done with result: {result} for seed {seed}")

                # Cancel all other processes
                for f in futures:
                    if f is not future and not f.done():
                        f.cancel()
                        print("Cancelled another process")

                # Handle result
                if result[0] == "unsat":
                    print("unsat")
                elif result[0] == "unknown":
                    print("weird thing happened")
                else:
                    print(f"Solution found with seed {seed}: {result}")
                    self.h_int_res = result

                
            except CancelledError as error:
                print("Seed ", seed, " is cancelled")
            except TimeoutError as error:
                print(f"Function took longer than {self.solver_timeout} seconds for seed {seed}")
            except ProcessExpired as error:
                print(f"Process {error.pid} expired for seed {seed}")
            except Exception as error:
                print(f"Function raised an exception for seed {seed}: {error}")
                traceback.print_exc()  # Print the full traceback to get more details

        return callback


    def plot_result(self, result_coef):
        print("result plotter called")
        fir_coefficients = np.array([])
        for i in range(len(result_coef)):
            fir_coefficients = np.append(fir_coefficients, result_coef[(i + 1) * -1])

        for i in range(len(result_coef) - 1):
            fir_coefficients = np.append(fir_coefficients, result_coef[i + 1])

        print(fir_coefficients)

        print("Fir coef in mp", fir_coefficients)

        # Compute the FFT of the coefficients
        N = 5120  # Number of points for the FFT
        frequency_response = np.fft.fft(fir_coefficients, N)
        frequencies = np.fft.fftfreq(N, d=1.0)[:N // 2]  # Extract positive frequencies up to Nyquist

        # Compute the magnitude and phase response for positive frequencies
        magnitude_response = np.abs(frequency_response)[:N // 2]

        # Convert magnitude response to dB
        magnitude_response_db = 20 * np.log10(np.where(magnitude_response == 0, 1e-10, magnitude_response))

        # Normalize frequencies to range from 0 to 1
        omega = frequencies * 2 * np.pi
        normalized_omega = omega / np.max(omega)
        self.ax1.set_ylim([-10, 10])

        for i in range(len(freqx_axis)):
            print(self.freqx_axis[i] , " and ", self.freq_upper_lin[i])

        # Plot input
        self.ax1.scatter(self.freqx_axis, self.freq_upper_lin, color='r', s=20, picker=5)
        self.ax1.scatter(self.freqx_axis, self.freq_lower_lin, color='b', s=20, picker=5)

        # Plot the updated upper_ydata
        self.ax1.plot(normalized_omega, magnitude_response, color='y')

        if self.app:
            self.app.canvas.draw()

    def plot_validation(self):
        print("Validation plotter called")
        self.half_order = (self.order_current // 2)
        sf = SolverFunc(self.filter_type, self.order_current)
        # Array to store the results of the frequency response computation
        computed_frequency_response = []

        # Recompute the frequency response for each frequency point
        for i in range(len(self.freqx_axis)):
            omega = self.freqx_axis[i]
            term_sum_exprs = 0

            # Compute the sum of products of coefficients and the cosine/sine terms
            for j in range(self.half_order + 1):
                cm_const = sf.cm_handler(j, omega)
                term_sum_exprs += self.h_int_res[j] * cm_const

            # Append the computed sum expression to the frequency response list
            computed_frequency_response.append(np.abs(term_sum_exprs))

        # Normalize frequencies to range from 0 to 1 for plotting purposes

        # Plot the computed frequency response
        self.ax2.plot([x / 1 for x in self.freqx_axis], computed_frequency_response, color='green', label='Computed Frequency Response')

        self.ax2.set_ylim(-10, 10)

        if self.app:
            self.app.canvas.draw()


if __name__ == '__main__':
    # Test inputs
    filter_type = 0
    order_upper = 14

    # Initialize freq_upper and freq_lower with NaN values
    freqx_axis = np.linspace(0, 1, 6 * order_upper)  # according to Mr. Kumms paper
    freq_upper = np.full(6 * order_upper, np.nan)
    freq_lower = np.full(6 * order_upper, np.nan)

    # Manually set specific values for the elements of freq_upper and freq_lower in dB
    freq_upper[10:20] = 4
    freq_lower[10:20] = -1

    freq_upper[40:50] = -2
    freq_lower[40:50] = -1000

    # Beyond this bound lowerbound will be ignored
    ignore_lowerbound_lin = 0.0001

    # Create FIRFilter instance
    fir_filter = FIRFilter(filter_type, order_upper, freqx_axis, freq_upper, freq_lower, ignore_lowerbound_lin)

    # Run solver and plot result
    fir_filter.runsolver()
    if fir_filter.h_int_res:
        


        fir_filter.plot_result(fir_filter.h_int_res)
        fir_filter.plot_validation()

    # Show plot
    plt.show()
