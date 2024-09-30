import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import time
import math
import random


class SolverFunc():
    def __init__(self,input_data):
        self.filter_type = None
        self.order_upperbound = None

        self.original_xdata = None
        self.original_upperbound_lin = None
        self.original_lowerbound_lin = None

        self.cutoffs_x = None
        self.cutoffs_upper_ydata_lin = None
        self.cutoffs_lower_ydata_lin = None

        self.solver_accuracy_multiplier = None

        # Dynamically assign values from input_data, skipping any keys that don't have matching attributes
        for key, value in input_data.items():
            if hasattr(self, key):  # Only set attributes that exist in the class
                setattr(self, key, value)
        
        self.overflow_count = 0

    
    def cm_handler(self,m,omega):
        if self.filter_type == 0:
            if m == 0:
                return 1
            cm=(2*np.cos(np.pi*omega*m))
            return cm
        
        #ignore the rest, its for later use if type 1 works
        if self.filter_type == 1:
            return 2*np.cos(omega*np.pi*(m+0.5))

        if self.filter_type == 2:
            return 2*np.sin(omega*np.pi*(m-1))

        if self.filter_type == 3:
            return 2*np.sin(omega*np.pi*(m+0.5))
        





class FIRFilter:
    def __init__(self, 
                 filter_type, 
                 order, 
                 freqx_axis, 
                 upperbound_lin, 
                 lowerbound_lin, 
                 ignore_lowerbound, 
                 adder_count, 
                 wordlength, 
                 adder_depth,
                 avail_dsp,
                 adder_wordlength_ext,
                 gain_upperbound,
                 gain_lowerbound,
                 coef_accuracy,
                 intW,
                 app = None
                 ):
        self.app = app
        self.fig, (self.ax1, self.ax2) = plt.subplots(2,1)
        
        self.filter_type = filter_type
        self.order = order
        self.freqx_axis = freqx_axis

        self.h_res = []
        self.gain_res = 0

        self.wordlength = wordlength
        self.max_adder = adder_count

        self.upperbound_lin=upperbound_lin
        self.lowerbound_lin=lowerbound_lin

        self.coef_accuracy = coef_accuracy
        self.intW = intW
        self.fracW = self.wordlength - self.intW

        self.gain_upperbound= gain_upperbound
        self.gain_lowerbound= gain_lowerbound

        self.ignore_lowerbound = ignore_lowerbound

        self.adder_depth = adder_depth
        self.avail_dsp = avail_dsp
        self.adder_wordlength = self.wordlength + adder_wordlength_ext
        self.result_model = {}

    def get_solver_func_dict(self):
        input_data_sf = {
        'filter_type': self.filter_type,
        'order_upperbound': self.order,
        }

        return input_data_sf
    
    def run_barebone(self, thread, minmax_option = None, h_zero_count = None):
        

        self.h_res = []
        self.gain_res = 0
        target_result = {}
        self.order_current = int(self.order)
        half_order = (self.order_current // 2) if self.filter_type == 0 or self.filter_type == 2 else (self.order_current // 2) - 1
        
        print("Gurobi solver called")
        sf = SolverFunc(self.get_solver_func_dict())

        # print(f"upperbound_lin: {self.upperbound_lin}")
        # print(f"lowerbound_lin: {self.lowerbound_lin}")


         # linearize the bounds
        internal_upperbound_lin = [math.ceil((f)*(10**self.coef_accuracy)*(2**(self.fracW))) if not np.isnan(f) else np.nan for f in self.upperbound_lin]
        internal_lowerbound_lin = [math.floor((f)*(10**self.coef_accuracy)*(2**(self.fracW))) if not np.isnan(f) else np.nan for f in self.lowerbound_lin]
        internal_ignore_lowerbound = self.ignore_lowerbound*(10**self.coef_accuracy)*(2**self.fracW)


        # print("Running Gurobi with the following parameters:")
        # print(f"thread: {thread}")
        # print(f"minmax_option: {minmax_option}")
        # print(f"h_zero_count: {h_zero_count}")
        # print(f"filter_type: {self.filter_type}")
        # print(f"order_current: {self.order}")
        # print(f"freqx_axis: {self.freqx_axis}")
        # print(f"upperbound_lin: {internal_upperbound_lin}")
        # print(f"lowerbound_lin: {internal_lowerbound_lin}")
        # print(f"ignore_lowerbound: {internal_ignore_lowerbound}")
        # print(f"gain_upperbound: {self.gain_upperbound}")
        # print(f"gain_lowerbound: {self.gain_lowerbound}")
        # print(f"wordlength: {self.wordlength}")
        # print(f"fracW: {self.fracW}")
        
        model = gp.Model()
        model.setParam('Threads', thread)
        model.setParam('OutputFlag', 0)


        h = [[model.addVar(vtype=GRB.BINARY, name=f'h_{a}_{w}') for w in range(self.wordlength)] for a in range(half_order + 1)]
        gain = model.addVar(vtype=GRB.CONTINUOUS, lb=self.gain_lowerbound, ub=self.gain_upperbound, name="gain")



        for omega in range(len(self.freqx_axis)):
            if np.isnan(internal_lowerbound_lin[omega]):
                continue

            h_sum_temp = 0

            for m in range(half_order+1):
                cm = sf.cm_handler(m, self.freqx_axis[omega])
                for w in range(self.wordlength):
                    if w==self.wordlength-1:
                        cm_word_prod= int(cm*(10** self.coef_accuracy)*(-1*(2**w)))
                    else: cm_word_prod= int(cm*(10** self.coef_accuracy)*(2**w))
                    h_sum_temp += h[m][w]*cm_word_prod

            model.update()
            # print(f"sum temp is{h_sum_temp}")
            model.addConstr(h_sum_temp <= gain*internal_upperbound_lin[omega])
            
            
            if internal_lowerbound_lin[omega] < internal_ignore_lowerbound:
                model.addConstr(h_sum_temp >= gain*-internal_upperbound_lin[omega])
            else:
                model.addConstr(h_sum_temp >= gain*internal_lowerbound_lin[omega])

        if minmax_option == 'try_h_zero_count':
            model.setObjective(0, GRB.MINIMIZE)
            if h_zero_count == None:
                raise TypeError("Gurobi: h_zero_count in Barebone cant be empty when try_h_zero_count is chosen")

            h_zero = [model.addVar(vtype=GRB.BINARY, name=f'h_zero_{a}') for a in range(half_order + 1)]
            h_zero_sum = 0
            for m in range(half_order + 1):
                for w in range(self.wordlength):
                    model.addGenConstrIndicator(h_zero[m], True, h[m][w] == 0)
                h_zero_sum += h_zero[m]
            model.addConstr(h_zero_sum >= h_zero_count)
        else:
            model.setObjective(0, GRB.MINIMIZE)



            
        
        print("Gurobi: Barebone running")
        model.optimize()

        satisfiability = 'unsat'

        if model.status == GRB.OPTIMAL:
            satisfiability = 'sat'

            for m in range(half_order + 1):
                fir_coef = 0
                for w in range(self.wordlength):
                    # Evaluate the boolean value from the model
                    bool_value = h[m][w].X
                    
                    # Convert boolean to integer (0 or 1) and calculate the term
                    if w==self.wordlength-1:
                        fir_coef += -2**(w-self.fracW)  * (1 if bool_value else 0)
                    elif w < self.fracW:                   
                        fir_coef += 2**(-1*(self.fracW-w)) * (1 if bool_value else 0)
                    else:
                        fir_coef += 2**(w-self.fracW) * (1 if bool_value else 0)
                
                self.h_res.append(fir_coef)
            # print("FIR Coeffs calculated: ",self.h_res)

            self.gain_res=gain.x
            #print("gain Coeffs: ", self.gain_res)
            if minmax_option == 'try_h_zero_count':
                target_result.update({
                        'satisfiability' : satisfiability,
                        'h_res' : self.h_res,
                        'gain_res' : self.gain_res,
                    })
                
            

                      
        else:
            print("Gurobi: Unsatisfiable")
            target_result.update({
                    'satisfiability' : satisfiability,
                })
            
        model.terminate()  # Dispose of the model
        del model

        return target_result
        

    def run_barebone_real(self,thread, minmax_option, time_limit = None ,h_zero_count = None, h_target = None):
        self.h_res = []
        self.gain_res = []

        self.order_current = int(self.order)
        half_order = (self.order_current // 2) if self.filter_type == 0 or self.filter_type == 2 else (self.order_current // 2) - 1
        
        print("Gurobi solver called")
        sf = SolverFunc(self.get_solver_func_dict())

         # linearize the bounds
        internal_upperbound_lin = [math.floor((f)*(10**self.coef_accuracy)) if not np.isnan(f) else np.nan for f in self.upperbound_lin]
        internal_lowerbound_lin = [math.ceil((f)*(10**self.coef_accuracy)) if not np.isnan(f) else np.nan for f in self.lowerbound_lin]
        internal_ignore_lowerbound = self.ignore_lowerbound*(10**self.coef_accuracy)

        # print("filter order:", self.order_current)
        # print(" filter_type:", self.filter_type)
        # print("freqx_axis:", self.freqx_axis)
        # print("ignore lower than:", internal_ignore_lowerbound)
        
        # print(f"lower {internal_lowerbound_lin}")
        # print(f"upper {internal_upperbound_lin}")
        # print(f"coef_accuracy {self.coef_accuracy}")
        # print(f"gain_upperbound {self.gain_upperbound}")
        # print(f"gain_lowerbound {self.gain_lowerbound}")

        
        model = gp.Model(f"presolve_model_{minmax_option}")
        model.setParam('Threads', thread)
        model.setParam('OutputFlag', 0)
        if time_limit != None:
            model.setParam('TimeLimit', time_limit)


        h_upperbound = ((2**(self.intW-1))-1)+(1-2**-self.fracW)
        h_lowerbound = -2**(self.intW-1)

        h = [model.addVar(vtype=GRB.CONTINUOUS,lb=h_lowerbound, ub=h_upperbound, name=f'h_{a}') for a in range(half_order + 1)]
        gain = model.addVar(vtype=GRB.CONTINUOUS, lb=self.gain_lowerbound, ub=self.gain_upperbound, name="gain")

        for omega in range(len(self.freqx_axis)):
            if np.isnan(internal_lowerbound_lin[omega]):
                continue

            h_sum_temp = 0

            for m in range(half_order+1):
                cm = sf.cm_handler(m, self.freqx_axis[omega])
                cm_word_prod= int(cm*(10** self.coef_accuracy))
                h_sum_temp += h[m]*cm_word_prod

            model.update()
            # print(f"sum temp is{h_sum_temp}")
            model.addConstr(h_sum_temp <= gain*internal_upperbound_lin[omega])
            
            
            if internal_lowerbound_lin[omega] < internal_ignore_lowerbound:
                model.addConstr(h_sum_temp >= gain*-internal_upperbound_lin[omega])
            else:
                model.addConstr(h_sum_temp >= gain*internal_lowerbound_lin[omega])
        
        print(f"Gurobi barebone_real: {minmax_option}")

        if minmax_option == None:
            model.setObjective(0, GRB.MAXIMIZE)
                
        if minmax_option == 'find_max_zero':
            h_zero = [model.addVar(vtype=GRB.BINARY, name=f'h_zero_{a}') for a in range(half_order + 1)]
            h_zero_sum = 0
            for m in range(half_order + 1):
                model.addGenConstrIndicator(h_zero[m], True, h[m] == 0)
                h_zero_sum += h_zero[m]
            model.setObjective(h_zero_sum, GRB.MAXIMIZE)

        elif minmax_option == 'find_min_gain':
            if h_zero_count == None:
                raise TypeError("Gurobi barebone_real: h_zero_count cant be empty when find_min_gain is chosen")

            h_zero = [model.addVar(vtype=GRB.BINARY, name=f'h_zero_{a}') for a in range(half_order + 1)]
            h_zero_sum = 0
            for m in range(half_order + 1):
                model.addGenConstrIndicator(h_zero[m], True, h[m] == 0)
                h_zero_sum += h_zero[m]
            model.addConstr(h_zero_sum >= h_zero_count)
            
            model.setObjective(gain, GRB.MINIMIZE)


        elif minmax_option == 'maximize_h' or minmax_option == 'minimize_h':
            if h_target == None:
                raise TypeError("Gurobi barebone_real: h_target cant be empty when maximize_h/minimize_h is chosen")
            
            if h_zero_count == None:
                raise TypeError("Gurobi barebone_real: h_zero_count cant be empty when find_min_gain is chosen")

            h_zero = [model.addVar(vtype=GRB.BINARY, name=f'h_zero_{a}') for a in range(half_order + 1)]
            h_zero_sum = 0
            for m in range(half_order + 1):
                model.addGenConstrIndicator(h_zero[m], True, h[m] == 0)
                h_zero_sum += h_zero[m]
            model.addConstr(h_zero_sum >= h_zero_count)

            if minmax_option == 'maximize_h':
                model.setObjective(h[h_target], GRB.MAXIMIZE)

            elif minmax_option == 'minimize_h':
                model.setObjective(h[h_target], GRB.MINIMIZE)


        print("Gurobi: MinMax running")
        start_time=time.time()
        model.optimize()
        satisfiability = 'unsat'

        if model.status == GRB.OPTIMAL:
            satisfiability = 'sat'
            end_time = time.time()

            for m in range(half_order + 1):
                h_value = h[m].X
                self.h_res.append(h_value)
            # print("FIR Coeffs calculated: ",self.h_res)

            self.gain_res = gain.x
            # print("gain Coeffs: ", self.gain_res)

    
                      
        else:
            print("Gurobi: Unsatisfiable")
            end_time = time.time()
            

        model.dispose()  # Dispose of the model
        del model

        duration = end_time - start_time
        return satisfiability, self.h_res ,duration


    def plot_result(self, result_coef, original_freqx_axis, original_upperbound_lin, original_lowerbound_lin):
        # print("Result plotter called with higher accuracy check")
        
        # Construct fir_coefficients from h_res
        fir_coefficients = np.concatenate((result_coef[::-1], result_coef[1:]))

        # print(fir_coefficients)
        # print("FIR Coefficients in higher accuracy test", fir_coefficients)

        # Compute the FFT of the coefficients at higher resolution
        N = len(original_freqx_axis)*2  # Use the original frequency resolution length for FFT
        frequency_response = np.fft.fft(fir_coefficients, N)
        frequencies = np.fft.fftfreq(N, d=1.0)[:N//2]  # Extract positive frequencies up to Nyquist

        # Compute the magnitude response for positive frequencies
        magnitude_response = np.abs(frequency_response)[:N//2]
        
        # Normalize frequencies to range from 0 to 1 for plotting
        omega = frequencies * 2 * np.pi
        normalized_omega = np.linspace(0, 1, len(magnitude_response))

        # Initialize leak detection
        leaks = []
        leaks_mag = []
        continous_leak_count = 0

        # Check for leaks by comparing the FFT result with the 10x accuracy bounds
        for i, mag in enumerate(magnitude_response):
            if mag > original_upperbound_lin[i] + 0.005: 
                leaks.append((i, mag))  # Collect the leak points
                leaks_mag.append(mag-original_upperbound_lin[i])
                if continous_flag == False:
                    continous_leak_count +=1
                continous_flag = True
            elif mag < original_lowerbound_lin[i] - 0.005:
                leaks.append((i, mag))  # Collect the leak points
                leaks_mag.append(np.abs(mag-original_lowerbound_lin[i]))
                if continous_flag == False:
                    continous_leak_count +=1
                continous_flag = True
            else: continous_flag = False

        # print(f"len(magnitude_response) {len(magnitude_response)}")
        # print(f"len(original_freqx_axis) {len(original_freqx_axis)}")
        # print(f"leaks {leaks}")

        # Plot the input bounds (using the original bounds, which are at higher accuracy)
        self.ax1.scatter(self.freqx_axis, self.upperbound_lin, color='r', s=20, picker=5, label="Upper Bound")
        self.ax1.scatter(self.freqx_axis, self.lowerbound_lin, color='b', s=20, picker=5, label="Lower Bound")

        # Plot the higher accuracy bounds (for validation)
        self.ax2.scatter(original_freqx_axis, original_upperbound_lin, color='r', s=20, picker=5, label="Upper Bound (10x Accuracy)")
        self.ax2.scatter(original_freqx_axis, original_lowerbound_lin, color='b', s=20, picker=5, label="Lower Bound (10x Accuracy)")

        # Plot the magnitude response from the calculated coefficients
        self.ax1.scatter(normalized_omega, magnitude_response, color='y', label="Magnitude Response", s=10, picker=5)
        self.ax2.scatter(normalized_omega, magnitude_response, color='y', label="Magnitude Response", s=10, picker=5)

        leaks_mag_avg = 0

        # Mark the leaks on the plot
        if leaks:
            leak_indices, leak_values = zip(*leaks)
            leaks_mag = [float(x) for x in leaks_mag]
            leaks_mag_avg = sum(leaks_mag)/len(leaks_mag)

            leak_freqs = [normalized_omega[i] for i in leak_indices]
            self.ax1.scatter(leak_freqs, leak_values, color='black', s=4, label="Leak Points", zorder=5)
            # self.ax2.scatter(leak_freqs, leak_values, color='black', s=4, label="Leak Points", zorder=5)

        self.ax1.set_ylim([-10, 10])
        self.ax2.set_ylim([-10, 10])

        if self.app:
            self.app.canvas.draw()

        # Return leaks for further analysis if needed
        return leaks,leaks_mag_avg,continous_leak_count


    def plot_validation(self):
        print("Validation plotter called")
        half_order = (self.order_current // 2)
        sf = SolverFunc(self.get_solver_func_dict())
        # Array to store the results of the frequency response computation
        computed_frequency_response = []
        
        # Recompute the frequency response for each frequency point
        for i in range(len(self.freqx_axis)):
            omega = self.freqx_axis[i]
            term_sum_exprs = 0
            
            # Compute the sum of products of coefficients and the cosine/sine terms
            for j in range(half_order+1):
                cm_const = sf.cm_handler(j, omega)
                term_sum_exprs += self.h_res[j] * cm_const
            
            # Append the computed sum expression to the frequency response list
            computed_frequency_response.append(np.abs(term_sum_exprs))
        
        # Normalize frequencies to range from 0 to 1 for plotting purposes

        # Plot the computed frequency response
        self.ax1.plot([x/1 for x in self.freqx_axis], computed_frequency_response, color='green', label='Computed Frequency Response')
        # self.ax2.plot([x/1 for x in self.freqx_axis], computed_frequency_response, color='green', label='Computed Frequency Response')

        self.ax2.set_ylim(-10,10)


        if self.app:
            self.app.canvas.draw()


def generate_freq_bounds(space,multiplier_to_test ,order_current):
   #random bounds generator
    random.seed(it)
    if order_current > 50:
        lower_cutoff = random.choice([0.25, 0.3])
        lower_middle_cutoff = random.choice([0.35, 0.4])

        upper_middle_cutoff = random.choice([ 0.7, 0.75])
        upper_cutoff = random.choice([ 0.8, 0.85, 0.9])
    else:
        lower_cutoff = random.choice([0.2, 0.3])
        lower_middle_cutoff = random.choice([0.45, 0.5])

        upper_middle_cutoff = random.choice([ 0.75, 0.8])
        upper_cutoff = random.choice([ 0.9, 0.95])

    lower_half_point = int(lower_cutoff * space)
    lower_middle_half_point = int(lower_middle_cutoff * space)
    upper_middle_half_point = int(upper_middle_cutoff * space)
    upper_half_point = int(upper_cutoff * space)
   
    
    end_point = space
    freqx_axis = np.linspace(0, 1, space)
    freq_upper = np.full(space, np.nan)
    freq_lower = np.full(space, np.nan)
    passband_upperbound = random.choice([0 , 0.2])
    passband_lowerbound = random.choice([0 , -0.2])
    stopband_upperbound = random.choice([-20, -30])
    stopband_upperbound = stopband_upperbound * math.ceil(order_current//10)

    if stopband_upperbound < -40 :
        stopband_upperbound = -40

    stopband_lowerbound = -1000
    
    freq_upper[0:lower_half_point] = stopband_upperbound
    freq_lower[0:lower_half_point] = stopband_lowerbound

    #transition
    freq_upper[lower_half_point:lower_middle_half_point] = passband_upperbound
    freq_lower[lower_half_point:lower_middle_half_point] = stopband_lowerbound

    freq_upper[lower_middle_half_point:upper_middle_half_point] = passband_upperbound
    freq_lower[lower_middle_half_point:upper_middle_half_point] = passband_lowerbound
    
    #transition
    freq_upper[upper_middle_half_point:upper_half_point] = passband_upperbound
    freq_lower[upper_middle_half_point:upper_half_point] = stopband_lowerbound

    freq_upper[upper_half_point:end_point] = stopband_upperbound
    freq_lower[upper_half_point:end_point] = stopband_lowerbound

    space_to_test = space * multiplier_to_test
    original_end_point = space_to_test
    original_freqx_axis = np.linspace(0, 1, space_to_test)
    original_freq_upper = np.full(space_to_test, np.nan)
    original_freq_lower = np.full(space_to_test, np.nan)

    
    original_lower_half_point = np.abs(original_freqx_axis - ((lower_half_point-1)/space)).argmin()
    original_lower_middle_half_point = np.abs(original_freqx_axis - ((lower_middle_half_point+1)/space)).argmin()
    original_upper_middle_half_point = np.abs(original_freqx_axis - ((upper_middle_half_point-1)/space)).argmin()
    original_upper_half_point = np.abs(original_freqx_axis - ((upper_half_point+1)/space)).argmin()
   
    original_freq_upper[0:original_lower_half_point] = stopband_upperbound
    original_freq_lower[0:original_lower_half_point] = stopband_lowerbound

    #transition
    original_freq_upper[original_lower_half_point:original_lower_middle_half_point] = passband_upperbound
    original_freq_lower[original_lower_half_point:original_lower_middle_half_point] = stopband_lowerbound

    original_freq_upper[original_lower_middle_half_point:original_upper_middle_half_point] = passband_upperbound
    original_freq_lower[original_lower_middle_half_point:original_upper_middle_half_point] = passband_lowerbound
    
    #transition
    original_freq_upper[original_upper_middle_half_point:original_upper_half_point] = passband_upperbound
    original_freq_lower[original_upper_middle_half_point:original_upper_half_point] = stopband_lowerbound

    original_freq_upper[original_upper_half_point:original_end_point] = stopband_upperbound
    original_freq_lower[original_upper_half_point:original_end_point] = stopband_lowerbound



     #beyond this bound lowerbound will be ignored
    ignore_lowerbound = -40

    
    #linearize the bound
    upperbound_lin = [10 ** (f / 20) if not np.isnan(f) else np.nan for f in freq_upper]
    lowerbound_lin = [10 ** (f / 20) if not np.isnan(f) else np.nan for f in freq_lower]

    original_upperbound_lin = [10 ** (f / 20) if not np.isnan(f) else np.nan for f in original_freq_upper]
    original_lowerbound_lin = [10 ** (f / 20) if not np.isnan(f) else np.nan for f in original_freq_lower]


    ignore_lowerbound_lin = 10 ** (ignore_lowerbound / 20)

    return freqx_axis,upperbound_lin,lowerbound_lin,ignore_lowerbound_lin,original_freqx_axis,original_upperbound_lin,original_lowerbound_lin , original_lower_half_point, original_lower_middle_half_point,original_upper_middle_half_point,original_upper_half_point

global it
it = 1

if __name__ == "__main__":
    
    filter_type = 0
    wordlength = 16
    
    ignore_lowerbound_lin = -20

    with open("accuracy_multipier_test.txt", "w") as file:
        file.write("accuracy; order; sat; duration; leak; leak_magnitude; leak_position\n")

    # Test inputs from 1 to 20 accuracy multiplier
    for i in range(7,21):
        for j in range(3):
            for order_current in range(10,161,4):
                accuracy = i
                space = order_current * accuracy
                freqx_axis,upperbound_lin,lowerbound_lin,ignore_lowerbound_lin,original_freqx_axis,original_upperbound_lin,original_lowerbound_lin , original_lower_half_point, original_lower_middle_half_point,original_upper_middle_half_point,original_upper_half_point = generate_freq_bounds(space,100,order_current)

                #higher resolution list to test the result

                it +=1

                # Create FIRFilter instance
                fir_filter = FIRFilter(
                            filter_type, 
                            order_current, 
                            freqx_axis, 
                            upperbound_lin, 
                            lowerbound_lin, 
                            ignore_lowerbound_lin, 
                            0, 
                            wordlength, 
                            0,
                            0,
                            0,
                            1,
                            1,
                            7,
                            4
                            )
                
                # Run solver and plot result
                satisfiability, h_res ,duration = fir_filter.run_barebone_real(0,'None',600)

                leak_indices = []
                leak_values = []
                leaks_mag_avg = 0
                leaks_count = 0

                leaks,leaks_mag_avg, leaks_count= fir_filter.plot_result(h_res, original_freqx_axis, original_upperbound_lin, original_lowerbound_lin)
                
                if leaks:
                    leak_indices, leak_values = zip(*leaks)
                    # print(f"leaks_count {leaks_count}")
                    # print(f"leaks_mag {leaks_mag_avg}")

                leak_pos = []
                for leak in leak_indices:
                    if leak < original_lower_half_point or leak > original_upper_half_point:
                        if 'leak in stopband' not in leak_pos:
                            leak_pos.append('leak in stopband')
                    elif (leak > original_lower_half_point and leak < original_lower_middle_half_point) or (leak > original_upper_middle_half_point and leak < original_upper_half_point):
                        if 'leak in transition band' not in leak_pos:
                            leak_pos.append('leak in transition band')
                    elif leak > original_lower_middle_half_point and leak < original_upper_middle_half_point:
                        if 'leak in passband' not in leak_pos:
                            leak_pos.append('leak in passband')
                
                print(satisfiability)

                if  satisfiability == 'unsat':
                    leak_indices = []
                    leak_values = []
                    leak_pos = []
                    leaks_mag_avg = 0
                    leaks_count = 0
                

                
                # fir_filter.plot_validation()

                # uncomment to Show plot
                # plt.show()

                with open("accuracy_multipier_test.txt", "a") as file:
                    file.write(f"{accuracy}; {order_current}; {satisfiability}; {duration};{leaks_count};{leaks_mag_avg};{leak_pos}\n")
                    
                print("Test ", i, " is completed")

    print("Benchmark completed and results saved to accuracy_multipier_test.txt")
