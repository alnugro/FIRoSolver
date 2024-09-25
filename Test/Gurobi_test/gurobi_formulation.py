import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import time
import math



class SolverFunc():
    def __init__(self,filter_type, order):
        self.filter_type=filter_type
        self.half_order = (order//2)
        self.overflow_count = 0

    def db_to_linear(self,db_arr):
        # Create a mask for NaN values
        nan_mask = np.isnan(db_arr)

        # Apply the conversion to non-NaN values (magnitude)
        linear_array = np.zeros_like(db_arr)
        linear_array[~nan_mask] = 10 ** (db_arr[~nan_mask] / 20)

        # Preserve NaN values
        linear_array[nan_mask] = np.nan
        return linear_array
    
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
        
    def overflow_handler(self, input_value, upper_bound, lower_bound, literal):
        self.overflow_count+=1
        overflow_coef = []
        overflow_lit = []

        if input_value > upper_bound:
            while input_value > upper_bound:
                overflow_coef.append(upper_bound)
                overflow_lit.append(literal)
                input_value -= upper_bound
            overflow_coef.append(input_value)
            overflow_lit.append(literal)
        
        elif input_value < lower_bound:
            while input_value < lower_bound:
                overflow_coef.append(lower_bound)
                overflow_lit.append(literal)
                input_value -= lower_bound
            overflow_coef.append(input_value)
            overflow_lit.append(literal)
        
        else:
            overflow_coef.append(input_value)
            overflow_lit.append(literal)
            print("somehting weird happens on overflow handler")

        return [overflow_lit, overflow_coef]




class FIRFilter:
    def __init__(self, filter_type, order_upper, freqx_axis, freq_upper, freq_lower, ignore_lowerbound_lin, adder_count, wordlength, app=None):
        self.filter_type = filter_type
        self.order_upper = order_upper
        self.freqx_axis = freqx_axis
        self.freq_upper = freq_upper
        self.freq_lower = freq_lower
        self.h_res = []
        self.gain_res = 0

        self.wordlength = wordlength
        self.N = adder_count

        self.app = app
        self.fig, (self.ax1, self.ax2) = plt.subplots(2,1)
        self.freq_upper_lin=0
        self.freq_lower_lin=0

        self.coef_accuracy = 3
        self.ih = 2
        self.fh = self.wordlength - self.ih

        
        self.gain_wordlength=6 #9 bits wordlength for gain
        self.ig = 3
        self.fg =  self.gain_wordlength - self.ig

        self.gain_upperbound= 1.4
        self.gain_lowerbound= 1
        self.gain_bound_accuracy = 2 #2 floating points

        
        

        self.ignore_lowerbound_lin = ignore_lowerbound_lin*(10**self.coef_accuracy)*(2**self.fh)


    def run_barebone_real(self,thread, minmax_option, h_zero_count = None, h_target = None):
        self.h_res = []
        self.gain_res = []
        target_result = {}
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
        model.optimize()
        satisfiability = 'unsat'

        if model.status == GRB.OPTIMAL:
            satisfiability = 'sat'

            for m in range(half_order + 1):
                h_value = h[m].X
                self.h_res.append(h_value)
            # print("FIR Coeffs calculated: ",self.h_res)

            self.gain_res = gain.x
            # print("gain Coeffs: ", self.gain_res)

            if minmax_option == 'find_max_zero':
                #asign h_zero value
                h_zero_sum_res= 0
                for m in range(half_order + 1):
                    h_zero_sum_res += h_zero[m].X
                target_result.update({
                    'satisfiability' : satisfiability,
                    'h_res' : self.h_res,
                    'max_h_zero' : h_zero_sum_res
                })

            elif minmax_option == 'find_min_gain':
                target_result.update({
                    'satisfiability' : satisfiability,
                    'min_gain' : self.gain_res,
                })
            
            elif minmax_option == 'maximize_h' or minmax_option == 'minimize_h':
                target_result.update({
                    'satisfiability' : satisfiability,
                    'target_h_res' : h[h_target].X,
                })
                      
        else:
            print("Gurobi: Unsatisfiable")
            target_result.update({
                    'satisfiability' : satisfiability,
                })

        model.dispose()  # Dispose of the model
        del model

        print(target_result)

        return target_result


    def runsolver(self):
        self.order_current = int(self.order_upper)
        half_order = (self.order_current // 2)
        
        print("solver called")
        sf = SolverFunc(self.filter_type, self.order_current)

        print("filter order:", self.order_current)
        print("ignore lower than:", self.ignore_lowerbound_lin)
        # linearize the bounds
        self.freq_upper_lin = [int((sf.db_to_linear(f))*(10**self.coef_accuracy)*(2**self.fh)) if not np.isnan(sf.db_to_linear(f)) else np.nan for f in self.freq_upper]
        self.freq_lower_lin = [int((sf.db_to_linear(f))*(10**self.coef_accuracy)*(2**self.fh)) if not np.isnan(sf.db_to_linear(f)) else np.nan for f in self.freq_lower]

        model = gp.Model("FIRFilterOptimization")


        h = [[Bool(f'h{a}_{w}') for w in range(self.wordlength)] for a in range(half_order+1)]
        gain= [Bool(f'gain{g}') for g in range(self.gain_wordlength)]

        vars = [model.addVar(vtype=GRB.BINARY, name=f"x{i}") for i in range(num_vars)]



        gain_coeffs = []
        gain_literalls=[]


        #bounds the gain
        self.gain_upperbound_int = int(self.gain_upperbound*2**self.fg*(10**self.gain_bound_accuracy))
        self.gain_lowerbound_int = int(self.gain_lowerbound*2**self.fg*(10**self.gain_bound_accuracy))

        # print(self.gain_upperbound_int)
        # print(self.gain_lowerbound_int)

        

        for g in range(self.gain_wordlength):
            gain_coeffs.append((2**g)*(10**self.gain_bound_accuracy))
            gain_literalls.append(gain[g])

        pb_gain_pairs = [(gain_literalls[i],gain_coeffs[i]) for i in range(len(gain_literalls))]
            
        solver.add(PbLe(pb_gain_pairs, self.gain_upperbound_int))
        solver.add(PbGe(pb_gain_pairs, self.gain_lowerbound_int))
            

            
        filter_literals = []
        filter_coeffs = []
        gain_freq_upper_prod_coeffs = []
        gain_freq_lower_prod_coeffs = []

        filter_upper_pb_pairs = []
        filter_lower_pb_pairs = []

        filter_overflow_literalls=[]
        filter_overflow_coeffs = []

        gain_upper_overflow_literalls=[]
        gain_upper_overflow_coeffs = []

        gain_lower_overflow_literalls=[]
        gain_lower_overflow_coeffs = []

        gain_upper_literalls = []
        gain_lower_literalls = []


        max_positive_int_pbfunc = 2147483647
        max_negative_int_pbfunc = -2147483648



        for omega in range(len(self.freqx_axis)):
            if np.isnan(self.freq_lower_lin[omega]):
                continue

            #clearing each list like this make the programm run faster, instead of decalring new one each time
            gain_literalls.clear()
            filter_literals.clear()
            filter_coeffs.clear()

            gain_freq_upper_prod_coeffs.clear()
            gain_freq_lower_prod_coeffs.clear()

            filter_upper_pb_pairs.clear()
            filter_lower_pb_pairs.clear()

            filter_overflow_literalls.clear()
            filter_overflow_coeffs.clear()

            gain_upper_overflow_literalls.clear()
            gain_upper_overflow_coeffs.clear()

            gain_lower_overflow_literalls.clear()
            gain_lower_overflow_coeffs.clear()
            
            gain_upper_literalls.clear()
            gain_lower_literalls.clear()

            for m in range(half_order+1):
                cm = sf.cm_handler(m, self.freqx_axis[omega])
                for w in range(self.wordlength):
                    if w==self.wordlength-1:
                        cm_word_prod= int(cm*(10** self.coef_accuracy)*(-1*(2**w))*(2**self.fg))
                    else: cm_word_prod= int(cm*(10** self.coef_accuracy)*(2**w)*2**self.fg)

                    if cm_word_prod > max_positive_int_pbfunc or cm_word_prod < max_negative_int_pbfunc:
                        overflow = sf.overflow_handler(cm_word_prod,max_positive_int_pbfunc,max_negative_int_pbfunc,h[m][w])
                        filter_overflow_literalls.extend(overflow[0])
                        filter_overflow_coeffs.extend(overflow[1])
                        print("overflow happened in the product of cm: appended this to the sum coeff:", overflow[1], " with literall: ", overflow[0])
                    else:
                        filter_coeffs.append(cm_word_prod)
                        filter_literals.append(h[m][w])

            for g in range(self.gain_wordlength):
                gain_upper_prod = int(-1 * (2**g) * self.freq_upper_lin[omega])
                 

                if gain_upper_prod > max_positive_int_pbfunc or gain_upper_prod < max_negative_int_pbfunc:
                    overflow = sf.overflow_handler(gain_upper_prod,max_positive_int_pbfunc,max_negative_int_pbfunc,gain[g])
                    gain_upper_overflow_literalls.extend(overflow[0])
                    gain_upper_overflow_coeffs.extend(overflow[1])
                    print("overflow happened in the gain upper product: appended this to the sum coeff:", overflow[1], " with literall: ", overflow[0])
                else:
                    gain_freq_upper_prod_coeffs.append(gain_upper_prod)
                    gain_upper_literalls.append(gain[g])

                if self.freq_lower_lin[omega] < self.ignore_lowerbound_lin:
                    gain_lower_prod=int((2**g) * self.freq_upper_lin[omega])
                    if gain_lower_prod > max_positive_int_pbfunc or gain_lower_prod < max_negative_int_pbfunc:
                        overflow = sf.overflow_handler(gain_lower_prod,max_positive_int_pbfunc,max_negative_int_pbfunc,gain[g])
                        gain_lower_overflow_literalls.extend(overflow[0])
                        gain_lower_overflow_coeffs.extend(overflow[1])
                        print("overflow happened in the gain lower product: appended this to the sum coeff:", overflow[1], " with literall: ", overflow[0])
                    else:
                        gain_freq_lower_prod_coeffs.append(gain_lower_prod)
                        gain_lower_literalls.append(gain[g])
                        print("ignored ",self.freq_lower_lin[omega], " in frequency = ", self.freqx_axis[omega])
                else:
                    gain_lower_prod=int(-1 *(2**g) * self.freq_lower_lin[omega])
                    if gain_lower_prod > max_positive_int_pbfunc or gain_lower_prod < max_negative_int_pbfunc:
                        overflow = sf.overflow_handler(gain_lower_prod,max_positive_int_pbfunc,max_negative_int_pbfunc,gain[g])
                        gain_lower_overflow_literalls.extend(overflow[0])
                        gain_lower_overflow_coeffs.extend(overflow[1])
                        print("overflow happened in the gain lower product: appended this to the sum coeff:", overflow[1], " with literall: ", overflow[0])
                    else:
                        gain_freq_lower_prod_coeffs.append(gain_lower_prod)
                        gain_lower_literalls.append(gain[g])

            filter_upper_pb_coeffs=filter_coeffs+gain_freq_upper_prod_coeffs+filter_overflow_coeffs+gain_upper_overflow_coeffs
            filter_upper_pb_literalls=filter_literals+gain_upper_literalls+filter_overflow_literalls+gain_upper_overflow_literalls

            #print("coeffs: ",filter_upper_pb_coeffs)
            #print("lit: ",filter_upper_pb_literalls)

            if len(filter_upper_pb_coeffs) != len(filter_upper_pb_literalls):
                raise("sumtin wong with upper filter pb")
            
            # else: print("filter upperbound length is validated")

            

            #z3 only take pairs
            filter_upper_pb_pairs = [(filter_upper_pb_literalls[i],filter_upper_pb_coeffs[i],) for i in range(len(filter_upper_pb_literalls))]
            solver.add(PbLe(filter_upper_pb_pairs, 0))

            
           
            filter_lower_pb_coeffs=filter_coeffs+gain_freq_lower_prod_coeffs+filter_overflow_coeffs+gain_lower_overflow_coeffs
            filter_lower_pb_literalls=filter_literals+gain_lower_literalls+filter_overflow_literalls+gain_lower_overflow_literalls


            # print("coeffs: ",filter_lower_pb_coeffs)
            # print("lit: ",filter_lower_pb_literalls)

            if len(filter_lower_pb_coeffs) != len(filter_lower_pb_literalls):
                raise("sumtin wong with upper filter pb")
            
            # else: print("filter lowerbound length is validated")

            
            filter_lower_pb_pairs = [(filter_lower_pb_literalls[i],filter_lower_pb_coeffs[i]) for i in range(len(filter_lower_pb_literalls))]
            
            #z3 only take pairs
            solver.add(PbGe(filter_lower_pb_pairs, 0))

            #end omega loop


        #bitshift sat starts here
        
        #input multiplexer
        c=[[Bool(f'c{i}{w}') for w in range(self.wordlength)] for i in range(self.N+1)]
        l=[[Bool(f'l{i}{w}') for w in range(self.wordlength)] for i in range(1, self.N+1)]
        r=[[Bool(f'r{i}{w}') for w in range(self.wordlength)] for i in range(1, self.N+1)]


        alpha = [[Bool(f'alpha{i}{a}') for a in range(i)] for i in range(1, self.N+1)]
        beta =[[ Bool(f'Beta{i}{a}') for a in range(i)] for i in range(1, self.N+1)] 

        #c0,w is always 0 except w=0
        for w in range(1,self.wordlength):
            solver.add(Not(c[0][w]))

        solver.add(c[0][0])


        #input multiplexer
        for i in range(1, self.N+1):
            alpha_sum = []
            beta_sum = []
            for a in range(i):
                for word in range(self.wordlength):
                    clause1_1 = Or(Not(alpha[i-1][a]), Not(c[a][word]), l[i-1][word])
                    clause1_2 = Or(Not(alpha[i-1][a]), c[a][word], Not(l[i-1][word]))
                    solver.add(And(clause1_1, clause1_2))

                    clause2_1 = Or(Not(beta[i-1][a]), Not(c[a][word]), r[i-1][word])
                    clause2_2 = Or(Not(beta[i-1][a]), c[a][word], Not(r[i-1][word]))
                    solver.add(And(clause2_1, clause2_2))

                #make a pair for pbeq with a weight of 1 for later
                alpha_sum.append((alpha[i-1][a], 1))
                beta_sum.append((beta[i-1][a], 1))

           
            solver.add(PbEq(alpha_sum,1))
            solver.add(PbEq(beta_sum,1))

        #Left Shifter
        #k is the shift selector
        gamma = [[Bool(f'gamma{i}{k}') for k in range(self.wordlength-1)] for i in range(1, self.N+1)]
        s     = [[Bool(f's{i}{w}') for w in range(self.wordlength)] for i in range(1, self.N+1)]


        for i in range(1, self.N+1):
            gamma_sum = []
            for k in range(self.wordlength-1):
                for j in range(self.wordlength-1-k):
                    clause3_1 = Or(Not(gamma[i-1][k]),Not(l[i-1][j]),s[i-1][j+k])
                    clause3_2 = Or(Not(gamma[i-1][k]),l[i-1][j],Not(s[i-1][j+k]))
                    solver.add(And(clause3_1, clause3_2))

                gamma_sum.append((gamma[i-1][k], 1))
            
            solver.add(PbEq(gamma_sum,1))


            for kf in range(1,self.wordlength-1):
                for b in range(kf):
                    clause4 = Or(Not(gamma[i-1][kf]),Not(s[i-1][b]))
                    clause5 = Or(Not(gamma[i-1][kf]), Not(l[i-1][self.wordlength-1]), l[i-1][self.wordlength-2-b])
                    clause6 = Or(Not(gamma[i-1][kf]), l[i-1][self.wordlength-1], Not(l[i-1][self.wordlength-2-b]))
                    solver.add(clause4)
                    solver.add(clause5)
                    solver.add(clause6)

            clause7_1= Or(Not(l[i-1][self.wordlength-1]), s[i-1][self.wordlength-1])
            clause7_2= Or(l[i-1][self.wordlength-1], Not(s[i-1][self.wordlength-1]))
            solver.add(And(clause7_1, clause7_2))


        delta = [Bool(f'delta{i}') for i in range(1, self.N+1)]
        u     = [[Bool(f'u{i}{w}') for w in range(self.wordlength)] for i in range(1, self.N+1)]
        x     = [[Bool(f'x{i}{w}') for w in range(self.wordlength)] for i in range(1, self.N+1)]

   
    
        #delta selector
        for i in range(1, self.N+1):
            for word in range(self.wordlength):
                clause8_1 = Or(Not(delta[i-1]),Not(s[i-1][word]),x[i-1][word])
                clause8_2 = Or(Not(delta[i-1]),s[i-1][word],Not(x[i-1][word]))
                solver.add(And(clause8_1, clause8_2))
                
                clause9_1 = Or(Not(delta[i-1]),Not(r[i-1][word]),u[i-1][word])
                clause9_2 = Or(Not(delta[i-1]),r[i-1][word],Not(u[i-1][word]))
                solver.add(And(clause9_1, clause9_2))

                clause10_1 = Or(delta[i-1],Not(s[i-1][word]),u[i-1][word])
                clause10_2 = Or(delta[i-1],s[i-1][word],Not(u[i-1][word]))
                solver.add(And(clause10_1, clause10_2))

                clause11_1 = Or(delta[i-1],Not(r[i-1][word]),x[i-1][word])
                clause11_2 = Or(delta[i-1],r[i-1][word],Not(x[i-1][word]))
                solver.add(And(clause11_1, clause11_2))

                solver.add(Or(delta[i-1], Not(delta[i-1])))
                
        epsilon = [Bool(f'epsilon{i}') for i in range(1, self.N+1)]
        y     = [[Bool(f'y{i}{w}') for w in range(self.wordlength)] for i in range(1, self.N+1)]


        #xor
        for i in range(1, self.N+1):
            for word in range(self.wordlength):
                clause12 = Or(u[i-1][word], epsilon[i-1], Not(y[i-1][word]))
                clause13 = Or(u[i-1][word], Not(epsilon[i-1]), y[i-1][word])
                clause14 = Or(Not(u[i-1][word]), epsilon[i-1], y[i-1][word])
                clause15 = Or(Not(u[i-1][word]), Not(epsilon[i-1]), Not(y[i-1][word]))
                solver.add(clause12)
                solver.add(clause13)
                solver.add(clause14)
                solver.add(clause15)

        
        
        

        #ripple carry
        z     = [[Bool(f'z{i}{w}') for w in range(self.wordlength)] for i in range(1, self.N+1)]
        cout  = [[Bool(f'cout{i}{w}') for w in range(self.wordlength)] for i in range(1, self.N+1)]

        
        for i in range(1, self.N+1):
            # Clauses for sum = a ⊕ b ⊕ cin at 0
            clause16 = Or(x[i-1][0], y[i-1][0], epsilon[i-1], Not(z[i-1][0]))
            clause17 = Or(x[i-1][0], y[i-1][0], Not(epsilon[i-1]), z[i-1][0])
            clause18 = Or(x[i-1][0], Not(y[i-1][0]), epsilon[i-1], z[i-1][0])
            clause19 = Or(Not(x[i-1][0]), y[i-1][0], epsilon[i-1], z[i-1][0])
            clause20 = Or(Not(x[i-1][0]), Not(y[i-1][0]), Not(epsilon[i-1]), z[i-1][0])
            clause21 = Or(Not(x[i-1][0]), Not(y[i-1][0]), epsilon[i-1], Not(z[i-1][0]))
            clause22 = Or(Not(x[i-1][0]), y[i-1][0], Not(epsilon[i-1]), Not(z[i-1][0]))
            clause23 = Or(x[i-1][0], Not(y[i-1][0]), Not(epsilon[i-1]), Not(z[i-1][0]))

            solver.add(clause16)
            solver.add(clause17)
            solver.add(clause18)
            solver.add(clause19)
            solver.add(clause20)
            solver.add(clause21)
            solver.add(clause22)
            solver.add(clause23)

            # Clauses for cout = (a AND b) OR (cin AND (a ⊕ b))
            clause24 = Or(Not(x[i-1][0]), Not(y[i-1][0]), cout[i-1][0])
            clause25 = Or(x[i-1][0], y[i-1][0], Not(cout[i-1][0]))
            clause26 = Or(Not(x[i-1][0]), Not(epsilon[i-1]), cout[i-1][0])
            clause27 = Or(x[i-1][0], epsilon[i-1], Not(cout[i-1][0]))
            clause28 = Or(Not(y[i-1][0]), Not(epsilon[i-1]), cout[i-1][0])
            clause29 = Or(y[i-1][0], epsilon[i-1], Not(cout[i-1][0]))

            solver.add(clause24)
            solver.add(clause25)
            solver.add(clause26)
            solver.add(clause27)
            solver.add(clause28)
            solver.add(clause29)

            for kf in range(1, self.wordlength):
                # Clauses for sum = a ⊕ b ⊕ cin at kf
                clause30 = Or(x[i-1][kf], y[i-1][kf], cout[i-1][kf-1], Not(z[i-1][kf]))
                clause31 = Or(x[i-1][kf], y[i-1][kf], Not(cout[i-1][kf-1]), z[i-1][kf])
                clause32 = Or(x[i-1][kf], Not(y[i-1][kf]), cout[i-1][kf-1], z[i-1][kf])
                clause33 = Or(Not(x[i-1][kf]), y[i-1][kf], cout[i-1][kf-1], z[i-1][kf])
                clause34 = Or(Not(x[i-1][kf]), Not(y[i-1][kf]), Not(cout[i-1][kf-1]), z[i-1][kf])
                clause35 = Or(Not(x[i-1][kf]), Not(y[i-1][kf]), cout[i-1][kf-1], Not(z[i-1][kf]))
                clause36 = Or(Not(x[i-1][kf]), y[i-1][kf], Not(cout[i-1][kf-1]), Not(z[i-1][kf]))
                clause37 = Or(x[i-1][kf], Not(y[i-1][kf]), Not(cout[i-1][kf-1]), Not(z[i-1][kf]))

                solver.add(clause30)
                solver.add(clause31)
                solver.add(clause32)
                solver.add(clause33)
                solver.add(clause34)
                solver.add(clause35)
                solver.add(clause36)
                solver.add(clause37)

                # Clauses for cout = (a AND b) OR (cin AND (a ⊕ b)) at kf
                clause38 = Or(Not(x[i-1][kf]), Not(y[i-1][kf]), cout[i-1][kf])
                clause39 = Or(x[i-1][kf], y[i-1][kf], Not(cout[i-1][kf]))
                clause40 = Or(Not(x[i-1][kf]), Not(cout[i-1][kf-1]), cout[i-1][kf])
                clause41 = Or(x[i-1][kf], cout[i-1][kf-1], Not(cout[i-1][kf]))
                clause42 = Or(Not(y[i-1][kf]), Not(cout[i-1][kf-1]), cout[i-1][kf])
                clause43 = Or(y[i-1][kf], cout[i-1][kf-1], Not(cout[i-1][kf]))

                solver.add(clause38)
                solver.add(clause39)
                solver.add(clause40)
                solver.add(clause41)
                solver.add(clause42)
                solver.add(clause43)

            clause44 = Or(epsilon[i-1], x[i-1][self.wordlength-1], u[i-1][self.wordlength-1], Not(z[i-1][self.wordlength-1]))
            clause45 = Or(epsilon[i-1], Not(x[i-1][self.wordlength-1]), Not(u[i-1][self.wordlength-1]), z[i-1][self.wordlength-1])
            clause46 = Or(Not(epsilon[i-1]), x[i-1][self.wordlength-1], Not(u[i-1][self.wordlength-1]), Not(z[i-1][self.wordlength-1]))
            clause47 = Or(Not(epsilon[i-1]), Not(x[i-1][self.wordlength-1]), u[i-1][self.wordlength-1], z[i-1][self.wordlength-1])

            solver.add(clause44)
            solver.add(clause45)
            solver.add(clause46)
            solver.add(clause47)


        #right shift
        zeta = [[Bool(f'zeta{i}{k}') for k in range(self.wordlength-1)] for i in range(1, self.N+1)]



        for i in range(1, self.N+1):
            zeta_sum = []
            for k in range(self.wordlength-1):
                for j in range(self.wordlength-1-k):
                    clause48_1 = Or(Not(zeta[i-1][k]),Not(z[i-1][j+k]),c[i][j])
                    clause48_2 = Or(Not(zeta[i-1][k]),z[i-1][j+k],Not(c[i][j]))
                    solver.add(And(clause48_1, clause48_2))

                zeta_sum.append((zeta[i-1][k], 1))
            
            solver.add(PbEq(zeta_sum,1))


            for kf in range(1,self.wordlength-1):
                for b in range(kf):
                    clause49_1 = Or(Not(zeta[i-1][kf]), Not(z[i-1][self.wordlength-1]), c[i][self.wordlength-2-b])
                    clause49_2 = Or(Not(zeta[i-1][kf]), z[i-1][self.wordlength-1], Not(c[i][self.wordlength-2-b]))
                    solver.add(clause49_1)
                    solver.add(clause49_2)

                    clause50 = Or(Not(zeta[i-1][kf]), Not(z[i-1][b]))
                    solver.add(clause50)
            
            clause51_1 = Or(Not(z[i-1][self.wordlength-1]), c[i][self.wordlength-1])
            clause51_2 = Or(z[i-1][self.wordlength-1], Not(c[i][self.wordlength-1]))
            solver.add(And(clause51_1, clause51_2))

      
            #bound ci,0 to be odd number 
            solver.add(c[i][0])

        #set connected coefficient
        connected_coefficient = half_order+1

        #solver connection
        # h = [[Bool(f'h{m}_{w}') for w in range(self.wordlength)] for m in range(half_order+1)]
        h0 = [Bool(f'h0{m}') for m in range(half_order+1)]
        t = [[Bool(f't{i}_{m}') for m in range(half_order+1)] for i in range(1, self.N+1)]
        e = [Bool(f'e{m}') for m in range(half_order+1)]

        e_sum = []
        for m in range(half_order+1):
            h_or_clause=[]
            t_or_clauses=[]
            

            for w in range(self.wordlength):
                h_or_clause.append(h[m][w])
            h_or_clause.append(h0[m])
            solver.add(Or(h_or_clause))

            for i in range(1, self.N+1):
                for word in range(self.wordlength):
                    clause52_1=Or(Not(t[i-1][m]), Not(e[m]), Not(c[i][word]),h[m][word])
                    clause52_2=Or(Not(t[i-1][m]), Not(e[m]), c[i][word],Not(h[m][word]))
                    solver.add(And(clause52_1, clause52_2))

                t_or_clauses.append(t[i-1][m])
            solver.add(Or(t_or_clauses))

            e_sum.append((e[m],1))
        
        solver.add(PbEq(e_sum,connected_coefficient))

        
        start_time=time.time()

        print("solver runing")  


        # print(filter_coeffs)
        # print(filter_literals)

        if solver.check() == sat:
            print("solver sat")
            model = solver.model()
            print(model)
            end_time = time.time()
            
            for m in range(half_order + 1):
                fir_coef = 0
                for w in range(self.wordlength):
                    # Evaluate the boolean value from the model
                    bool_value = model.eval(h[m][w], model_completion=True)
                    print(f"h{m}_{w} = ",bool_value)

                    # Convert boolean to integer (0 or 1) and calculate the term
                    if w==self.wordlength-1:
                        fir_coef += -2**(w-self.fh)  * (1 if bool_value else 0)
                    elif w < self.fh:                   
                        fir_coef += 2**(-1*(self.fh-w)) * (1 if bool_value else 0)
                    else:
                        fir_coef += 2**(w-self.fh) * (1 if bool_value else 0)
                
                self.h_res.append(fir_coef)
            print("FIR Coeffs calculated: ",self.h_res)
            
            gain_coef=0
            for g in range(self.gain_wordlength):
                # Evaluate the boolean value from the model
                bool_value = model.eval(gain[g], model_completion=True)
                print(f"gain{g}= ",bool_value)



                # Convert boolean to integer (0 or 1) and calculate the term

                if g < self.fg:                   
                    gain_coef += 2**-(self.fg-g) * (1 if bool_value else 0)
                else: 
                    gain_coef += 2**(g-self.fg) * (1 if bool_value else 0)

            self.gain_res=gain_coef
            print("gain Coeffs: ", self.gain_res)
                      

            

        else:
            print("Unsatisfiable")
            end_time = time.time()

        print("solver stopped")
        duration = end_time - start_time
        print(f"Duration: {duration} seconds")

    def plot_result(self, result_coef):
        print("result plotter called")
        fir_coefficients = np.array([])
        for i in range(len(result_coef)):
            fir_coefficients = np.append(fir_coefficients, result_coef[(i+1)*-1])

        for i in range(len(result_coef)-1):
            fir_coefficients = np.append(fir_coefficients, result_coef[i+1])

        print(fir_coefficients)

        print("Fir coef in mp", fir_coefficients)

        # Compute the FFT of the coefficients
        N = 5120  # Number of points for the FFT
        frequency_response = np.fft.fft(fir_coefficients, N)
        frequencies = np.fft.fftfreq(N, d=1.0)[:N//2]  # Extract positive frequencies up to Nyquist

        # Compute the magnitude and phase response for positive frequencies
        magnitude_response = np.abs(frequency_response)[:N//2]

        # Convert magnitude response to dB
        magnitude_response_db = 20 * np.log10(np.where(magnitude_response == 0, 1e-10, magnitude_response))

        # print("magdb in mp", magnitude_response_db)

        # Normalize frequencies to range from 0 to 1
        omega= frequencies * 2 * np.pi
        normalized_omega = omega / np.max(omega)
        self.ax1.set_ylim([-10, 10])
        # Convert lists to numpy arrays
        freq_upper_lin_array = np.array(self.freq_upper_lin, dtype=np.float64)
        freq_lower_lin_array = np.array(self.freq_lower_lin, dtype=np.float64)

        # Perform element-wise division
        self.freq_upper_lin = ((freq_upper_lin_array/((10**self.coef_accuracy)*(2**self.fh))) * self.gain_res).tolist()
        self.freq_lower_lin = ((freq_lower_lin_array/((10**self.coef_accuracy)*(2**self.fh))) * self.gain_res).tolist()


        #plot input
        self.ax1.scatter(self.freqx_axis, self.freq_upper_lin, color='r', s=20, picker=5)
        self.ax1.scatter(self.freqx_axis, self.freq_lower_lin, color='b', s=20, picker=5)

        # Plot the updated upper_ydata
        self.ax1.plot(normalized_omega, magnitude_response, color='y')

        if self.app:
            self.app.canvas.draw()

    def plot_validation(self):
        print("Validation plotter called")
        half_order = (self.order_current // 2)
        sf = SolverFunc(self.filter_type, self.order_current)
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
        self.ax2.plot([x/1 for x in self.freqx_axis], computed_frequency_response, color='green', label='Computed Frequency Response')

        self.ax2.set_ylim(-10,10)


        if self.app:
            self.app.canvas.draw()



    

    

if __name__ == "__main__":
    # Test inputs
    filter_type = 0
    order_upper = 6
    accuracy = 5
    adder_count = 2
    wordlength = 10

    # Initialize freq_upper and freq_lower with NaN values
    freqx_axis = np.linspace(0, 1, accuracy*order_upper) #according to Mr. Kumms paper
    freq_upper = np.full(accuracy * order_upper, np.nan)
    freq_lower = np.full(accuracy * order_upper, np.nan)

    # Manually set specific values for the elements of freq_upper and freq_lower in dB
    lower_half_point = int(0.3*(accuracy*order_upper))
    upper_half_point = int(0.9*(accuracy*order_upper))
    end_point = accuracy*order_upper

    freq_upper[0:lower_half_point] = 5
    freq_lower[0:lower_half_point] = -1

    freq_upper[upper_half_point:end_point] = -40
    freq_lower[upper_half_point:end_point] = -1000




    #beyond this bound lowerbound will be ignored
    ignore_lowerbound = -40

    # Create FIRFilter instance
    fir_filter = FIRFilter(filter_type, order_upper, freqx_axis, freq_upper, freq_lower, ignore_lowerbound, adder_count, wordlength)

    # Run solver and plot result
    fir_filter.runsolver()
    fir_filter.plot_result(fir_filter.h_res)
    fir_filter.plot_validation()

    # Show plot
    plt.show()