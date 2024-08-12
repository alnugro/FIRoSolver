import numpy as np
from z3 import *
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
        
    def overflow_handler(self, input_coeffs, literal):
        max_positive_int_pbfunc = 2147483647
        max_negative_int_pbfunc = -2147483648

        self.overflow_count+=1
        overflow_coef = []
        overflow_lit = []

        if input_coeffs > max_positive_int_pbfunc:
            while input_coeffs > max_positive_int_pbfunc:
                overflow_coef.append(max_positive_int_pbfunc)
                overflow_lit.append(literal)
                input_coeffs -= max_positive_int_pbfunc
            overflow_coef.append(input_coeffs)
            overflow_lit.append(literal)
            print("overflow happened in:", input_coeffs, " with literall: ", literal)
        
        elif input_coeffs < max_negative_int_pbfunc:
            while input_coeffs < max_negative_int_pbfunc:
                overflow_coef.append(max_negative_int_pbfunc)
                overflow_lit.append(literal)
                input_coeffs -= max_negative_int_pbfunc
            overflow_coef.append(input_coeffs)
            overflow_lit.append(literal)
            print("overflow happened in:", input_coeffs, " with literall: ", literal)
        
        else:
            overflow_coef.append(input_coeffs)
            overflow_lit.append(literal)

        return overflow_lit, overflow_coef




class FIRFilterZ3:
    def __init__(self, filter_type, order_upper, freqx_axis, freq_upper, freq_lower, ignore_lowerbound, adder_count, wordlength, app=None):
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
        self.intW = 4
        self.fracW = self.wordlength - self.intW

        
        self.gain_wordlength=6 #9 bits wordlength for gain
        self.gain_intW = 3
        self.gain_fracW =  self.gain_wordlength - self.gain_intW

        self.gain_upperbound= 1.4
        self.gain_lowerbound= 1
        self.gain_bound_accuracy = 2 #2 floating points

        
        

        self.ignore_lowerbound = ignore_lowerbound




    def runsolver(self):
        self.order_current = int(self.order_upper)
        half_order = (self.order_current // 2)
        
        print("solver called")
        sf = SolverFunc(self.filter_type, self.order_current)

        print("filter order:", self.order_current)
        print("ignore lower than:", self.ignore_lowerbound)
        # linearize the bounds
        self.freq_upper_lin = [int((sf.db_to_linear(f))*(10**self.coef_accuracy)*(2**(self.fracW-self.gain_fracW))) if not np.isnan(sf.db_to_linear(f)) else np.nan for f in self.freq_upper]
        self.freq_lower_lin = [int((sf.db_to_linear(f))*(10**self.coef_accuracy)*(2**(self.fracW-self.gain_fracW))) if not np.isnan(sf.db_to_linear(f)) else np.nan for f in self.freq_lower]
        self.ignore_lowerbound_np = np.array(self.ignore_lowerbound, dtype=float)

        self.ignore_lowerbound_lin = sf.db_to_linear(self.ignore_lowerbound_np)
        self.ignore_lowerbound_lin = self.ignore_lowerbound_lin*(10**self.coef_accuracy)*(2**self.fracW)

        h = [[Bool(f'h{a}_{w}') for w in range(self.wordlength)] for a in range(half_order+1)]
        gain= [Bool(f'gain{g}') for g in range(self.gain_wordlength)]

        solver = Solver()


        gain_coeffs = []
        gain_literalls=[]


        #bounds the gain
        self.gain_upperbound_int = int(self.gain_upperbound*2**self.gain_fracW*(10**self.gain_bound_accuracy))
        self.gain_lowerbound_int = int(self.gain_lowerbound*2**self.gain_fracW*(10**self.gain_bound_accuracy))

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

        gain_upper_literalls = []
        gain_lower_literalls = []


        



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



            
            gain_upper_literalls.clear()
            gain_lower_literalls.clear()

            for m in range(half_order+1):
                cm = sf.cm_handler(m, self.freqx_axis[omega])
                for w in range(self.wordlength):
                    if w==self.wordlength-1:
                        cm_word_prod= int(cm*(10** self.coef_accuracy)*(-1*(2**w)))
                    else: cm_word_prod= int(cm*(10** self.coef_accuracy)*(2**w))
                    filter_literals_temp, filter_coeffs_temp = sf.overflow_handler(cm_word_prod,h[m][w])
                    filter_coeffs.extend(filter_coeffs_temp)
                    filter_literals.extend(filter_literals_temp)

            for g in range(self.gain_wordlength):
                gain_upper_prod = int(-1 * (2**g) * self.freq_upper_lin[omega])
                gain_upper_literalls_temp, gain_freq_upper_prod_coeffs_temp = sf.overflow_handler(gain_upper_prod,gain[g])
                gain_freq_upper_prod_coeffs.extend(gain_freq_upper_prod_coeffs_temp)
                gain_upper_literalls.extend(gain_upper_literalls_temp)

                if self.freq_lower_lin[omega] < self.ignore_lowerbound_lin:
                    gain_lower_prod=int((2**g) * self.freq_upper_lin[omega])
                    gain_lower_literalls_temp, gain_freq_lower_prod_coeffs_temp = sf.overflow_handler(gain_lower_prod,gain[g])
                    gain_freq_lower_prod_coeffs.extend(gain_freq_lower_prod_coeffs_temp)
                    gain_lower_literalls.extend(gain_lower_literalls_temp)
                    print("ignored ",self.freq_lower_lin[omega], " in frequency = ", self.freqx_axis[omega])
                else:
                    gain_lower_prod=int(-1 *(2**g) * self.freq_lower_lin[omega])
                    gain_lower_literalls_temp, gain_freq_lower_prod_coeffs_temp = sf.overflow_handler(gain_lower_prod,gain[g])
                    gain_freq_lower_prod_coeffs.extend(gain_freq_lower_prod_coeffs_temp)
                    gain_lower_literalls.extend(gain_lower_literalls_temp)

            filter_upper_pb_coeffs=filter_coeffs+gain_freq_upper_prod_coeffs
            filter_upper_pb_literalls=filter_literals+gain_upper_literalls

            # print("coeffs: ",filter_upper_pb_coeffs)
            # print("lit: ",filter_upper_pb_literalls)

            if len(filter_upper_pb_coeffs) != len(filter_upper_pb_literalls):
                raise("sumtin wong with upper filter pb")
            
            # else: print("filter upperbound length is validated")

            

            #z3 only take pairs
            filter_upper_pb_pairs = [(filter_upper_pb_literalls[i],filter_upper_pb_coeffs[i],) for i in range(len(filter_upper_pb_literalls))]
            solver.add(PbLe(filter_upper_pb_pairs, 0))

            
           
            filter_lower_pb_coeffs=filter_coeffs+gain_freq_lower_prod_coeffs
            filter_lower_pb_literalls=filter_literals+gain_lower_literalls

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
        c=[[Bool(f'c{i}_{w}') for w in range(self.wordlength)] for i in range(self.N+1)]
        l=[[Bool(f'l{i}_{w}') for w in range(self.wordlength)] for i in range(1, self.N+1)]
        r=[[Bool(f'r{i}_{w}') for w in range(self.wordlength)] for i in range(1, self.N+1)]


        alpha = [[Bool(f'alpha{i}_{a}') for a in range(i)] for i in range(1, self.N+1)]
        beta =[[ Bool(f'Beta{i}_{a}') for a in range(i)] for i in range(1, self.N+1)] 


        # c0,w is always 0 except 1
        for w in range(self.fracW+1, self.wordlength):
            solver.add(Not(c[0][w]))

        for w in range(self.fracW):
            solver.add(Not(c[0][w]))

        solver.add(c[0][self.fracW])


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

                alpha_sum.append(alpha[i-1][a])
                beta_sum.append(beta[i-1][a])

           
            solver.add(AtMost(*alpha_sum,1))
            solver.add(AtLeast(*alpha_sum,1))

            solver.add(AtMost(*beta_sum,1))
            solver.add(AtLeast(*beta_sum,1))

        #Left Shifter
        #k is the shift selector
        gamma = [[Bool(f'gamma{i}_{k}') for k in range(self.wordlength-1)] for i in range(1, self.N+1)]
        s     = [[Bool(f's{i}_{w}') for w in range(self.wordlength)] for i in range(1, self.N+1)]


        for i in range(1, self.N+1):
            gamma_sum = []
            for k in range(self.wordlength-1):
                for j in range(self.wordlength-1-k):
                    clause3_1 = Or(Not(gamma[i-1][k]),Not(l[i-1][j]),s[i-1][j+k])
                    clause3_2 = Or(Not(gamma[i-1][k]),l[i-1][j],Not(s[i-1][j+k]))
                    # solver.add(And(clause3_1, clause3_2))
                    solver.add(clause3_1)
                    solver.add(clause3_2)


                gamma_sum.append(gamma[i-1][k])
            
            solver.add(AtMost(*gamma_sum,1))
            solver.add(AtLeast(*gamma_sum,1))


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
            # solver.add(And(clause7_1, clause7_2))
            solver.add(clause7_1)
            solver.add(clause7_2)
            


        delta = [Bool(f'delta{i}') for i in range(1, self.N+1)]
        u     = [[Bool(f'u{i}_{w}') for w in range(self.wordlength)] for i in range(1, self.N+1)]
        x     = [[Bool(f'x{i}_{w}') for w in range(self.wordlength)] for i in range(1, self.N+1)]

   
    
        #delta selector
        for i in range(1, self.N+1):
            for word in range(self.wordlength):
                clause8_1 = Or(Not(delta[i-1]),Not(s[i-1][word]),x[i-1][word])
                clause8_2 = Or(Not(delta[i-1]),s[i-1][word],Not(x[i-1][word]))
                # solver.add(And(clause8_1, clause8_2))
                solver.add(clause8_1)
                solver.add(clause8_2)
                
                clause9_1 = Or(Not(delta[i-1]),Not(r[i-1][word]),u[i-1][word])
                clause9_2 = Or(Not(delta[i-1]),r[i-1][word],Not(u[i-1][word]))
                # solver.add(And(clause9_1, clause9_2))
                solver.add(clause9_1)
                solver.add(clause9_2)

                clause10_1 = Or(delta[i-1],Not(s[i-1][word]),u[i-1][word])
                clause10_2 = Or(delta[i-1],s[i-1][word],Not(u[i-1][word]))
                # solver.add(And(clause10_1, clause10_2))
                solver.add(clause10_1)
                solver.add(clause10_2)

                clause11_1 = Or(delta[i-1],Not(r[i-1][word]),x[i-1][word])
                clause11_2 = Or(delta[i-1],r[i-1][word],Not(x[i-1][word]))
                # solver.add(And(clause11_1, clause11_2))
                solver.add(clause11_1)
                solver.add(clause11_2)

                solver.add(Or(delta[i-1], Not(delta[i-1])))
                
        epsilon = [Bool(f'epsilon{i}') for i in range(1, self.N+1)]
        y     = [[Bool(f'y{i}_{w}') for w in range(self.wordlength)] for i in range(1, self.N+1)]


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
        z     = [[Bool(f'z{i}_{w}') for w in range(self.wordlength)] for i in range(1, self.N+1)]
        cout  = [[Bool(f'cout{i}_{w}') for w in range(self.wordlength)] for i in range(1, self.N+1)]

        
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
        zeta = [[Bool(f'zeta{i}_{k}') for k in range(self.wordlength-1)] for i in range(1, self.N+1)]



        for i in range(1, self.N+1):
            zeta_sum = []
            for k in range(self.wordlength-1):
                for j in range(self.wordlength-1-k):
                    clause48_1 = Or(Not(zeta[i-1][k]),Not(z[i-1][j+k]),c[i][j])
                    clause48_2 = Or(Not(zeta[i-1][k]),z[i-1][j+k],Not(c[i][j]))
                    # solver.add(And(clause48_1, clause48_2))
                    solver.add(clause48_1)
                    solver.add(clause48_2)

                zeta_sum.append(zeta[i-1][k])
            
            solver.add(AtMost(*zeta_sum,1))
            solver.add(AtLeast(*zeta_sum,1))


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
            # solver.add(And(clause51_1, clause51_2))
            solver.add(clause51_1)
            solver.add(clause51_2)

      
            #bound ci,0 to be odd number 
            solver.add(c[i][0])

        #set connected coefficient
        connected_coefficient = half_order+1

        #solver connection
        h = [[Bool(f'h{m}_{w}') for w in range(self.wordlength)] for m in range(half_order+1)]
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
                    # solver.add(And(clause52_1, clause52_2)
                    solver.add(clause52_1)
                    solver.add(clause52_2)


                t_or_clauses.append(t[i-1][m])
            solver.add(Or(t_or_clauses))

            e_sum.append(e[m])
        
        solver.add(AtMost(*e_sum,connected_coefficient))
        solver.add(AtLeast(*e_sum,connected_coefficient))


        
        start_time=time.time()

        print("solver runing")



        # print(filter_coeffs)
        # print(filter_literals)

        satifiability = False

        if solver.check() == sat:
            satifiability = True

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
                        fir_coef += -2**(w-self.fracW)  * (1 if bool_value else 0)
                    elif w < self.fracW:                   
                        fir_coef += 2**(-1*(self.fracW-w)) * (1 if bool_value else 0)
                    else:
                        fir_coef += 2**(w-self.fracW) * (1 if bool_value else 0)
                
                self.h_res.append(fir_coef)
            print("FIR Coeffs calculated: ",self.h_res)
            
            gain_coef=0
            for g in range(self.gain_wordlength):
                # Evaluate the boolean value from the model
                bool_value = model.eval(gain[g], model_completion=True)
                print(f"gain{g}= ",bool_value)

                # Convert boolean to integer (0 or 1) and calculate the term

                if g < self.gain_fracW:                   
                    gain_coef += 2**-(self.gain_fracW-g) * (1 if bool_value else 0)
                else: 
                    gain_coef += 2**(g-self.gain_fracW) * (1 if bool_value else 0)

            self.gain_res=gain_coef
            print("gain Coeffs: ", self.gain_res)
                      

        else:
            print("Unsatisfiable")
            end_time = time.time()

        print("solver stopped")
        duration = end_time - start_time
        print(f"Duration: {duration} seconds")

        return duration , satifiability


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
        self.freq_upper_lin = ((freq_upper_lin_array/((10**self.coef_accuracy)*(2**(self.fracW-self.gain_fracW)))) * self.gain_res).tolist()
        self.freq_lower_lin = ((freq_lower_lin_array/((10**self.coef_accuracy)*(2**(self.fracW-self.gain_fracW)))) * self.gain_res).tolist()


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
    order_upper = 10
    accuracy = 10
    adder_count = 10
    wordlength = 10

    # Initialize freq_upper and freq_lower with NaN values
    freqx_axis = np.linspace(0, 1, accuracy*order_upper) #according to Mr. Kumms paper
    freq_upper = np.full(accuracy * order_upper, np.nan)
    freq_lower = np.full(accuracy * order_upper, np.nan)

    # Manually set specific values for the elements of freq_upper and freq_lower in dB
    lower_half_point = int(0.4*(accuracy*order_upper))
    upper_half_point = int(0.6*(accuracy*order_upper))
    end_point = accuracy*order_upper

    freq_upper[0:lower_half_point] = 5
    freq_lower[0:lower_half_point] = 0

    freq_upper[upper_half_point:end_point] = -30
    freq_lower[upper_half_point:end_point] = -1000



    #beyond this bound lowerbound will be ignored
    ignore_lowerbound = -40

    # Create FIRFilter instance
    fir_filter = FIRFilterZ3(filter_type, order_upper, freqx_axis, freq_upper, freq_lower, ignore_lowerbound, adder_count, wordlength)

    # Run solver and plot result
    fir_filter.runsolver()
    fir_filter.plot_result(fir_filter.h_res)
    fir_filter.plot_validation()

    # Show plot
    plt.show()
