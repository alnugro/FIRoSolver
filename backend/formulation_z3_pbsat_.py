import numpy as np
from z3 import *
import matplotlib.pyplot as plt
import time
from solver_func import SolverFunc

class FIRFilterZ3:
    def __init__(self, 
                 filter_type, 
                 order_upper, 
                 freqx_axis, 
                 freq_upper, 
                 freq_lower, 
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
                 ):
        self.filter_type = filter_type
        self.order_upper = order_upper
        self.freqx_axis = freqx_axis
        self.freq_upper = freq_upper
        self.freq_lower = freq_lower
        self.h_res = []
        self.gain_res = 0

        self.wordlength = wordlength
        self.max_adder = adder_count
        self.adder_wordlength = self.wordlength + adder_wordlength_ext


        self.freq_upper_lin=0
        self.freq_lower_lin=0

        self.coef_accuracy = coef_accuracy
        self.intW = intW
        self.fracW = self.wordlength - self.intW

        
        self.gain_wordlength=6 #9 bits wordlength for gain
        self.gain_intW = 3
        self.gain_fracW =  self.gain_wordlength - self.gain_intW

        self.gain_upperbound= gain_upperbound
        self.gain_lowerbound= gain_lowerbound
        self.gain_bound_accuracy = 2 #2 floating points


        self.ignore_lowerbound = ignore_lowerbound

        self.adder_depth = adder_depth
        self.avail_dsp = avail_dsp
        self.result_model = {}





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

        h = [[Bool(f'h_{a}_{w}') for w in range(self.wordlength)] for a in range(half_order+1)]
        gain= [Bool(f'gain_{g}') for g in range(self.gain_wordlength)]

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
        c=[[Bool(f'c_{i}_{w}') for w in range(self.adder_wordlength)] for i in range(self.max_adder+2)]
        l=[[Bool(f'l_{i}_{w}') for w in range(self.adder_wordlength)] for i in range(1, self.max_adder+1)]
        r=[[Bool(f'r_{i}_{w}') for w in range(self.adder_wordlength)] for i in range(1, self.max_adder+1)]

        
        alpha = [[Bool(f'alpha_{i}_{a}') for a in range(i)] for i in range(1, self.max_adder+1)]
        beta =[[ Bool(f'Beta_{i}_{a}') for a in range(i)] for i in range(1, self.max_adder+1)] 

        
        # c0,w is always 0 except 1
        for w in range(self.fracW+1, self.adder_wordlength):
            solver.add(Not(c[0][w]))

        for w in range(self.fracW):
            solver.add(Not(c[0][w]))

        solver.add(c[0][self.fracW])

        #bound ci,0 to be odd number
        for i in range(1,self.max_adder+1):
            solver.add(c[i][0])

        #last c or c[N+1] is connected to ground, so all zeroes
        for w in range(self.adder_wordlength):
            solver.add(Not(c[self.max_adder+1][w]))


        #input multiplexer
        for i in range(1, self.max_adder+1):
            alpha_sum = []
            beta_sum = []
            for a in range(i):
                for word in range(self.adder_wordlength):
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
        gamma = [[Bool(f'gamma_{i}_{k}') for k in range(self.adder_wordlength-1)] for i in range(1, self.max_adder+1)]
        s     = [[Bool(f's_{i}_{w}') for w in range(self.adder_wordlength)] for i in range(1, self.max_adder+1)]


        for i in range(1, self.max_adder+1):
            gamma_sum = []
            for k in range(self.adder_wordlength-1):
                for j in range(self.adder_wordlength-1-k):
                    clause3_1 = Or(Not(gamma[i-1][k]),Not(l[i-1][j]),s[i-1][j+k])
                    clause3_2 = Or(Not(gamma[i-1][k]),l[i-1][j],Not(s[i-1][j+k]))
                    # solver.add(And(clause3_1, clause3_2))
                    solver.add(clause3_1)
                    solver.add(clause3_2)


                gamma_sum.append(gamma[i-1][k])
            
            solver.add(AtMost(*gamma_sum,1))
            solver.add(AtLeast(*gamma_sum,1))


            for kf in range(1,self.adder_wordlength-1):
                for b in range(kf):
                    clause4 = Or(Not(gamma[i-1][kf]),Not(s[i-1][b]))
                    clause5 = Or(Not(gamma[i-1][kf]), Not(l[i-1][self.adder_wordlength-1]), l[i-1][self.adder_wordlength-2-b])
                    clause6 = Or(Not(gamma[i-1][kf]), l[i-1][self.adder_wordlength-1], Not(l[i-1][self.adder_wordlength-2-b]))
                    solver.add(clause4)
                    solver.add(clause5)
                    solver.add(clause6)

            clause7_1= Or(Not(l[i-1][self.adder_wordlength-1]), s[i-1][self.adder_wordlength-1])
            clause7_2= Or(l[i-1][self.adder_wordlength-1], Not(s[i-1][self.adder_wordlength-1]))
            # solver.add(And(clause7_1, clause7_2))
            solver.add(clause7_1)
            solver.add(clause7_2)
            


        delta = [Bool(f'delta_{i}') for i in range(1, self.max_adder+1)]
        u     = [[Bool(f'u_{i}_{w}') for w in range(self.adder_wordlength)] for i in range(1, self.max_adder+1)]
        x     = [[Bool(f'x_{i}_{w}') for w in range(self.adder_wordlength)] for i in range(1, self.max_adder+1)]

   
    
        #delta selector
        for i in range(1, self.max_adder+1):
            for word in range(self.adder_wordlength):
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

                
        epsilon = [Bool(f'epsilon_{i}') for i in range(1, self.max_adder+1)]
        y     = [[Bool(f'y_{i}_{w}') for w in range(self.adder_wordlength)] for i in range(1, self.max_adder+1)]


        #xor
        for i in range(1, self.max_adder+1):
            for word in range(self.adder_wordlength):
                clause12 = Or(u[i-1][word], epsilon[i-1], Not(y[i-1][word]))
                clause13 = Or(u[i-1][word], Not(epsilon[i-1]), y[i-1][word])
                clause14 = Or(Not(u[i-1][word]), epsilon[i-1], y[i-1][word])
                clause15 = Or(Not(u[i-1][word]), Not(epsilon[i-1]), Not(y[i-1][word]))
                solver.add(clause12)
                solver.add(clause13)
                solver.add(clause14)
                solver.add(clause15)

        
        #ripple carry
        z     = [[Bool(f'z_{i}_{w}') for w in range(self.adder_wordlength)] for i in range(1, self.max_adder+1)]
        cout  = [[Bool(f'cout_{i}_{w}') for w in range(self.adder_wordlength)] for i in range(1, self.max_adder+1)]

        
        for i in range(1, self.max_adder+1):
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

            for kf in range(1, self.adder_wordlength):
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

            clause44 = Or(epsilon[i-1], x[i-1][self.adder_wordlength-1], u[i-1][self.adder_wordlength-1], Not(z[i-1][self.adder_wordlength-1]))
            clause45 = Or(epsilon[i-1], Not(x[i-1][self.adder_wordlength-1]), Not(u[i-1][self.adder_wordlength-1]), z[i-1][self.adder_wordlength-1])
            clause46 = Or(Not(epsilon[i-1]), x[i-1][self.adder_wordlength-1], Not(u[i-1][self.adder_wordlength-1]), Not(z[i-1][self.adder_wordlength-1]))
            clause47 = Or(Not(epsilon[i-1]), Not(x[i-1][self.adder_wordlength-1]), u[i-1][self.adder_wordlength-1], z[i-1][self.adder_wordlength-1])

            solver.add(clause44)
            solver.add(clause45)
            solver.add(clause46)
            solver.add(clause47)


        #right shift
        zeta = [[Bool(f'zeta_{i}_{k}') for k in range(self.adder_wordlength-1)] for i in range(1, self.max_adder+1)]



        for i in range(1, self.max_adder+1):
            zeta_sum = []
            for k in range(self.adder_wordlength-1):
                for j in range(self.adder_wordlength-1-k):
                    clause48_1 = Or(Not(zeta[i-1][k]),Not(z[i-1][j+k]),c[i][j])
                    clause48_2 = Or(Not(zeta[i-1][k]),z[i-1][j+k],Not(c[i][j]))
                    # solver.add(And(clause48_1, clause48_2))
                    solver.add(clause48_1)
                    solver.add(clause48_2)

                zeta_sum.append(zeta[i-1][k])
            
            solver.add(AtMost(*zeta_sum,1))
            solver.add(AtLeast(*zeta_sum,1))


            for kf in range(1,self.adder_wordlength-1):
                for b in range(kf):
                    clause49_1 = Or(Not(zeta[i-1][kf]), Not(z[i-1][self.adder_wordlength-1]), c[i][self.adder_wordlength-2-b])
                    clause49_2 = Or(Not(zeta[i-1][kf]), z[i-1][self.adder_wordlength-1], Not(c[i][self.adder_wordlength-2-b]))
                    solver.add(clause49_1)
                    solver.add(clause49_2)

                    clause50 = Or(Not(zeta[i-1][kf]), Not(z[i-1][b]))
                    solver.add(clause50)
            
            clause51_1 = Or(Not(z[i-1][self.adder_wordlength-1]), c[i][self.adder_wordlength-1])
            clause51_2 = Or(z[i-1][self.adder_wordlength-1], Not(c[i][self.adder_wordlength-1]))
            # solver.add(And(clause51_1, clause51_2))
            solver.add(clause51_1)
            solver.add(clause51_2)

        


            

        #set connected coefficient
        connected_coefficient = half_order+1-self.avail_dsp

        #solver connection
        theta = [[Bool(f'theta_{i}_{m}') for m in range(half_order+1)] for i in range(self.max_adder+2)]
        iota = [Bool(f'iota_{m}') for m in range(half_order+1)]
        t = [[Bool(f't_{m}_{w}') for w in range(self.adder_wordlength)] for m in range(half_order+1)]
        

        
        iota_sum = []
        for m in range(half_order+1):
            theta_or = []
            for i in range(self.max_adder+2):
                for word in range(self.adder_wordlength):
                    clause52_1=Or(Not(theta[i][m]), Not(iota[m]), Not(c[i][word]),t[m][word])
                    clause52_2=Or(Not(theta[i][m]), Not(iota[m]), c[i][word],Not(t[m][word]))
                    solver.add(clause52_1)
                    solver.add(clause52_2)
                theta_or.append(theta[i][m])
            # print(f"theta or {theta_or}")
            solver.add(Or(*theta_or))
        
        for m in range(half_order+1):
            iota_sum.append(iota[m])
        solver.add(AtMost(*iota_sum,connected_coefficient))
        solver.add(AtLeast(*iota_sum,connected_coefficient))

        #Left Shifter in result module
        #k is the shift selector
        h_ext = [[Bool(f'h_ext_{m}_{w}') for w in range(self.adder_wordlength)] for m in range(half_order+1)]
        phi = [[Bool(f'phi_{m}_{k}') for k in range(self.adder_wordlength-1)] for m in range(half_order+1)]

        for m in range(half_order+1):
            phi_sum = []
            for k in range(self.adder_wordlength-1):
                for j in range(self.adder_wordlength-1-k):
                    clause53_1 = Or(Not(phi[m][k]),Not(t[m][j]),h_ext[m][j+k])
                    clause53_2 = Or(Not(phi[m][k]),t[m][j],Not(h_ext[m][j+k]))
                    # solver.add(And(clause3_1, clause3_2))
                    solver.add(clause53_1)
                    solver.add(clause53_2)


                phi_sum.append(phi[m][k])
            
            solver.add(AtMost(*phi_sum,1))
            solver.add(AtLeast(*phi_sum,1))


            for kf in range(1,self.adder_wordlength-1):
                for b in range(kf):
                    clause54 = Or(Not(phi[m][kf]),Not(h_ext[m][b]))
                    clause55 = Or(Not(phi[m][kf]), Not(t[m][self.adder_wordlength-1]), t[m][self.adder_wordlength-2-b])
                    clause56 = Or(Not(phi[m][kf]), t[m][self.adder_wordlength-1], Not(t[m][self.adder_wordlength-2-b]))
                    solver.add(clause54)
                    solver.add(clause55)
                    solver.add(clause56)

            clause57_1= Or(Not(t[m][self.adder_wordlength-1]), h_ext[m][self.adder_wordlength-1])
            clause57_2= Or(t[m][self.adder_wordlength-1], Not(h_ext[m][self.adder_wordlength-1]))
            # solver.add(And(clause7_1, clause7_2))
            solver.add(clause57_1)
            solver.add(clause57_2)

        for m in range(half_order+1):
            for word in range(self.adder_wordlength):
                if word <= self.wordlength-1:
                    clause58 = Or(h[m][word],Not(h_ext[m][word]))
                    clause59 = Or(Not(h[m][word]),h_ext[m][word])
                    # solver.add(And(clause3_1, clause3_2))
                    solver.add(clause58)
                    solver.add(clause59)
                else: 
                    clause58 = Or(h[m][self.wordlength-1],Not(h_ext[m][word]))
                    clause59 = Or(Not(h[m][self.wordlength-1]),h_ext[m][word])
                    # solver.add(And(clause3_1, clause3_2))
                    solver.add(clause58)
                    solver.add(clause59)

        # adder depth constraint
        if self.adder_depth > 0:
            psi_alpha = [[Bool(f'psi_alpha_{i}_{d}') for d in range(self.adder_depth)] for i in range(1, self.max_adder+1)]
            psi_beta = [[Bool(f'psi_beta_{i}_{d}') for d in range(self.adder_depth)] for i in range(1, self.max_adder+1)]
            psi_alpha_sum = []
            psi_beta_sum = []

            for i in range(1, self.max_adder+1):
                clause60 = Or(Not(psi_alpha[i-1][0]),alpha[i-1][0])
                clause61 = Or(Not(psi_beta[i-1][0]),beta[i-1][0])
                solver.add(clause60)
                solver.add(clause61)
                for d in range(self.adder_depth):
                    psi_alpha_sum.append(psi_alpha[i-1][d])
                    psi_beta_sum.append(psi_beta[i-1][d])

                solver.add(AtMost(*psi_alpha_sum,1))
                solver.add(AtLeast(*psi_alpha_sum,1))
                solver.add(AtMost(*psi_beta_sum,1))
                solver.add(AtLeast(*psi_beta_sum,1))

                if d == 1:
                    continue
                for d in range(1,self.adder_depth):
                    for a in range(i-1):
                        clause63 = Or(Not(psi_alpha[i-1][d]),alpha[i-1][a])
                        clause64 = Or(Not(psi_alpha[i-1][d]),psi_alpha[i-1][d-1])
                        solver.add(clause63)
                        solver.add(clause64)

                        clause65 = Or(Not(psi_beta[i-1][d]),beta[i-1][a])
                        clause66 = Or(Not(psi_beta[i-1][d]),psi_beta[i-1][d-1])
                        solver.add(clause65)
                        solver.add(clause66)



        
        start_time=time.time()
        solver.set(unsat_core=True)


        print("solver runing")



        # print(filter_coeffs)
        # print(filter_literals)

        satifiability = 'unsat'

        if solver.check() == sat:

            satifiability = 'sat'

            print("solver sat")
            model = solver.model()

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

            # Store gain coefficients
            self.result_model[f"gain"] =  self.gain_res
            
            # Store h coefficients
            for m in range(half_order + 1):
                self.result_model[f"h[{m}]"] = self.h_res[m]

            # Store h coefficients
            for m in range(half_order + 1):
                for w in range(self.wordlength):
                    self.result_model[f"h[{m}][{w}]"] = 1 if model.eval(h[m][w], model_completion=True) else 0

            # Store alpha and beta selectors
            for i in range(len(alpha)):
                for a in range(len(alpha[i])):
                    self.result_model[f'alpha[{i+1}][{a}]'] = 1 if model.eval(alpha[i][a], model_completion=True) else 0
                for a in range(len(beta[i])):
                    self.result_model[f'beta[{i+1}][{a}]'] = 1 if model.eval(beta[i][a], model_completion=True) else 0

            # Store gamma (left shift selectors)
            for i in range(len(gamma)):
                for k in range(self.adder_wordlength - 1):
                    self.result_model[f'gamma[{i+1}][{k}]'] = 1 if model.eval(gamma[i][k], model_completion=True) else 0

            # Store delta selectors and u/x arrays
            for i in range(len(delta)):
                self.result_model[f'delta[{i+1}]'] = 1 if model.eval(delta[i], model_completion=True) else 0


            # Store epsilon selectors and y array (XOR results)
            for i in range(len(epsilon)):
                self.result_model[f'epsilon[{i+1}]'] = 1 if model.eval(epsilon[i], model_completion=True) else 0

            # Store zeta (right shift selectors)
            for i in range(len(zeta)):
                for k in range(self.adder_wordlength - 1):
                    self.result_model[f'zeta[{i+1}][{k}]'] = 1 if model.eval(zeta[i][k], model_completion=True) else 0
            
            # Store phi (left shift selectors in result)
            for i in range(len(phi)):
                for k in range(self.adder_wordlength - 1):
                    self.result_model[f'phi[{i}][{k}]'] = 1 if model.eval(phi[i][k], model_completion=True) else 0

            # Store theta array
            for i in range(len(theta)):
                for m in range(half_order + 1):
                    self.result_model[f'theta[{i}][{m}]'] = 1 if model.eval(theta[i][m], model_completion=True) else 0

            # Store iota array
            for m in range(len(iota)):
                self.result_model[f'iota[{m}]'] = 1 if model.eval(iota[m], model_completion=True) else 0


            end_time = time.time()
            
            
                      

        else:
            print(f"unsat core {solver.unsat_core()}")
            print("Unsatisfiable")
            end_time = time.time()

        print("solver stopped")
        duration = end_time - start_time
        print(f"Duration: {duration} seconds")

        # for item in self.result_model:
        #     print(f"result of {item} is {self.result_model[item]}")

        print(f"\n************Z3 Report****************")
        print(f"Total number of variables            : ")
        print(f"Total number of constraints (clauses): {len(solver.assertions())}\n" )

        return duration , satifiability


if __name__ == "__main__":
    # Test inputs
    filter_type = 0
    order_upper = 20
    accuracy = 1
    adder_count = 3
    wordlength = 10
    
    adder_depth = 2
    avail_dsp = 0
    adder_wordlength_ext = 2
    gain_upperbound = 3
    gain_lowerbound = 1
    coef_accuracy = 4
    intW = 4

    space = int(accuracy*order_upper)
    # Initialize freq_upper and freq_lower with NaN values
    freqx_axis = np.linspace(0, 1, space) #according to Mr. Kumms paper
    freq_upper = np.full(space, np.nan)
    freq_lower = np.full(space, np.nan)

    # Manually set specific values for the elements of freq_upper and freq_lower in dB
    lower_half_point = int(0.5*(space))
    upper_half_point = int(0.6*(space))
    end_point = space

    freq_upper[0:lower_half_point] = 3
    freq_lower[0:lower_half_point] = -1

    freq_upper[upper_half_point:end_point] = -40
    freq_lower[upper_half_point:end_point] = -1000


    #beyond this bound lowerbound will be ignored
    ignore_lowerbound = -40

    # Create FIRFilter instance
    fir_filter = FIRFilterZ3(
                 filter_type, 
                 order_upper, 
                 freqx_axis, 
                 freq_upper, 
                 freq_lower, 
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
                 )

    # Run solver and plot result
    fir_filter.runsolver()
    
    