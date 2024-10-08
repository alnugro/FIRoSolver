import numpy as np
from z3 import *
import matplotlib.pyplot as plt
import time


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
    def __init__(self, filter_type, order_upper, freqx_axis, freq_upper, freq_lower, ignore_lowerbound_lin, adder_count, app=None):
        self.filter_type = filter_type
        self.order_upper = order_upper
        self.freqx_axis = freqx_axis
        self.freq_upper = freq_upper
        self.freq_lower = freq_lower
        self.h_int_res = []
        self.app = app
        self.fig, (self.ax1, self.ax2) = plt.subplots(2,1)
        self.freq_upper_lin=0
        self.freq_lower_lin=0
        self.filter_accuracy = 4
        self.wordlength= 8
        self.gain_res = 0

        self.gain_wordlength=9 #9 bits wordlength for gain
        self.gain_accuracy = 2 #2 floating points accuracy with max 5.12

        self.gain_upperbound= 1.4
        self.gain_lowerbound= 0.7

        self.N_adder = adder_count


        self.order_current = int(self.order_upper)
        self.half_order = (self.order_current // 2)
        self.connected_coefficient = self.half_order+1

        
        

        self.ignore_lowerbound_lin = ignore_lowerbound_lin*10**(self.filter_accuracy-self.gain_accuracy)




    def runsolver(self):
        

        
        print("solver called")
        sf = SolverFunc(self.filter_type, self.order_current)

        print("filter order:", self.order_current)
        print("ignore lower than:", self.ignore_lowerbound_lin)
        # linearize the bounds
        self.freq_upper_lin = [int((sf.db_to_linear(f)) * 10**(self.filter_accuracy-self.gain_accuracy)) if not np.isnan(sf.db_to_linear(f)) else np.nan for f in self.freq_upper]
        self.freq_lower_lin = [int((sf.db_to_linear(f)) * 10**(self.filter_accuracy-self.gain_accuracy)) if not np.isnan(sf.db_to_linear(f)) else np.nan for f in self.freq_lower]

        # print(len(self.freq_upper_lin) )
        # print(len(self.freq_lower_lin) )
        # print(len(self.freqx_axis) )

        #print(self.freqx_axis, "_",self.freq_upper_lin,"_", self.freq_lower_lin)


        h = [[Bool(f'h{a}_{w}') for w in range(self.wordlength)] for a in range(self.half_order+1)]
        gain= [Bool(f'gain{g}') for g in range(self.gain_wordlength)]

        solver = Solver()


        gain_coeffs = []
        gain_literalls=[]
        #bounds the gain
        self.gain_upperbound_int = int(self.gain_upperbound*(10**self.gain_accuracy))
        self.gain_lowerbound_int = int(self.gain_lowerbound*(10**self.gain_accuracy))

        print(self.gain_upperbound_int)
        print(self.gain_lowerbound_int)

        

        for g in range(self.gain_wordlength):
            gain_coeffs.append(2**g)
            gain_literalls.append(gain[g])

        pb_gain_pairs = [(gain_literalls[i],gain_coeffs[i]) for i in range(len(gain_literalls))]
            
        solver.add(PbLe(pb_gain_pairs, self.gain_upperbound_int))
        solver.add(PbGe(pb_gain_pairs, self.gain_lowerbound_int))

        filter_bool_literalls=[]
        filter_bool_weights = []
        
        for a in range(self.half_order + 1):
            filter_bool_literalls.clear()
            filter_bool_weights.clear()
            for w in range(self.wordlength):
                if w==self.wordlength-1:
                    filter_bool_weights.append(-1*2**w)
                else: filter_bool_weights.append(2**w)
                filter_bool_literalls.append(h[a][w])
            filter_bool_pairs=[(filter_bool_literalls[i],filter_bool_weights[i]) for i in range(len(filter_bool_literalls))]
            solver.add(PbLe(filter_bool_pairs, 2**self.wordlength))
            solver.add(PbGe(filter_bool_pairs, 0))
            

            
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



        for x in range(len(self.freqx_axis)):
            if np.isnan(self.freq_lower_lin[x]):
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

            for a in range(self.half_order+1):
                cm = sf.cm_handler(a, self.freqx_axis[x])
                for w in range(self.wordlength):
                    if w==self.wordlength-1:
                        cm_word_prod= int(cm*(10**self.filter_accuracy)*(-1*(2**w)))
                    else: cm_word_prod= int(cm*(10**self.filter_accuracy)*(2**w))

                    if cm_word_prod > max_positive_int_pbfunc or cm_word_prod < max_negative_int_pbfunc:
                        overflow = sf.overflow_handler(cm_word_prod,max_positive_int_pbfunc,max_negative_int_pbfunc,h[a][w])
                        filter_overflow_literalls.extend(overflow[0])
                        filter_overflow_coeffs.extend(overflow[1])
                        print("overflow happened in the product of cm: appended this to the sum coeff:", overflow[1], " with literall: ", overflow[0])
                        continue

                    filter_coeffs.append(cm_word_prod)
                    filter_literals.append(h[a][w])

            for g in range(self.gain_wordlength):
                gain_upper_prod = int(-1 * (2**g) * self.freq_upper_lin[x])
                 

                if gain_upper_prod > max_positive_int_pbfunc or gain_upper_prod < max_negative_int_pbfunc:
                    overflow = sf.overflow_handler(gain_upper_prod,max_positive_int_pbfunc,max_negative_int_pbfunc,gain[g])
                    gain_upper_overflow_literalls.extend(overflow[0])
                    gain_upper_overflow_coeffs.extend(overflow[1])
                    print("overflow happened in the gain upper product: appended this to the sum coeff:", overflow[1], " with literall: ", overflow[0])
                    continue
                gain_freq_upper_prod_coeffs.append(gain_upper_prod)
                gain_upper_literalls.append(gain[g])

                if self.freq_lower_lin[x] < self.ignore_lowerbound_lin:
                    gain_lower_prod=int((2**g) * self.freq_upper_lin[x])
                    if gain_lower_prod > max_positive_int_pbfunc or gain_lower_prod < max_negative_int_pbfunc:
                        overflow = sf.overflow_handler(gain_lower_prod,max_positive_int_pbfunc,max_negative_int_pbfunc,gain[g])
                        gain_lower_overflow_literalls.extend(overflow[0])
                        gain_lower_overflow_coeffs.extend(overflow[1])
                        print("overflow happened in the gain lower product: appended this to the sum coeff:", overflow[1], " with literall: ", overflow[0])
                        continue
                    gain_freq_lower_prod_coeffs.append(gain_lower_prod)
                    gain_lower_literalls.append(gain[g])
                    print("ignored ",self.freq_lower_lin[x], " in frequency = ", self.freqx_axis[x])
                else:
                    gain_lower_prod=int(-1 *(2**g) * self.freq_lower_lin[x])
                    if gain_lower_prod > max_positive_int_pbfunc or gain_lower_prod < max_negative_int_pbfunc:
                        overflow = sf.overflow_handler(gain_lower_prod,max_positive_int_pbfunc,max_negative_int_pbfunc,gain[g])
                        gain_lower_overflow_literalls.extend(overflow[0])
                        gain_lower_overflow_coeffs.extend(overflow[1])
                        print("overflow happened in the gain lower product: appended this to the sum coeff:", overflow[1], " with literall: ", overflow[0])
                        continue
                    gain_freq_lower_prod_coeffs.append(gain_lower_prod)
                    gain_lower_literalls.append(gain[g])

            filter_upper_pb_coeffs=filter_coeffs+gain_freq_upper_prod_coeffs+filter_overflow_coeffs+gain_upper_overflow_coeffs
            filter_upper_pb_literalls=filter_literals+gain_upper_literalls+filter_overflow_literalls+gain_upper_overflow_literalls

            #print("coeffs: ",filter_upper_pb_coeffs)
            #print("lit: ",filter_upper_pb_literalls)

            if len(filter_upper_pb_coeffs) != len(filter_upper_pb_literalls):
                raise("sumtin wong with upper filter pb")
            
            else: print("filter upperbound length is validated")

            #z3 only take pairs
            filter_upper_pb_pairs = [(filter_upper_pb_literalls[i],filter_upper_pb_coeffs[i],) for i in range(len(filter_upper_pb_literalls))]
            solver.add(PbLe(filter_upper_pb_pairs, 0))

            
           
            filter_lower_pb_coeffs=filter_coeffs+gain_freq_lower_prod_coeffs+filter_overflow_coeffs+gain_lower_overflow_coeffs
            filter_lower_pb_literalls=filter_literals+gain_lower_literalls+filter_overflow_literalls+gain_lower_overflow_literalls

            print("coeffs: ",filter_lower_pb_coeffs)
            print("lit: ",filter_lower_pb_literalls)

            if len(filter_upper_pb_coeffs) != len(filter_upper_pb_literalls):
                raise("sumtin wong with upper filter pb")
            
            else: print("filter lowerbound length is validated")

            
            filter_lower_pb_pairs = [(filter_lower_pb_literalls[i],filter_lower_pb_coeffs[i]) for i in range(len(filter_lower_pb_literalls))]
            
            #z3 only take pairs
            solver.add(PbGe(filter_lower_pb_pairs, 0))
        

        #bitshift parts
        #input multiplexer
        c=[[Bool(f'c{i}{w}') for w in range(self.wordlength)] for i in range(self.N_adder+1)]
        l=[[Bool(f'l{i}{w}') for w in range(self.wordlength)] for i in range(1, self.N_adder+1)]
        r=[[Bool(f'r{i}{w}') for w in range(self.wordlength)] for i in range(1, self.N_adder+1)]


        alpha = [[Bool(f'alpha{i}{a}') for a in range(i)] for i in range(1, self.N_adder+1)]
        beta =[[ Bool(f'Beta{i}{a}') for a in range(i)] for i in range(1, self.N_adder+1)] 


        #c0,w is always 0 except w=0
        for w in range(1,self.wordlength):
            solver.add(Not(c[0][w]))

        solver.add(c[0][0])


        #input multiplexer
        for i in range(1, self.N_adder+1):
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
        gamma = [[Bool(f'gamma{i}{k}') for k in range(self.wordlength-1)] for i in range(1, self.N_adder+1)]
        ls     = [[Bool(f'ls{i}{w}') for w in range(self.wordlength)] for i in range(1, self.N_adder+1)]


        for i in range(1, self.N_adder+1):
            gamma_sum = []
            for k in range(self.wordlength-1):
                for j in range(self.wordlength-1-k):
                    clause3_1 = Or(Not(gamma[i-1][k]),Not(l[i-1][j]),ls[i-1][j+k])
                    clause3_2 = Or(Not(gamma[i-1][k]),l[i-1][j],Not(ls[i-1][j+k]))
                    solver.add(And(clause3_1, clause3_2))

                gamma_sum.append((gamma[i-1][k], 1))
            
            solver.add(PbEq(gamma_sum,1))


            for kf in range(1,self.wordlength-1):
                for b in range(kf):
                    clause4 = Or(Not(gamma[i-1][kf]),Not(ls[i-1][b]))
                    clause5 = Or(Not(gamma[i-1][kf]), Not(l[i-1][self.wordlength-1]), l[i-1][self.wordlength-2-b])
                    clause6 = Or(Not(gamma[i-1][kf]), l[i-1][self.wordlength-1], Not(l[i-1][self.wordlength-2-b]))
                    solver.add(clause4)
                    solver.add(clause5)
                    solver.add(clause6)

            clause7_1= Or(Not(l[i-1][self.wordlength-1]), ls[i-1][self.wordlength-1])
            clause7_2= Or(l[i-1][self.wordlength-1], Not(ls[i-1][self.wordlength-1]))
            solver.add(And(clause7_1, clause7_2))


        delta = [Bool(f'delta{i}') for i in range(1, self.N_adder+1)]
        rd     = [[Bool(f'rd{i}{w}') for w in range(self.wordlength)] for i in range(1, self.N_adder+1)]
        la     = [[Bool(f'la{i}{w}') for w in range(self.wordlength)] for i in range(1, self.N_adder+1)]

   
    
        #delta selector
        for i in range(1, self.N_adder+1):
            for word in range(self.wordlength):
                clause8_1 = Or(Not(delta[i-1]),Not(ls[i-1][word]),la[i-1][word])
                clause8_2 = Or(Not(delta[i-1]),ls[i-1][word],Not(la[i-1][word]))
                solver.add(And(clause8_1, clause8_2))
                
                clause9_1 = Or(Not(delta[i-1]),Not(r[i-1][word]),rd[i-1][word])
                clause9_2 = Or(Not(delta[i-1]),r[i-1][word],Not(rd[i-1][word]))
                solver.add(And(clause9_1, clause9_2))

                clause10_1 = Or(delta[i-1],Not(ls[i-1][word]),rd[i-1][word])
                clause10_2 = Or(delta[i-1],ls[i-1][word],Not(rd[i-1][word]))
                solver.add(And(clause10_1, clause10_2))

                clause11_1 = Or(delta[i-1],Not(r[i-1][word]),la[i-1][word])
                clause11_2 = Or(delta[i-1],r[i-1][word],Not(la[i-1][word]))
                solver.add(And(clause11_1, clause11_2))

                solver.add(Or(delta[i-1], Not(delta[i-1])))
                
        epsilon = [Bool(f'epsilon{i}') for i in range(1, self.N_adder+1)]
        ra     = [[Bool(f'ra{i}{w}') for w in range(self.wordlength)] for i in range(1, self.N_adder+1)]


        #xor
        for i in range(1, self.N_adder+1):
            for word in range(self.wordlength):
                clause12 = Or(rd[i-1][word], epsilon[i-1], Not(ra[i-1][word]))
                clause13 = Or(rd[i-1][word], Not(epsilon[i-1]), ra[i-1][word])
                clause14 = Or(Not(rd[i-1][word]), epsilon[i-1], ra[i-1][word])
                clause15 = Or(Not(rd[i-1][word]), Not(epsilon[i-1]), Not(ra[i-1][word]))
                solver.add(clause12)
                solver.add(clause13)
                solver.add(clause14)
                solver.add(clause15)

        
        
        

        #ripple carry
        z     = [[Bool(f'z{i}{w}') for w in range(self.wordlength)] for i in range(1, self.N_adder+1)]
        cout  = [[Bool(f'cout{i}{w}') for w in range(self.wordlength)] for i in range(1, self.N_adder+1)]

        
        for i in range(1, self.N_adder+1):
            # Clauses for sum = a ⊕ b ⊕ cin at 0
            clause16 = Or(la[i-1][0], ra[i-1][0], epsilon[i-1], Not(z[i-1][0]))
            clause17 = Or(la[i-1][0], ra[i-1][0], Not(epsilon[i-1]), z[i-1][0])
            clause18 = Or(la[i-1][0], Not(ra[i-1][0]), epsilon[i-1], z[i-1][0])
            clause19 = Or(Not(la[i-1][0]), ra[i-1][0], epsilon[i-1], z[i-1][0])
            clause20 = Or(Not(la[i-1][0]), Not(ra[i-1][0]), Not(epsilon[i-1]), z[i-1][0])
            clause21 = Or(Not(la[i-1][0]), Not(ra[i-1][0]), epsilon[i-1], Not(z[i-1][0]))
            clause22 = Or(Not(la[i-1][0]), ra[i-1][0], Not(epsilon[i-1]), Not(z[i-1][0]))
            clause23 = Or(la[i-1][0], Not(ra[i-1][0]), Not(epsilon[i-1]), Not(z[i-1][0]))

            solver.add(clause16)
            solver.add(clause17)
            solver.add(clause18)
            solver.add(clause19)
            solver.add(clause20)
            solver.add(clause21)
            solver.add(clause22)
            solver.add(clause23)

            # Clauses for cout = (a AND b) OR (cin AND (a ⊕ b))
            clause24 = Or(Not(la[i-1][0]), Not(ra[i-1][0]), cout[i-1][0])
            clause25 = Or(la[i-1][0], ra[i-1][0], Not(cout[i-1][0]))
            clause26 = Or(Not(la[i-1][0]), Not(epsilon[i-1]), cout[i-1][0])
            clause27 = Or(la[i-1][0], epsilon[i-1], Not(cout[i-1][0]))
            clause28 = Or(Not(ra[i-1][0]), Not(epsilon[i-1]), cout[i-1][0])
            clause29 = Or(ra[i-1][0], epsilon[i-1], Not(cout[i-1][0]))

            solver.add(clause24)
            solver.add(clause25)
            solver.add(clause26)
            solver.add(clause27)
            solver.add(clause28)
            solver.add(clause29)

            for kf in range(1, self.wordlength):
                # Clauses for sum = a ⊕ b ⊕ cin at kf
                clause30 = Or(la[i-1][kf], ra[i-1][kf], cout[i-1][kf-1], Not(z[i-1][kf]))
                clause31 = Or(la[i-1][kf], ra[i-1][kf], Not(cout[i-1][kf-1]), z[i-1][kf])
                clause32 = Or(la[i-1][kf], Not(ra[i-1][kf]), cout[i-1][kf-1], z[i-1][kf])
                clause33 = Or(Not(la[i-1][kf]), ra[i-1][kf], cout[i-1][kf-1], z[i-1][kf])
                clause34 = Or(Not(la[i-1][kf]), Not(ra[i-1][kf]), Not(cout[i-1][kf-1]), z[i-1][kf])
                clause35 = Or(Not(la[i-1][kf]), Not(ra[i-1][kf]), cout[i-1][kf-1], Not(z[i-1][kf]))
                clause36 = Or(Not(la[i-1][kf]), ra[i-1][kf], Not(cout[i-1][kf-1]), Not(z[i-1][kf]))
                clause37 = Or(la[i-1][kf], Not(ra[i-1][kf]), Not(cout[i-1][kf-1]), Not(z[i-1][kf]))

                solver.add(clause30)
                solver.add(clause31)
                solver.add(clause32)
                solver.add(clause33)
                solver.add(clause34)
                solver.add(clause35)
                solver.add(clause36)
                solver.add(clause37)

                # Clauses for cout = (a AND b) OR (cin AND (a ⊕ b)) at kf
                clause38 = Or(Not(la[i-1][kf]), Not(ra[i-1][kf]), cout[i-1][kf])
                clause39 = Or(la[i-1][kf], ra[i-1][kf], Not(cout[i-1][kf]))
                clause40 = Or(Not(la[i-1][kf]), Not(cout[i-1][kf-1]), cout[i-1][kf])
                clause41 = Or(la[i-1][kf], cout[i-1][kf-1], Not(cout[i-1][kf]))
                clause42 = Or(Not(ra[i-1][kf]), Not(cout[i-1][kf-1]), cout[i-1][kf])
                clause43 = Or(ra[i-1][kf], cout[i-1][kf-1], Not(cout[i-1][kf]))

                solver.add(clause38)
                solver.add(clause39)
                solver.add(clause40)
                solver.add(clause41)
                solver.add(clause42)
                solver.add(clause43)

            clause44 = Or(epsilon[i-1], la[i-1][self.wordlength-1], rd[i-1][self.wordlength-1], Not(z[i-1][self.wordlength-1]))
            clause45 = Or(epsilon[i-1], Not(la[i-1][self.wordlength-1]), Not(rd[i-1][self.wordlength-1]), z[i-1][self.wordlength-1])
            clause46 = Or(Not(epsilon[i-1]), la[i-1][self.wordlength-1], Not(rd[i-1][self.wordlength-1]), Not(z[i-1][self.wordlength-1]))
            clause47 = Or(Not(epsilon[i-1]), Not(la[i-1][self.wordlength-1]), rd[i-1][self.wordlength-1], z[i-1][self.wordlength-1])

            solver.add(clause44)
            solver.add(clause45)
            solver.add(clause46)
            solver.add(clause47)


        #right shift
        zeta = [[Bool(f'zeta{i}{k}') for k in range(self.wordlength-1)] for i in range(1, self.N_adder+1)]



        for i in range(1, self.N_adder+1):
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



        
        
        #solver connection
        c_res = [[Bool(f'c_res{m}_{w}') for w in range(self.wordlength)] for m in range(self.half_order+1)]
        t = [[Bool(f't{i}_{m}') for m in range(self.half_order+1)] for i in range(1, self.N_adder+1)]
        e = [Bool(f'e{m}') for m in range(self.half_order+1)]

        e_sum = []
        for m in range(self.half_order+1):
            t_or_clauses=[]

            for i in range(1, self.N_adder+1):
                for word in range(self.wordlength):
                    clause52_1=Or(Not(t[i-1][m]), Not(e[m]), Not(c[i][word]),c_res[m][word])
                    clause52_2=Or(Not(t[i-1][m]), Not(e[m]), c[i][word],Not(c_res[m][word]))
                    solver.add(And(clause52_1, clause52_2))

                t_or_clauses.append(t[i-1][m])
            solver.add(Or(t_or_clauses))

            e_sum.append((e[m],1))
        
        solver.add(PbEq(e_sum,self.connected_coefficient))


        h0 = [Bool(f'h0{m}') for m in range(self.half_order+1)]

        #h0 init
        for m in range(self.half_order+1):
            h_or_clause=[]
            for w in range(self.wordlength):
                h_or_clause.append(h[m][w])
            h_or_clause.append(h0[m])
            solver.add(Or(h_or_clause))

        #Left Shifter for result
        #k is the shift selector
        gamma_res = [[Bool(f'gamma_res{i}{k}') for k in range(self.wordlength-1)] for i in range(self.half_order+1)]

        for m in range(self.half_order+1):
            gamma_sum = []
            for k in range(self.wordlength-1):
                for j in range(self.wordlength-1-k):
                    clause3_1 = Or(Not(gamma_res[m][k]),Not(c_res[m][j]),h[m][j+k])
                    clause3_2 = Or(Not(gamma_res[m][k]),c_res[m][j],Not(h[m][j+k]))
                    solver.add(And(clause3_1, clause3_2))

                gamma_sum.append((gamma_res[m][k], 1))
            
            solver.add(PbEq(gamma_sum,1))


            for kf in range(1,self.wordlength-1):
                for b in range(kf):
                    clause4 = Or(Not(gamma_res[m][kf]),Not(h[m][b]))
                    clause5 = Or(Not(gamma_res[m][kf]), Not(c_res[m][self.wordlength-1]), c_res[m][self.wordlength-2-b])
                    clause6 = Or(Not(gamma_res[m][kf]), c_res[m][self.wordlength-1], Not(c_res[m][self.wordlength-2-b]))
                    solver.add(clause4)
                    solver.add(clause5)
                    solver.add(clause6)

            clause7_1= Or(Not(c_res[m][self.wordlength-1]), h[m][self.wordlength-1])
            clause7_2= Or(c_res[m][self.wordlength-1], Not(h[m][self.wordlength-1]))
            solver.add(And(clause7_1, clause7_2))
        
        start_time=time.time()

        print("solver runing")  


        # print(filter_coeffs)
        # print(filter_literals)

        if solver.check() == sat:
            print("solver sat")
            model = solver.model()
            print(model)
            end_time = time.time()
            
            for a in range(self.half_order + 1):
                fir_coef = 0
                for w in range(self.wordlength):
                    # Evaluate the boolean value from the model
                    bool_value = model.eval(h[a][w], model_completion=True)
                    # Convert boolean to integer (0 or 1) and calculate the term
                    if w==self.wordlength-1:
                        fir_coef += -2**w * (1 if bool_value else 0)
                    else:                    
                        fir_coef += 2**w * (1 if bool_value else 0)
                
                self.h_int_res.append(fir_coef)
            print("FIR Coeffs calculated: ",self.h_int_res)
            
            gain_coef=0
            for g in range(self.gain_wordlength):
                # Evaluate the boolean value from the model
                bool_value = model.eval(gain[g], model_completion=True)
                # Convert boolean to integer (0 or 1) and calculate the term
                gain_coef += 2**g * (1 if bool_value else 0)
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
        self.freq_upper_lin = ((freq_upper_lin_array * (self.gain_res/10**self.gain_accuracy)) / 10**(self.filter_accuracy-self.gain_accuracy)).tolist()
        self.freq_lower_lin = ((freq_lower_lin_array * (self.gain_res/10**self.gain_accuracy)) / 10**(self.filter_accuracy-self.gain_accuracy)).tolist()


        #plot input
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
            for j in range(self.half_order+1):
                cm_const = sf.cm_handler(j, omega)
                term_sum_exprs += self.h_int_res[j] * cm_const
            
            # Append the computed sum expression to the frequency response list
            computed_frequency_response.append(np.abs(term_sum_exprs))
        
        # Normalize frequencies to range from 0 to 1 for plotting purposes

        # Plot the computed frequency response
        #self.ax1.plot([x/1 for x in self.freqx_axis], computed_frequency_response, color='green', label='Computed Frequency Response')

        self.ax2.set_ylim(-10,10)


        if self.app:
            self.app.canvas.draw()



    

    
# Test inputs
filter_type = 0
order_upper = 5
accuracy = 5
adder_count = 4


# Initialize freq_upper and freq_lower with NaN values
freqx_axis = np.linspace(0, 1, accuracy*order_upper) #according to Mr. Kumms paper
freq_upper = np.full(accuracy * order_upper, np.nan)
freq_lower = np.full(accuracy * order_upper, np.nan)

# Manually set specific values for the elements of freq_upper and freq_lower in dB
freq_upper[1:5] = 10
freq_lower[1:5] = -2

# freq_upper[50:100] = -3
# freq_lower[50:100] = -1000



#beyond this bound lowerbound will be ignored
ignore_lowerbound_lin = 0.0001

# Create FIRFilter instance
fir_filter = FIRFilter(filter_type, order_upper, freqx_axis, freq_upper, freq_lower, ignore_lowerbound_lin, adder_count)

# Run solver and plot result
fir_filter.runsolver()
fir_filter.plot_result(fir_filter.h_int_res)
fir_filter.plot_validation()

# Show plot
plt.show()

