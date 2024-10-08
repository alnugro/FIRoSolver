import numpy as np
from z3 import *
import matplotlib.pyplot as plt
import time

class bitshift():
    def __init__(self, wordlength, verbose=False):
        self.wordlength = wordlength
        self.N_adder = 15
        self.verbose = verbose


        #input multiplexer
        c=[[Bool(f'c{i}{w}') for w in range(self.wordlength)] for i in range(self.N_adder+1)]
        l=[[Bool(f'l{i}{w}') for w in range(self.wordlength)] for i in range(1, self.N_adder+1)]
        r=[[Bool(f'r{i}{w}') for w in range(self.wordlength)] for i in range(1, self.N_adder+1)]


        alpha = [[Bool(f'alpha{i}{a}') for a in range(i)] for i in range(1, self.N_adder+1)]
        beta =[[ Bool(f'Beta{i}{a}') for a in range(i)] for i in range(1, self.N_adder+1)] 

        solver = Solver()


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
        ls     = [[Bool(f's{i}{w}') for w in range(self.wordlength)] for i in range(1, self.N_adder+1)]


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
        rd     = [[Bool(f'u{i}{w}') for w in range(self.wordlength)] for i in range(1, self.N_adder+1)]
        la     = [[Bool(f'x{i}{w}') for w in range(self.wordlength)] for i in range(1, self.N_adder+1)]

   
    
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
        ra     = [[Bool(f'y{i}{w}') for w in range(self.wordlength)] for i in range(1, self.N_adder+1)]


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



        half_order = (50 // 2)
        connected_coefficient = half_order+1
        
        
        #solver connection
        c_res = [[Bool(f'c_res{m}_{w}') for w in range(self.wordlength)] for m in range(half_order+1)]
        t = [[Bool(f't{i}_{m}') for m in range(half_order+1)] for i in range(1, self.N_adder+1)]
        e = [Bool(f'e{m}') for m in range(half_order+1)]

        e_sum = []
        for m in range(half_order+1):
            t_or_clauses=[]

            for i in range(1, self.N_adder+1):
                for word in range(self.wordlength):
                    clause52_1=Or(Not(t[i-1][m]), Not(e[m]), Not(c[i][word]),c_res[m][word])
                    clause52_2=Or(Not(t[i-1][m]), Not(e[m]), c[i][word],Not(c_res[m][word]))
                    solver.add(And(clause52_1, clause52_2))

                t_or_clauses.append(t[i-1][m])
            solver.add(Or(t_or_clauses))

            e_sum.append((e[m],1))
        
        solver.add(PbEq(e_sum,connected_coefficient))


        h = [[Bool(f'h{m}_{w}') for w in range(self.wordlength)] for m in range(half_order+1)]
        h0 = [Bool(f'h0{m}') for m in range(half_order+1)]

        #h0 init
        for m in range(half_order+1):
            h_or_clause=[]
            for w in range(self.wordlength):
                h_or_clause.append(h[m][w])
            h_or_clause.append(h0[m])
            solver.add(Or(h_or_clause))

        #Left Shifter for result
        #k is the shift selector
        gamma_res = [[Bool(f'gamma_res{i}{k}') for k in range(self.wordlength-1)] for i in range(half_order+1)]

        for m in range(half_order+1):
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

        
        # #test case
        # for m in range(half_order+1):
        #     hm=self.int_to_signed_binary_list(m+3, self.wordlength)
        #     hm_and = []
        #     for w in range(self.wordlength):
        #         if hm[w]==1:
        #             hm_and.append(h[m][w])
        #         else:
        #             hm_and.append(Not(h[m][w]))
        #     print(hm_and)
        #     solver.add(And(hm_and))

 

        


        if solver.check() == sat:
            print("its sat")
            model = solver.model()
            for i in range(len(c)):
                for w in range(wordlength):
                    val = model[c[i][w]]
                    print(f'c[{i}][{w}] = {val if val is not None else "None"}')

            for i in range(len(l)):
                for w in range(wordlength):
                    val = model[l[i][w]]
                    print(f'l[{i+1}][{w}] = {val if val is not None else "None"}')

            for i in range(len(r)):
                for w in range(wordlength):
                    val = model[r[i][w]]
                    print(f'r[{i+1}][{w}] = {val if val is not None else "None"}')

            for i in range(len(alpha)):
                for a in range(len(alpha[i])):
                    val = model[alpha[i][a]]
                    print(f'alpha[{i+1}][{a}] = {val if val is not None else "None"}')

            for i in range(len(beta)):
                for a in range(len(beta[i])):
                    val = model[beta[i][a]]
                    print(f'beta[{i+1}][{a}] = {val if val is not None else "None"}')

            for i in range(len(gamma)):
                for k in range(wordlength-1):
                    val = model[gamma[i][k]]
                    print(f'gamma[{i+1}][{k}] = {val if val is not None else "None"}')

            for i in range(len(ls)):
                for w in range(wordlength):
                    val = model[ls[i][w]]
                    print(f'ls[{i+1}][{w}] = {val if val is not None else "None"}')

            for i in range(len(delta)):
                val = model[delta[i]]
                print(f'delta[{i+1}] = {val if val is not None else "None"}')

            for i in range(len(rd)):
                for w_idx in range(wordlength):
                    val = model[rd[i][w_idx]]
                    print(f'rd[{i+1}][{w_idx}] = {val if val is not None else "None"}')

            for i in range(len(la)):
                for w_idx in range(wordlength):
                    val = model[la[i][w_idx]]
                    print(f'la[{i+1}][{w_idx}] = {val if val is not None else "None"}')

            for i in range(len(epsilon)):
                val = model[epsilon[i]]
                print(f'epsilon[{i+1}] = {val if val is not None else "None"}')

            for i in range(len(ra)):
                for w in range(wordlength):
                    val = model[ra[i][w]]
                    print(f'ra[{i+1}][{w}] = {val if val is not None else "None"}')

            for i in range(len(z)):
                for w in range(wordlength):
                    val = model[z[i][w]]
                    print(f'z[{i+1}][{w}] = {val if val is not None else "None"}')

            for i in range(len(cout)):
                for w in range(wordlength):
                    val = model[cout[i][w]]
                    print(f'cout[{i+1}][{w}] = {val if val is not None else "None"}')

            for i in range(len(zeta)):
                for k in range(wordlength-1):
                    val = model[zeta[i][k]]
                    print(f'zeta[{i+1}][{k}] = {val if val is not None else "None"}')

            for i in range(len(h)):
                for w in range(wordlength):
                    val = model[h[i][w]]
                    print(f'h[{i}][{w}] = {val if val is not None else "None"}')

            for i in range(len(h0)):
                val = model[h0[i]]
                print(f'h0[{i}] = {val if val is not None else "None"}')

            for i in range(len(t)):
                for m in range(half_order+1):
                    val = model[t[i][m]]
                    print(f't[{i+1}][{m}] = {val if val is not None else "None"}')

            for i in range(len(e)):
                val = model[e[i]]
                print(f'e[{i}] = {val if val is not None else "None"}')
        else:
            print("No solution")


    def int_to_signed_binary_list(self, value, word_length):
        # Calculate the range for the word length
        min_val = -(1 << (word_length - 1))
        max_val = (1 << (word_length - 1)) - 1
        
        # Check if the value fits in the given word length
        if value < min_val or value > max_val:
            raise ValueError(f"Value {value} out of range for a {word_length}-bit signed integer")
        
        # Handle negative values using two's complement
        if value < 0:
            value = (1 << word_length) + value

        # Get the binary representation and pad with leading zeros if necessary
        bin_representation = format(value, f'0{word_length}b')
        
        print(bin_representation)
        
        # Convert the binary representation to a list of integers (0 or 1)
        bin_list = [int(bit) for bit in bin_representation]
        
        return bin_list





if __name__ == '__main__':
    wordlength = 6 #min wordlength would be 2
    bitshift(wordlength, verbose=True)
