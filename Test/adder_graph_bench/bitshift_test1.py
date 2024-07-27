import numpy as np
from z3 import *
import matplotlib.pyplot as plt
import time

class bitshift():
    def __init__(self, out, wordlength, verbose=False):
        self.wordlength = wordlength
        self.out = out
        self.N = 3
        self.verbose = verbose


        #input multiplexer
        c=[[Bool(f'c{i}{w}') for w in range(self.wordlength)] for i in range(self.N+1)]
        l=[[Bool(f'l{i}{w}') for w in range(self.wordlength)] for i in range(1, self.N+1)]
        r=[[Bool(f'r{i}{w}') for w in range(self.wordlength)] for i in range(1, self.N+1)]


        alpha = [[Bool(f'alpha{i}{a}') for a in range(i)] for i in range(1, self.N+1)]
        beta =[[ Bool(f'Beta{i}{a}') for a in range(i)] for i in range(1, self.N+1)] 

        solver = Solver()


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
        


        if solver.check() == sat:
            print("its sat")
            model = solver.model()
            for i in range(len(c)):
                for w in range(wordlength):
                    print(f'c[{i}][{w}] = {model[c[i][w]]}')

            for i in range(len(l)):
                for w in range(wordlength):
                    print(f'l[{i+1}][{w}] = {model[l[i][w]]}')

            for i in range(len(r)):
                for w in range(wordlength):
                    print(f'r[{i+1}][{w}] = {model[r[i][w]]}')

            for i in range(len(alpha)):
                for a in range(len(alpha[i])):
                    print(f'alpha[{i+1}][{a}] = {model[alpha[i][a]]}')

            for i in range(len(beta)):
                for a in range(len(beta[i])):
                    print(f'beta[{i+1}][{a}] = {model[beta[i][a]]}')

            for i in range(len(gamma)):
                for k in range(wordlength-1):
                    print(f'gamma[{i+1}][{k}] = {model[gamma[i][k]]}')

            for i in range(len(s)):
                for w in range(wordlength):
                    print(f's[{i+1}][{w}] = {model[s[i][w]]}')

            for i in range(len(delta)):
                print(f'delta[{i+1}] = {model[delta[i]]}')

            for i in range(len(u)):
                for w_idx in range(wordlength):
                    print(f'u[{i+1}][{w_idx}] = {model[u[i][w_idx]]}')

            for i in range(len(x)):
                for w_idx in range(wordlength):
                    print(f'x[{i+1}][{w_idx}] = {model[x[i][w_idx]]}')

            for i in range(len(epsilon)):
                print(f'epsilon[{i+1}] = {model[epsilon[i]]}')

            for i in range(len(y)):
                for w in range(wordlength):
                    print(f'y[{i+1}][{w}] = {model[y[i][w]]}')

            for i in range(len(z)):
                for w in range(wordlength):
                    print(f'z[{i+1}][{w}] = {model[z[i][w]]}')

            for i in range(len(cout)):
                for w in range(wordlength):
                    print(f'cout[{i+1}][{w}] = {model[cout[i][w]]}')

            for i in range(len(zeta)):
                for k in range(wordlength-1):
                    print(f'zeta[{i+1}][{k}] = {model[zeta[i][k]]}')
        else:
            print("No solution")





if __name__ == '__main__':
    hm = (25, 23, 11,25,75)
    wordlength = 6 #min wordlength would be 2
    bitshift(hm, wordlength, verbose=True)
