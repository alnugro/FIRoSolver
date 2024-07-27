import numpy as np
from z3 import *
import matplotlib.pyplot as plt
import time

class bitshift():
    def __init__(self, out, wordlength, verbose=False):
        self.wordlength = wordlength
        self.out = out
        self.N = 4
        self.verbose = verbose

        solver = Solver()

        self.x = [[Bool(f'x{i}{w}') for w in range(self.wordlength)] for i in range(1, self.N+1)]
        self.epsilon = [Bool(f'epsilon{i}') for i in range(1, self.N+1)]
        self.y = [[Bool(f'y{i}{w}') for w in range(self.wordlength)] for i in range(1, self.N+1)]

        # Ripple carry
        self.z = [[Bool(f'z{i}{w}') for w in range(self.wordlength)] for i in range(1, self.N+1)]
        self.cout = [[Bool(f'cout{i}{w}') for w in range(self.wordlength)] for i in range(1, self.N+1)]



        self.u = [[Bool(f'w{i}{w}') for w in range(self.wordlength)] for i in range(1, self.N+1)]

        #test case
        for i in range(1, self.N+1):
            solver.add(Not(self.epsilon[i-1]))
            solver.add(Not(self.u[i-1][2]))
            solver.add(Not(self.u[i-1][1]))
            solver.add(self.u[i-1][0])

            solver.add(self.x[i-1][0])
            solver.add(Not(self.x[i-1][1]))
            solver.add(Not(self.x[i-1][2]))


        # XOR
        for i in range(1, self.N+1):
            for word in range(self.wordlength):
                clause12 = Or(self.u[i-1][word], self.epsilon[i-1], Not(self.y[i-1][word]))
                clause13 = Or(self.u[i-1][word], Not(self.epsilon[i-1]), self.y[i-1][word])
                clause14 = Or(Not(self.u[i-1][word]), self.epsilon[i-1], self.y[i-1][word])
                clause15 = Or(Not(self.u[i-1][word]), Not(self.epsilon[i-1]), Not(self.y[i-1][word]))
                solver.add(clause12)
                solver.add(clause13)
                solver.add(clause14)
                solver.add(clause15)
        


        for i in range(1, self.N+1):
            # Clauses for sum = a ⊕ b ⊕ cin at 0
            clause16 = Or(self.x[i-1][0], self.y[i-1][0], self.epsilon[i-1], Not(self.z[i-1][0]))
            clause17 = Or(self.x[i-1][0], self.y[i-1][0], Not(self.epsilon[i-1]), self.z[i-1][0])
            clause18 = Or(self.x[i-1][0], Not(self.y[i-1][0]), self.epsilon[i-1], self.z[i-1][0])
            clause19 = Or(Not(self.x[i-1][0]), self.y[i-1][0], self.epsilon[i-1], self.z[i-1][0])
            clause20 = Or(Not(self.x[i-1][0]), Not(self.y[i-1][0]), Not(self.epsilon[i-1]), self.z[i-1][0])
            clause21 = Or(Not(self.x[i-1][0]), Not(self.y[i-1][0]), self.epsilon[i-1], Not(self.z[i-1][0]))
            clause22 = Or(Not(self.x[i-1][0]), self.y[i-1][0], Not(self.epsilon[i-1]), Not(self.z[i-1][0]))
            clause23 = Or(self.x[i-1][0], Not(self.y[i-1][0]), Not(self.epsilon[i-1]), Not(self.z[i-1][0]))

            solver.add(clause16)
            solver.add(clause17)
            solver.add(clause18)
            solver.add(clause19)
            solver.add(clause20)
            solver.add(clause21)
            solver.add(clause22)
            solver.add(clause23)

            # Clauses for cout = (a AND b) OR (cin AND (a ⊕ b))
            clause24 = Or(Not(self.x[i-1][0]), Not(self.y[i-1][0]), self.cout[i-1][0])
            clause25 = Or(self.x[i-1][0], self.y[i-1][0], Not(self.cout[i-1][0]))
            clause26 = Or(Not(self.x[i-1][0]), Not(self.epsilon[i-1]), self.cout[i-1][0])
            clause27 = Or(self.x[i-1][0], self.epsilon[i-1], Not(self.cout[i-1][0]))
            clause28 = Or(Not(self.y[i-1][0]), Not(self.epsilon[i-1]), self.cout[i-1][0])
            clause29 = Or(self.y[i-1][0], self.epsilon[i-1], Not(self.cout[i-1][0]))

            solver.add(clause24)
            solver.add(clause25)
            solver.add(clause26)
            solver.add(clause27)
            solver.add(clause28)
            solver.add(clause29)

            for kf in range(1, self.wordlength):
                # Clauses for sum = a ⊕ b ⊕ cin at kf
                clause30 = Or(self.x[i-1][kf], self.y[i-1][kf], self.cout[i-1][kf-1], Not(self.z[i-1][kf]))
                clause31 = Or(self.x[i-1][kf], self.y[i-1][kf], Not(self.cout[i-1][kf-1]), self.z[i-1][kf])
                clause32 = Or(self.x[i-1][kf], Not(self.y[i-1][kf]), self.cout[i-1][kf-1], self.z[i-1][kf])
                clause33 = Or(Not(self.x[i-1][kf]), self.y[i-1][kf], self.cout[i-1][kf-1], self.z[i-1][kf])
                clause34 = Or(Not(self.x[i-1][kf]), Not(self.y[i-1][kf]), Not(self.cout[i-1][kf-1]), self.z[i-1][kf])
                clause35 = Or(Not(self.x[i-1][kf]), Not(self.y[i-1][kf]), self.cout[i-1][kf-1], Not(self.z[i-1][kf]))
                clause36 = Or(Not(self.x[i-1][kf]), self.y[i-1][kf], Not(self.cout[i-1][kf-1]), Not(self.z[i-1][kf]))
                clause37 = Or(self.x[i-1][kf], Not(self.y[i-1][kf]), Not(self.cout[i-1][kf-1]), Not(self.z[i-1][kf]))

                solver.add(clause30)
                solver.add(clause31)
                solver.add(clause32)
                solver.add(clause33)
                solver.add(clause34)
                solver.add(clause35)
                solver.add(clause36)
                solver.add(clause37)

                # Clauses for cout = (a AND b) OR (cin AND (a ⊕ b)) at kf
                clause38 = Or(Not(self.x[i-1][kf]), Not(self.y[i-1][kf]), self.cout[i-1][kf])
                clause39 = Or(self.x[i-1][kf], self.y[i-1][kf], Not(self.cout[i-1][kf]))
                clause40 = Or(Not(self.x[i-1][kf]), Not(self.cout[i-1][kf-1]), self.cout[i-1][kf])
                clause41 = Or(self.x[i-1][kf], self.cout[i-1][kf-1], Not(self.cout[i-1][kf]))
                clause42 = Or(Not(self.y[i-1][kf]), Not(self.cout[i-1][kf-1]), self.cout[i-1][kf])
                clause43 = Or(self.y[i-1][kf], self.cout[i-1][kf-1], Not(self.cout[i-1][kf]))

                solver.add(clause38)
                solver.add(clause39)
                solver.add(clause40)
                solver.add(clause41)
                solver.add(clause42)
                solver.add(clause43)

            clause44 = Or(self.epsilon[i-1], self.x[i-1][self.wordlength-1], self.u[i-1][self.wordlength-1], Not(self.z[i-1][self.wordlength-1]))
            clause45 = Or(self.epsilon[i-1], Not(self.x[i-1][self.wordlength-1]), Not(self.u[i-1][self.wordlength-1]), self.z[i-1][self.wordlength-1])
            clause46 = Or(Not(self.epsilon[i-1]), self.x[i-1][self.wordlength-1], Not(self.u[i-1][self.wordlength-1]), Not(self.z[i-1][self.wordlength-1]))
            clause47 = Or(Not(self.epsilon[i-1]), Not(self.x[i-1][self.wordlength-1]), self.u[i-1][self.wordlength-1], self.z[i-1][self.wordlength-1])

            solver.add(clause44)
            solver.add(clause45)
            solver.add(clause46)
            solver.add(clause47)

        self.solver = solver

    def run(self):
        if self.solver.check() == sat:
            print("its sat")
            model = self.solver.model()
            self.print_model(model)
        else:
            print("No solution")

    def print_model(self, model):
        print("Model:")
    
        for w in range(self.wordlength):
            print(f'x[{0}][{w}] = {model[self.x[0][w]]}')

        print(f'epsilon[{0}] = {model[self.epsilon[0]]}')

        for word in range(self.wordlength):
            print(f'u[{0}][{word}] = {model[self.u[0][word]]}')
        
        for w in range(self.wordlength):
            print(f'y[{0}][{w}] = {model[self.y[0][w]]}')

        for w in range(self.wordlength):
            print(f'z[{0}][{w}] = {model[self.z[0][w]]}')

        for w in range(self.wordlength):
            print(f'cout[{0}][{w}] = {model[self.cout[0][w]]}')

if __name__ == '__main__':
    hm = (25, 23, 11, 25, 75)
    wordlength = 3  # min wordlength would be 2
    bitshift_instance = bitshift(hm, wordlength, verbose=True)
    bitshift_instance.run()
