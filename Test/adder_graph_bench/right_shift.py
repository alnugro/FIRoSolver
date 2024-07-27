import numpy as np
from z3 import *
import matplotlib.pyplot as plt
import time

class bitshift():
    def __init__(self, out, wordlength, verbose=False):
        self.wordlength = wordlength
        self.out = out
        self.N = 6
        self.verbose = verbose

        # Input multiplexer
        c = [[Bool(f'c{i}{w}') for w in range(self.wordlength)] for i in range(self.N+1)]
        solver = Solver()

        # Ripple carry
        z = [[Bool(f'z{i}{w}') for w in range(self.wordlength)] for i in range(1, self.N+1)]

        # Right shift
        zeta = [[Bool(f'zeta{i}{k}') for k in range(self.wordlength-1)] for i in range(1, self.N+1)]

        # Test case
        for i in range(1, self.N+1):
            solver.add(z[i-1][5])
            solver.add(zeta[i-1][2])

        for i in range(1, self.N+1):
            zeta_sum = []
            for k in range(self.wordlength-1):
                for j in range(self.wordlength-1-k):
                    clause48_1 = Or(Not(zeta[i-1][k]), Not(z[i-1][j+k]), c[i][j])
                    clause48_2 = Or(Not(zeta[i-1][k]), z[i-1][j+k], Not(c[i][j]))
                    solver.add(And(clause48_1, clause48_2))

                zeta_sum.append((zeta[i-1][k], 1))

            solver.add(PbEq(zeta_sum, 1))

            for kf in range(1, self.wordlength-1):
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

            # Bound c[i][0] to be an odd number
            solver.add(c[i][0])

        self.solver = solver
        self.c = c
        self.z = z
        self.zeta = zeta

    def run(self):
        if self.solver.check() == sat:
            print("its sat")
            model = self.solver.model()
            self.print_model(model)
        else:
            print("No solution")

    def print_model(self, model):
        print("Model:")
        for i in range(len(self.c)):
            for w in range(self.wordlength):
                print(f'c[{i}][{w}] = {model[self.c[i][w]]}')

        for i in range(len(self.z)):
            for w in range(self.wordlength):
                print(f'z[{i+1}][{w}] = {model[self.z[i][w]]}')

        for i in range(len(self.zeta)):
            for k in range(self.wordlength-1):
                print(f'zeta[{i+1}][{k}] = {model[self.zeta[i][k]]}')

if __name__ == '__main__':
    hm = (25, 23, 11, 25, 75)
    wordlength = 6  # min wordlength would be 2
    bitshift_instance = bitshift(hm, wordlength, verbose=True)
    bitshift_instance.run()
