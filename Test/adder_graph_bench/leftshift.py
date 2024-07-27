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

        # Input multiplexer
        self.l = [[Bool(f'l{i}{w}') for w in range(self.wordlength)] for i in range(1, self.N+1)]

        self.solver = Solver()

    
        # Left Shifter
        self.gamma = [[Bool(f'gamma{i}{k}') for k in range(self.wordlength-1)] for i in range(1, self.N+1)]
        self.s = [[Bool(f's{i}{w}') for w in range(self.wordlength)] for i in range(1, self.N+1)]

        # Test case
        for i in range(1, self.N+1):
            self.solver.add(self.l[i-1][4])
            self.solver.add(Not(self.l[i-1][5]))

            self.solver.add(self.gamma[i-1][4])

        for i in range(1, self.N+1):
            gamma_sum = []
            for k in range(self.wordlength-1):
                for j in range(self.wordlength-1-k):
                    clause3_1 = Or(Not(self.gamma[i-1][k]), Not(self.l[i-1][j]), self.s[i-1][j+k])
                    clause3_2 = Or(Not(self.gamma[i-1][k]), self.l[i-1][j], Not(self.s[i-1][j+k]))
                    self.solver.add(And(clause3_1, clause3_2))

                gamma_sum.append((self.gamma[i-1][k], 1))

            self.solver.add(PbEq(gamma_sum, 1))

            for kf in range(1, self.wordlength-1):
                for b in range(kf):
                    clause4 = Or(Not(self.gamma[i-1][kf]), Not(self.s[i-1][b]))
                    clause5 = Or(Not(self.gamma[i-1][kf]), Not(self.l[i-1][self.wordlength-1]), self.l[i-1][self.wordlength-2-b])
                    clause6 = Or(Not(self.gamma[i-1][kf]), self.l[i-1][self.wordlength-1], Not(self.l[i-1][self.wordlength-2-b]))
                    self.solver.add(clause4)
                    self.solver.add(clause5)
                    self.solver.add(clause6)

            clause7_1 = Or(Not(self.l[i-1][self.wordlength-1]), self.s[i-1][self.wordlength-1])
            clause7_2 = Or(self.l[i-1][self.wordlength-1], Not(self.s[i-1][self.wordlength-1]))
            self.solver.add(And(clause7_1, clause7_2))

    def run(self):
        if self.solver.check() == sat:
            print("its sat")
            model = self.solver.model()
            self.print_model(model)
        else:
            print("No solution")

    def print_model(self, model):
        print("Model:")
        for i in range(len(self.l)):
            for w in range(self.wordlength):
                print(f'l[{i+1}][{w}] = {model[self.l[i][w]]}')

        for i in range(len(self.s)):
            for w in range(self.wordlength):
                print(f's[{i+1}][{w}] = {model[self.s[i][w]]}')

        for i in range(len(self.gamma)):
            for k in range(self.wordlength-1):
                print(f'gamma[{i+1}][{k}] = {model[self.gamma[i][k]]}')

if __name__ == '__main__':
    hm = (25, 23, 11, 25, 75)
    wordlength = 6  # min wordlength would be 2
    bitshift_instance = bitshift(hm, wordlength, verbose=True)
    bitshift_instance.run()
