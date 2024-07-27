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
        self.c = [[Bool(f'c{i}{w}') for w in range(self.wordlength)] for i in range(self.N+1)]
        self.l = [[Bool(f'l{i}{w}') for w in range(self.wordlength)] for i in range(1, self.N+1)]
        self.r = [[Bool(f'r{i}{w}') for w in range(self.wordlength)] for i in range(1, self.N+1)]

        self.alpha = [[Bool(f'alpha{i}{a}') for a in range(i)] for i in range(1, self.N+1)]
        self.beta = [[Bool(f'Beta{i}{a}') for a in range(i)] for i in range(1, self.N+1)] 

        self.solver = Solver()

        # c 0 is always 1
        for w in range(self.wordlength):
            if w == 2:
                self.solver.add(Not(self.c[0][w]))
                continue

            self.solver.add(self.c[0][w])

        # Input multiplexer
        for i in range(1, self.N+1):
            alpha_sum = []
            beta_sum = []
            for a in range(i):
                for word in range(self.wordlength):
                    clause1_1 = Or(Not(self.alpha[i-1][a]), Not(self.c[a][word]), self.l[i-1][word])
                    clause1_2 = Or(Not(self.alpha[i-1][a]), self.c[a][word], Not(self.l[i-1][word]))
                    self.solver.add(And(clause1_1, clause1_2))

                    clause2_1 = Or(Not(self.beta[i-1][a]), Not(self.c[a][word]), self.r[i-1][word])
                    clause2_2 = Or(Not(self.beta[i-1][a]), self.c[a][word], Not(self.r[i-1][word]))
                    self.solver.add(And(clause2_1, clause2_2))

                alpha_sum.append((self.alpha[i-1][a], 1))
                beta_sum.append((self.beta[i-1][a], 1))

            self.solver.add(PbEq(alpha_sum, 1))
            self.solver.add(PbEq(beta_sum, 1))

        for i in range(1, self.N+1):
            self.solver.add(self.alpha[i-1][0])
            self.solver.add(self.beta[i-1][0])

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

        for i in range(len(self.l)):
            for w in range(self.wordlength):
                print(f'l[{i+1}][{w}] = {model[self.l[i][w]]}')

        for i in range(len(self.r)):
            for w in range(self.wordlength):
                print(f'r[{i+1}][{w}] = {model[self.r[i][w]]}')

        for i in range(len(self.alpha)):
            for a in range(len(self.alpha[i])):
                print(f'alpha[{i+1}][{a}] = {model[self.alpha[i][a]]}')

        for i in range(len(self.beta)):
            for a in range(len(self.beta[i])):
                print(f'beta[{i+1}][{a}] = {model[self.beta[i][a]]}')

if __name__ == '__main__':
    hm = (25, 23, 11, 25, 75)
    wordlength = 6  # min wordlength would be 2
    bitshift_instance = bitshift(hm, wordlength, verbose=True)
    bitshift_instance.run()
