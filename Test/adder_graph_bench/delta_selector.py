import numpy as np
from z3 import *
import matplotlib.pyplot as plt
import time

class bitshift():
    def __init__(self, out, wordlength, verbose=False):
        self.wordlength = wordlength
        self.out = out
        self.N = 1
        self.verbose = verbose

        solver = Solver()

        # Left Shifter
        self.s = [[Bool(f's{i}{w}') for w in range(self.wordlength)] for i in range(1, self.N+1)]
        self.r = [[Bool(f'r{i}{w}') for w in range(self.wordlength)] for i in range(1, self.N+1)]

        self.delta = [Bool(f'delta{i}') for i in range(1, self.N+1)]
        self.w = [[Bool(f'w{i}{w}') for w in range(self.wordlength)] for i in range(1, self.N+1)]
        self.x = [[Bool(f'x{i}{w}') for w in range(self.wordlength)] for i in range(1, self.N+1)]

        # Test bench
        for i in range(1, self.N+1):
            solver.add(self.s[i-1][0])
            solver.add(self.r[i-1][1])

        # Delta selector
        for i in range(1, self.N+1):
            for word in range(self.wordlength):
                clause8_1 = Or(Not(self.delta[i-1]), Not(self.s[i-1][word]), self.x[i-1][word])
                clause8_2 = Or(Not(self.delta[i-1]), self.s[i-1][word], Not(self.x[i-1][word]))
                solver.add(And(clause8_1, clause8_2))

                clause9_1 = Or(Not(self.delta[i-1]), Not(self.r[i-1][word]), self.w[i-1][word])
                clause9_2 = Or(Not(self.delta[i-1]), self.r[i-1][word], Not(self.w[i-1][word]))
                solver.add(And(clause9_1, clause9_2))

                clause10_1 = Or(self.delta[i-1], Not(self.s[i-1][word]), self.w[i-1][word])
                clause10_2 = Or(self.delta[i-1], self.s[i-1][word], Not(self.w[i-1][word]))
                solver.add(And(clause10_1, clause10_2))

                clause11_1 = Or(self.delta[i-1], Not(self.r[i-1][word]), self.x[i-1][word])
                clause11_2 = Or(self.delta[i-1], self.r[i-1][word], Not(self.x[i-1][word]))
                solver.add(And(clause11_1, clause11_2))

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
        for i in range(len(self.s)):
            for w in range(self.wordlength):
                print(f's[{i+1}][{w}] = {model[self.s[i][w]]}')

        for i in range(len(self.r)):
            for w in range(self.wordlength):
                print(f'r[{i+1}][{w}] = {model[self.r[i][w]]}')

        for i in range(len(self.delta)):
            print(f'delta[{i+1}] = {model[self.delta[i]]}')

        for i in range(len(self.w)):
            for w in range(self.wordlength):
                print(f'w[{i+1}][{w}] = {model[self.w[i][w]]}')

        for i in range(len(self.x)):
            for w in range(self.wordlength):
                print(f'x[{i+1}][{w}] = {model[self.x[i][w]]}')

if __name__ == '__main__':
    hm = (25, 23, 11, 25, 75)
    wordlength = 3  # min wordlength would be 2
    bitshift_instance = bitshift(hm, wordlength, verbose=True)
    bitshift_instance.run()
