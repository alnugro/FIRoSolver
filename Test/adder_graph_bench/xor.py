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

        self.w = [[Bool(f'w{i}{w}') for w in range(self.wordlength)] for i in range(1, self.N+1)]
        self.epsilon = [Bool(f'epsilon{i}') for i in range(1, self.N+1)]
        self.y = [[Bool(f'y{i}{w}') for w in range(self.wordlength)] for i in range(1, self.N+1)]

        for i in range(1, self.N+1):
            solver.add(self.epsilon[i-1])
            solver.add(self.w[i-1][2])
            solver.add(Not(self.w[i-1][3]))
            solver.add(self.w[i-1][4])

        # XOR
        for i in range(1, self.N+1):
            for word in range(self.wordlength):
                clause12 = Or(self.w[i-1][word], self.epsilon[i-1], Not(self.y[i-1][word]))
                clause13 = Or(self.w[i-1][word], Not(self.epsilon[i-1]), self.y[i-1][word])
                clause14 = Or(Not(self.w[i-1][word]), self.epsilon[i-1], self.y[i-1][word])
                clause15 = Or(Not(self.w[i-1][word]), Not(self.epsilon[i-1]), Not(self.y[i-1][word]))
                solver.add(clause12)
                solver.add(clause13)
                solver.add(clause14)
                solver.add(clause15)

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
        for i in range(len(self.w)):
            for w in range(self.wordlength):
                print(f'w[{i+1}][{w}] = {model[self.w[i][w]]}')

        for i in range(len(self.epsilon)):
            print(f'epsilon[{i+1}] = {model[self.epsilon[i]]}')

        for i in range(len(self.y)):
            for w in range(self.wordlength):
                print(f'y[{i+1}][{w}] = {model[self.y[i][w]]}')

if __name__ == '__main__':
    hm = (25, 23, 11, 25, 75)
    wordlength = 6  # min wordlength would be 2
    bitshift_instance = bitshift(hm, wordlength, verbose=True)
    bitshift_instance.run()
