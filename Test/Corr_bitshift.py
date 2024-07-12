

import numpy as np
from z3 import *
import matplotlib.pyplot as plt
import time

class bitshift():
    def __init__(self, out, wordlength):
        self.wordlength = wordlength
        self.out = out
        self.A_M = 10
        solver = Solver()

        c_a = [Int(f'c_a{i}') for i in range(self.A_M + 1)]
        solver.add(c_a[0] == 1)

        c_sh_sg_a_i = [[Int(f'c_sh_sg_a_i{j}{i}') for i in range(2)] for j in range(self.A_M)]
        for j in range(self.A_M):
            solver.add(c_a[j + 1] == c_sh_sg_a_i[j][0] + c_sh_sg_a_i[j][1])

        c_a_i = [[Int(f'c_a_i{j}{i}') for i in range(2)] for j in range(self.A_M)]
        c_a_i_k = [[[Bool(f'c_a_i_k{j}{i}{k}') for k in range(self.A_M)] for i in range(2)] for j in range(self.A_M)]

        for j in range(self.A_M):
            for i in range(2):
                c_a_i_k_sum = 0
                for k in range(j):
                    solver.add(Implies(c_a_i_k[j][i][k], c_a_i[j][i] == c_a[k]))
                    c_a_i_k_sum += c_a_i_k[j][i][k]
                solver.add(c_a_i_k_sum == 1)

        c_sh_a_i = [[Int(f'c_sh_a_i{j}{i}') for i in range(2)] for j in range(self.A_M)]
        sh_a_i_s = [[[Bool(f'sh_a_i_s{j}{i}{k}') for k in range(2 * self.wordlength + 1)] for i in range(2)] for j in range(self.A_M)]

        for j in range(self.A_M):
            for i in range(2):
                sh_a_i_s_sum = 0
                for k in range(2 * self.wordlength + 1):
                    if k > self.wordlength and i == 0:
                        solver.add(sh_a_i_s[j][i][k] == False)
                    if k < self.wordlength and i == 0:
                        solver.add(sh_a_i_s[j][0][k] == sh_a_i_s[j][1][k])
                    shift = k - self.wordlength
                    solver.add(Implies(sh_a_i_s[j][i][k], c_sh_a_i[j][i] == (2 ** shift) * c_a_i[j][i]))
                    sh_a_i_s_sum += sh_a_i_s[j][i][k]
                solver.add(sh_a_i_s_sum == 1)

        sg_a_i = [[Bool(f'sg_a_i{j}{i}') for i in range(2)] for j in range(self.A_M)]

        for j in range(self.A_M):
            solver.add(sg_a_i[j][0] + sg_a_i[j][1] <= 1)
            for i in range(2):
                solver.add(Implies(sg_a_i[j][i], -1 * c_sh_a_i[j][i] == c_sh_sg_a_i[j][i]))
                solver.add(Implies(Not(sg_a_i[j][i]), c_sh_a_i[j][i] == c_sh_sg_a_i[j][i]))

        o_a_m_s_sg = [[[[Bool(f'o_a_m_s_sg{j}{i}{k}{l}') for l in range(2)] for k in range(2 * self.wordlength + 1)] for i in range(len(self.out))] for j in range(self.A_M + 1)]

        for i in range(len(self.out)):
            o_a_m_s_sg_sum = 0
            for j in range(self.A_M + 1):
                for k in range(2 * self.wordlength + 1):
                    shift = k - self.wordlength
                    for l in range(2):
                        o_a_m_s_sg_sum += o_a_m_s_sg[j][i][k][l]
                        solver.add(Implies(o_a_m_s_sg[j][i][k][l], (-1 ** l) * (2 ** shift) * c_a[j] == self.out[i]))
            solver.add(o_a_m_s_sg_sum == 1)

        print("solver Running")

        if solver.check() == sat:
            model = solver.model()
            print(model)
        else:
            print("No solution")

if __name__ == '__main__':
    hm = (1, 3, 2)
    wordlength = 10
    bitshift(hm, wordlength)
