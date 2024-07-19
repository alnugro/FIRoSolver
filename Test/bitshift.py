import numpy as np
from z3 import *
import matplotlib.pyplot as plt
import time

class bitshift():
    def __init__(self, out, wordlength):
        self.wordlength = wordlength
        self.out = out
        self.A_M = 2
        solver = Solver()

        c_a = [Int(f'c_a{a}') for a in range(self.A_M + 1)]
        solver.add(c_a[0] == 1)

        c_sh_sg_a_i = [[Int(f'c_sh_sg_a_i{a}{i}') for i in range(2)] for a in range(self.A_M)]
        for a in range(self.A_M):
            solver.add(c_a[a + 1] == c_sh_sg_a_i[a][0] + c_sh_sg_a_i[a][1])

        c_a_i = [[Int(f'c_a_i{a}{i}') for i in range(2)] for a in range(self.A_M)]
        c_a_i_k = [[[Bool(f'c_a_i_k{a}{i}{k}') for k in range(self.A_M)] for i in range(2)] for a in range(self.A_M)]

        for a in range(self.A_M):
            for i in range(2):
                c_a_i_k_sum = 0
                for k in range(a):
                    c_a_i_k_sum += c_a_i_k[a][i][k]
                    solver.add(Implies(c_a_i_k[a][i][k], c_a_i[a][i] == c_a[k]))
                if c_a_i_k_sum == 0: #somehow if 0 is added it will yield unsatisfiable
                    continue
                #print("added this", c_a_i_k_sum)
                solver.add(c_a_i_k_sum == 1)

        c_sh_a_i = [[Int(f'c_sh_a_i{a}{i}') for i in range(2)] for a in range(self.A_M)]
        sh_a_i_s = [[[Bool(f'sh_a_i_s{a}{i}{s}') for s in range(2 * self.wordlength + 1)] for i in range(2)] for a in range(self.A_M)]

        for a in range(self.A_M):
            for i in range(2):
                sh_a_i_s_sum = 0
                for s in range(2 * self.wordlength + 1):
                    if s > self.wordlength and i == 0:
                        solver.add(sh_a_i_s[a][i][s] == False)
                    if s < self.wordlength and i == 0:
                        solver.add(sh_a_i_s[a][0][s] == sh_a_i_s[a][1][s])
                    shift = s - self.wordlength
                    solver.add(Implies(sh_a_i_s[a][i][s], c_sh_a_i[a][i] == (2 ** shift) * c_a_i[a][i]))
                    sh_a_i_s_sum += sh_a_i_s[a][i][s]
                solver.add(sh_a_i_s_sum == 1)

        sg_a_i = [[Bool(f'sg_a_i{a}{i}') for i in range(2)] for a in range(self.A_M)]

        for a in range(self.A_M):
            solver.add(sg_a_i[a][0] + sg_a_i[a][1] <= 1)
            for i in range(2):
                solver.add(Implies(sg_a_i[a][i], -1 * c_sh_a_i[a][i] == c_sh_sg_a_i[a][i]))
                solver.add(Implies(Not(sg_a_i[a][i]), c_sh_a_i[a][i] == c_sh_sg_a_i[a][i]))

        o_a_m_s_sg = [[[[Bool(f'o_a_m_s_sg{a}{i}{s}{sg}') for sg in range(2)] for s in range(2 * self.wordlength + 1)] for i in range(len(self.out))] for a in range(self.A_M + 1)]

        for i in range(len(self.out)):
            o_a_m_s_sg_sum = 0
            for a in range(self.A_M + 1):
                for s in range(2 * self.wordlength + 1):
                    shift = s - self.wordlength
                    for sg in range(2):
                        o_a_m_s_sg_sum += o_a_m_s_sg[a][i][s][sg]
                        solver.add(Implies(o_a_m_s_sg[a][i][s][sg], (-1 ** sg) * (2 ** shift) * c_a[a] == self.out[i]))
            solver.add(o_a_m_s_sg_sum == 1)

        print("solver Running")

        if solver.check() == sat:
            print("its sat")
            model = solver.model()
            print(model)
        else:
            print("No solution")

if __name__ == '__main__':
    hm = (1, 3, 2)
    wordlength = 10
    bitshift(hm, wordlength)
