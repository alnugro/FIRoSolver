import numpy as np
from z3 import *
import matplotlib.pyplot as plt
import time

class bitshift():
    def __init__(self, out, wordlength):
        self.wordlength = wordlength
        self.out = out
        self.A_M=5
        solver = Solver()
        c_a = [Int(f'c_a{i}') for i in range(self.A_M+1)] #constant of input a, a is [i] here to make it more readable
        solver.add(c_a[0]==1) #c0 is 1, it is the first adder input
        c_sh_sg_a_i= [[Int(f'c_sh_sg_a_i{j}{i}') for j in range(self.A_M)] for i in range(2)] #shifted and signed corrected output of the adder a. a is j and i is the adder pos i, 0 is left, 1 is right
        for j in range(self.A_M):
            solver.add(c_a[j+1]==c_sh_sg_a_i[j][0]+c_sh_sg_a_i[j][1]) #sum the left and right adder to ca, ca_sh_sg goes from 0 to AM-1 and ca goes from 0 to AM with C[0]=1

        c_a_i=[[Int(f'c_a_i{j}{i}') for j in range(self.A_M)] for i in range(2)]
        c_a_i_k=[[[Bool(f'c_a_i_k{j}{i}{k}') for j in range(self.A_M)] for i in range(2)] for k in range(self.A_M)]
        
        for j in range(self.A_M):
            for i in range(2):
                c_a_i_k_sum = 0
                for k in range(j):
                    solver.add(Implies(c_a_i_k[j][i][k],c_a_i[j][i]==c_a[k]))
                    c_a_i_k_sum+=c_a_i_k[j][i][k]
                solver.add(c_a_i_k_sum==1)

    
                    

        c_sh_a_i= [[Int(f'c_sh_a_i{j}{i}') for j in range(self.A_M)] for i in range(2)]
        sh_a_i_s= [[[Bool(f'sh_a_i_s{j}{i}{k}') for j in range(self.A_M)] for i in range(2)] for k in range(2*self.wordlength+1)]

        for j in range(self.A_M):
            

            for i in range(2):
                sh_a_i_s_sum = 0
                for k in range(2*self.wordlength+1):
                    if k > self.wordlength & i==0: #middlepoint or where the shift = 0 is the self.wordlength
                        solver.add(sh_a_i_s[j][i][k]==0)

                    if k < self.wordlength & i==0:
                        solver.add(sh_a_i_s[j][0][k]==sh_a_i_s[j][1][k])

                    shift= k-1*self.wordlength
                    solver.add(Implies(sh_a_i_s[j][i][k] ,c_sh_a_i[j][i]==(2**shift)*c_a_i[j][i]))
                    sh_a_i_s_sum+=sh_a_i_s[j][i][k]
                solver.add(sh_a_i_s_sum==1)

        sg_a_i= [[Bool(f'sg_a_i{j}{i}{k}') for j in range(self.A_M)] for i in range(2)]

        for j in range(self.A_M):
            solver.add(sg_a_i[j][0]+sg_a_i[j][1]<=1)
            for i in range(2):
                solver.add(Implies(sg_a_i[j][i] ,-1*c_sh_a_i[j][i]==c_sh_sg_a_i[j][i]))
                solver.add(Implies(Not(sg_a_i)[j][i] ,c_sh_a_i[j][i]==c_sh_sg_a_i[j][i]))

        o_a_m_s_sg=[[[[Bool(f'o_a_m_s_sg{j}{i}{k}{l}') for j in range(self.A_M)] for i in range(len(self.out))] for k in range(2*self.wordlength+1)] for l in range(2)]
        
        for i in range(len(self.out)):
            o_a_m_s_sg_sum = 0
            for j in range(self.A_M+1):
                for k in range(2*self.wordlength+1):
                    shift= k-1*self.wordlength
                    for l in range(2):
                        o_a_m_s_sg_sum+=o_a_m_s_sg[j][i][k][l]
                        solver.add(Implies(o_a_m_s_sg[j][i][k][l], (-1**l)*(2**shift)*c_a[j]==self.out[i]))

            solver.add(o_a_m_s_sg_sum == 1)


                           


        print("solver Running")



        if solver.check() == sat:
            model = solver.model()
            print(model)
        else:
            print("No solution")




        





        
        
        


if __name__ == '__main__':
    hm=(15, 16 ,12)
    wordlength=10
    bitshift(hm, wordlength)