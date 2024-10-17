import numpy as np
from z3 import *
import matplotlib.pyplot as plt
import time

class bitshift():
    def __init__(self, wordlength, verbose=False):
        self.wordlength = 8
        self.verbose = verbose

        self.adder_wordlength = self.wordlength
        self.max_adder = 3
        self.adder_depth = 0
        self.result_model = {}
        self.coef_to_try = [75,99]


    def runsolver(self):
        half_order = len(self.coef_to_try) - 1

        ctx = z3.Context()


        
        solver = Solver(ctx=ctx)
       
        h = [[Bool(f'h_{a}_{w}', ctx=ctx) for w in range(self.wordlength)] for a in range(half_order + 1)]
        h_sum = []

        for m in range(half_order + 1):
            h_sum = []
            for w in range(self.wordlength):
                h_sum.append((h[m][w], 2 ** (w)))
            solver.add(PbEq((h_sum), self.coef_to_try[m]))
        


        # input multiplexer
        c = [[Bool(f'c_{i}_{w}', ctx=ctx) for w in range(self.adder_wordlength)] for i in range(self.max_adder + 2)]
        l = [[Bool(f'l_{i}_{w}', ctx=ctx) for w in range(self.adder_wordlength)] for i in range(1, self.max_adder + 1)]
        r = [[Bool(f'r_{i}_{w}', ctx=ctx) for w in range(self.adder_wordlength)] for i in range(1, self.max_adder + 1)]

        alpha = [[Bool(f'alpha_{i}_{a}', ctx=ctx) for a in range(i)] for i in range(1, self.max_adder + 1)]
        beta = [[Bool(f'Beta_{i}_{a}', ctx=ctx) for a in range(i)] for i in range(1, self.max_adder + 1)]

        # c0,w is always 0 except at 1
        for w in range(1, self.adder_wordlength):
            solver.add(Not(c[0][w]))

        solver.add(c[0][0])

        # bound ci,0 to be odd number
        for i in range(1, self.max_adder + 1):
            solver.add(c[i][0])

        # last c or c[N+1] is connected to ground, so all zeroes
        for w in range(self.adder_wordlength):
            solver.add(Not(c[self.max_adder + 1][w]))

        # input multiplexer
        for i in range(1, self.max_adder + 1):
            alpha_sum = []
            beta_sum = []
            for a in range(i):
                for word in range(self.adder_wordlength):
                    clause1_1 = Or(Not(alpha[i - 1][a]), Not(c[a][word]), l[i - 1][word])
                    clause1_2 = Or(Not(alpha[i - 1][a]), c[a][word], Not(l[i - 1][word]))
                    solver.add(And(clause1_1, clause1_2))

                    clause2_1 = Or(Not(beta[i - 1][a]), Not(c[a][word]), r[i - 1][word])
                    clause2_2 = Or(Not(beta[i - 1][a]), c[a][word], Not(r[i - 1][word]))
                    solver.add(And(clause2_1, clause2_2))

                alpha_sum.append(alpha[i - 1][a])
                beta_sum.append(beta[i - 1][a])

            solver.add(AtMost(*alpha_sum, 1))
            solver.add(AtLeast(*alpha_sum, 1))

            solver.add(AtMost(*beta_sum, 1))
            solver.add(AtLeast(*beta_sum, 1))

        # Left Shifter
        # k is the shift selector
        gamma = [[Bool(f'gamma_{i}_{k}', ctx=ctx) for k in range(self.adder_wordlength - 1)] for i in range(1, self.max_adder + 1)]
        s = [[Bool(f's_{i}_{w}', ctx=ctx) for w in range(self.adder_wordlength)] for i in range(1, self.max_adder + 1)]

        for i in range(1, self.max_adder + 1):
            gamma_sum = []
            for k in range(self.adder_wordlength - 1):
                for j in range(self.adder_wordlength - 1 - k):
                    clause3_1 = Or(Not(gamma[i - 1][k]), Not(l[i - 1][j]), s[i - 1][j + k])
                    clause3_2 = Or(Not(gamma[i - 1][k]), l[i - 1][j], Not(s[i - 1][j + k]))
                    # solver.add(And(clause3_1, clause3_2))
                    solver.add(clause3_1)
                    solver.add(clause3_2)

                gamma_sum.append(gamma[i - 1][k])

            solver.add(AtMost(*gamma_sum, 1))
            solver.add(AtLeast(*gamma_sum, 1))

            for kf in range(1, self.adder_wordlength - 1):
                for b in range(kf):
                    clause4 = Or(Not(gamma[i - 1][kf]), Not(s[i - 1][b]))
                    clause5 = Or(Not(gamma[i - 1][kf]), Not(l[i - 1][self.adder_wordlength - 1]), l[i - 1][self.adder_wordlength - 2 - b])
                    clause6 = Or(Not(gamma[i - 1][kf]), l[i - 1][self.adder_wordlength - 1], Not(l[i - 1][self.adder_wordlength - 2 - b]))
                    solver.add(clause4)
                    solver.add(clause5)
                    solver.add(clause6)

            clause7_1 = Or(Not(l[i - 1][self.adder_wordlength - 1]), s[i - 1][self.adder_wordlength - 1])
            clause7_2 = Or(l[i - 1][self.adder_wordlength - 1], Not(s[i - 1][self.adder_wordlength - 1]))
            # solver.add(And(clause7_1, clause7_2))
            solver.add(clause7_1)
            solver.add(clause7_2)

        delta = [Bool(f'delta_{i}', ctx=ctx) for i in range(1, self.max_adder + 1)]
        u = [[Bool(f'u_{i}_{w}', ctx=ctx) for w in range(self.adder_wordlength)] for i in range(1, self.max_adder + 1)]
        x = [[Bool(f'x_{i}_{w}', ctx=ctx) for w in range(self.adder_wordlength)] for i in range(1, self.max_adder + 1)]

        # delta selector
        for i in range(1, self.max_adder + 1):
            for word in range(self.adder_wordlength):
                clause8_1 = Or(Not(delta[i - 1]), Not(s[i - 1][word]), x[i - 1][word])
                clause8_2 = Or(Not(delta[i - 1]), s[i - 1][word], Not(x[i - 1][word]))
                # solver.add(And(clause8_1, clause8_2))
                solver.add(clause8_1)
                solver.add(clause8_2)

                clause9_1 = Or(Not(delta[i - 1]), Not(r[i - 1][word]), u[i - 1][word])
                clause9_2 = Or(Not(delta[i - 1]), r[i - 1][word], Not(u[i - 1][word]))
                # solver.add(And(clause9_1, clause9_2))
                solver.add(clause9_1)
                solver.add(clause9_2)

                clause10_1 = Or(delta[i - 1], Not(s[i - 1][word]), u[i - 1][word])
                clause10_2 = Or(delta[i - 1], s[i - 1][word], Not(u[i - 1][word]))
                # solver.add(And(clause10_1, clause10_2))
                solver.add(clause10_1)
                solver.add(clause10_2)

                clause11_1 = Or(delta[i - 1], Not(r[i - 1][word]), x[i - 1][word])
                clause11_2 = Or(delta[i - 1], r[i - 1][word], Not(x[i - 1][word]))
                # solver.add(And(clause11_1, clause11_2))
                solver.add(clause11_1)
                solver.add(clause11_2)

        epsilon = [Bool(f'epsilon_{i}', ctx=ctx) for i in range(1, self.max_adder + 1)]
        y = [[Bool(f'y_{i}_{w}', ctx=ctx) for w in range(self.adder_wordlength)] for i in range(1, self.max_adder + 1)]

        # xor
        for i in range(1, self.max_adder + 1):
            for word in range(self.adder_wordlength):
                clause12 = Or(u[i - 1][word], epsilon[i - 1], Not(y[i - 1][word]))
                clause13 = Or(u[i - 1][word], Not(epsilon[i - 1]), y[i - 1][word])
                clause14 = Or(Not(u[i - 1][word]), epsilon[i - 1], y[i - 1][word])
                clause15 = Or(Not(u[i - 1][word]), Not(epsilon[i - 1]), Not(y[i - 1][word]))
                solver.add(clause12)
                solver.add(clause13)
                solver.add(clause14)
                solver.add(clause15)

        # ripple carry
        z = [[Bool(f'z_{i}_{w}', ctx=ctx) for w in range(self.adder_wordlength)] for i in range(1, self.max_adder + 1)]
        cout = [[Bool(f'cout_{i}_{w}', ctx=ctx) for w in range(self.adder_wordlength)] for i in range(1, self.max_adder + 1)]

        for i in range(1, self.max_adder + 1):
            # Clauses for sum = a ⊕ b ⊕ cin at 0
            clause16 = Or(x[i - 1][0], y[i - 1][0], epsilon[i - 1], Not(z[i - 1][0]))
            clause17 = Or(x[i - 1][0], y[i - 1][0], Not(epsilon[i - 1]), z[i - 1][0])
            clause18 = Or(x[i - 1][0], Not(y[i - 1][0]), epsilon[i - 1], z[i - 1][0])
            clause19 = Or(Not(x[i - 1][0]), y[i - 1][0], epsilon[i - 1], z[i - 1][0])
            clause20 = Or(Not(x[i - 1][0]), Not(y[i - 1][0]), Not(epsilon[i - 1]), z[i - 1][0])
            clause21 = Or(Not(x[i - 1][0]), Not(y[i - 1][0]), epsilon[i - 1], Not(z[i - 1][0]))
            clause22 = Or(Not(x[i - 1][0]), y[i - 1][0], Not(epsilon[i - 1]), Not(z[i - 1][0]))
            clause23 = Or(x[i - 1][0], Not(y[i - 1][0]), Not(epsilon[i - 1]), Not(z[i - 1][0]))

            solver.add(clause16)
            solver.add(clause17)
            solver.add(clause18)
            solver.add(clause19)
            solver.add(clause20)
            solver.add(clause21)
            solver.add(clause22)
            solver.add(clause23)

            # Clauses for cout = (a AND b) OR (cin AND (a ⊕ b))
            clause24 = Or(Not(x[i - 1][0]), Not(y[i - 1][0]), cout[i - 1][0])
            clause25 = Or(x[i - 1][0], y[i - 1][0], Not(cout[i - 1][0]))
            clause26 = Or(Not(x[i - 1][0]), Not(epsilon[i - 1]), cout[i - 1][0])
            clause27 = Or(x[i - 1][0], epsilon[i - 1], Not(cout[i - 1][0]))
            clause28 = Or(Not(y[i - 1][0]), Not(epsilon[i - 1]), cout[i - 1][0])
            clause29 = Or(y[i - 1][0], epsilon[i - 1], Not(cout[i - 1][0]))

            solver.add(clause24)
            solver.add(clause25)
            solver.add(clause26)
            solver.add(clause27)
            solver.add(clause28)
            solver.add(clause29)

            for kf in range(1, self.adder_wordlength):
                # Clauses for sum = a ⊕ b ⊕ cin at kf
                clause30 = Or(x[i - 1][kf], y[i - 1][kf], cout[i - 1][kf - 1], Not(z[i - 1][kf]))
                clause31 = Or(x[i - 1][kf], y[i - 1][kf], Not(cout[i - 1][kf - 1]), z[i - 1][kf])
                clause32 = Or(x[i - 1][kf], Not(y[i - 1][kf]), cout[i - 1][kf - 1], z[i - 1][kf])
                clause33 = Or(Not(x[i - 1][kf]), y[i - 1][kf], cout[i - 1][kf - 1], z[i - 1][kf])
                clause34 = Or(Not(x[i - 1][kf]), Not(y[i - 1][kf]), Not(cout[i - 1][kf - 1]), z[i - 1][kf])
                clause35 = Or(Not(x[i - 1][kf]), Not(y[i - 1][kf]), cout[i - 1][kf - 1], Not(z[i - 1][kf]))
                clause36 = Or(Not(x[i - 1][kf]), y[i - 1][kf], Not(cout[i - 1][kf - 1]), Not(z[i - 1][kf]))
                clause37 = Or(x[i - 1][kf], Not(y[i - 1][kf]), Not(cout[i - 1][kf - 1]), Not(z[i - 1][kf]))

                solver.add(clause30)
                solver.add(clause31)
                solver.add(clause32)
                solver.add(clause33)
                solver.add(clause34)
                solver.add(clause35)
                solver.add(clause36)
                solver.add(clause37)

                # Clauses for cout = (a AND b) OR (cin AND (a ⊕ b)) at kf
                clause38 = Or(Not(x[i - 1][kf]), Not(y[i - 1][kf]), cout[i - 1][kf])
                clause39 = Or(x[i - 1][kf], y[i - 1][kf], Not(cout[i - 1][kf]))
                clause40 = Or(Not(x[i - 1][kf]), Not(cout[i - 1][kf - 1]), cout[i - 1][kf])
                clause41 = Or(x[i - 1][kf], cout[i - 1][kf - 1], Not(cout[i - 1][kf]))
                clause42 = Or(Not(y[i - 1][kf]), Not(cout[i - 1][kf - 1]), cout[i - 1][kf])
                clause43 = Or(y[i - 1][kf], cout[i - 1][kf - 1], Not(cout[i - 1][kf]))

                solver.add(clause38)
                solver.add(clause39)
                solver.add(clause40)
                solver.add(clause41)
                solver.add(clause42)
                solver.add(clause43)

            clause44 = Or(epsilon[i - 1], x[i - 1][self.adder_wordlength - 1], u[i - 1][self.adder_wordlength - 1], Not(z[i - 1][self.adder_wordlength - 1]))
            clause45 = Or(epsilon[i - 1], Not(x[i - 1][self.adder_wordlength - 1]), Not(u[i - 1][self.adder_wordlength - 1]), z[i - 1][self.adder_wordlength - 1])
            clause46 = Or(Not(epsilon[i - 1]), x[i - 1][self.adder_wordlength - 1], Not(u[i - 1][self.adder_wordlength - 1]), Not(z[i - 1][self.adder_wordlength - 1]))
            clause47 = Or(Not(epsilon[i - 1]), Not(x[i - 1][self.adder_wordlength - 1]), u[i - 1][self.adder_wordlength - 1], z[i - 1][self.adder_wordlength - 1])

            solver.add(clause44)
            solver.add(clause45)
            solver.add(clause46)
            solver.add(clause47)

        # right shift
        zeta = [[Bool(f'zeta_{i}_{k}', ctx=ctx) for k in range(self.adder_wordlength - 1)] for i in range(1, self.max_adder + 1)]

        for i in range(1, self.max_adder + 1):
            zeta_sum = []
            for k in range(self.adder_wordlength - 1):
                for j in range(self.adder_wordlength - 1 - k):
                    clause48_1 = Or(Not(zeta[i - 1][k]), Not(z[i - 1][j + k]), c[i][j])
                    clause48_2 = Or(Not(zeta[i - 1][k]), z[i - 1][j + k], Not(c[i][j]))
                    # solver.add(And(clause48_1, clause48_2))
                    solver.add(clause48_1)
                    solver.add(clause48_2)

                zeta_sum.append(zeta[i - 1][k])

            solver.add(AtMost(*zeta_sum, 1))
            solver.add(AtLeast(*zeta_sum, 1))

            for kf in range(1, self.adder_wordlength - 1):
                for b in range(kf):
                    clause49_1 = Or(Not(zeta[i - 1][kf]), Not(z[i - 1][self.adder_wordlength - 1]), c[i][self.adder_wordlength - 2 - b])
                    clause49_2 = Or(Not(zeta[i - 1][kf]), z[i - 1][self.adder_wordlength - 1], Not(c[i][self.adder_wordlength - 2 - b]))
                    solver.add(clause49_1)
                    solver.add(clause49_2)

                    clause50 = Or(Not(zeta[i - 1][kf]), Not(z[i - 1][b]))
                    solver.add(clause50)

            clause51_1 = Or(Not(z[i - 1][self.adder_wordlength - 1]), c[i][self.adder_wordlength - 1])
            clause51_2 = Or(z[i - 1][self.adder_wordlength - 1], Not(c[i][self.adder_wordlength - 1]))
            # solver.add(And(clause51_1, clause51_2))
            solver.add(clause51_1)
            solver.add(clause51_2)

        # set connected coefficient
        connected_coefficient = half_order + 1

        # solver connection
        theta = [[Bool(f'theta_{i}_{m}', ctx=ctx) for m in range(half_order + 1)] for i in range(self.max_adder + 2)]
        iota = [Bool(f'iota_{m}', ctx=ctx) for m in range(half_order + 1)]

        iota_sum = []
        for m in range(half_order + 1):
            theta_or = []
            for i in range(self.max_adder + 2):
                for word in range(self.adder_wordlength):
                    clause52_1 = Or(Not(theta[i][m]), Not(iota[m]), Not(c[i][word]), h[m][word])
                    clause52_2 = Or(Not(theta[i][m]), Not(iota[m]), c[i][word], Not(h[m][word]))
                    solver.add(clause52_1)
                    solver.add(clause52_2)
                theta_or.append(theta[i][m])
            # print(f"theta or {theta_or}")
            solver.add(Or(*theta_or))

        for m in range(half_order + 1):
            iota_sum.append(iota[m])
        solver.add(AtMost(*iota_sum, connected_coefficient))
        solver.add(AtLeast(*iota_sum, connected_coefficient))

       

    
        
        start_time=time.time()
        
        print("solver running")


        satisfiability = 'unsat'

        if solver.check() == sat:
            
            end_time = time.time()


            satisfiability = 'sat'

            print("solver sat")
            model = solver.model()

           


            # Store h
            h_values = []
            for i in range(len(h)):
                h_row = []
                for a in range(len(h[i])):
                    value = 1 if model.eval(h[i][a], model_completion=True) else 0
                    h_row.append(value)
                h_values.append(h_row)
            self.result_model.update({"h": h_values})

            # Store alpha selectors
            alpha_values = []
            for i in range(len(alpha)):
                alpha_row = []
                for a in range(len(alpha[i])):
                    value = 1 if model.eval(alpha[i][a], model_completion=True) else 0
                    alpha_row.append(value)
                alpha_values.append(alpha_row)
            self.result_model.update({"alpha": alpha_values})

            # Store beta selectors
            beta_values = []
            for i in range(len(beta)):
                beta_row = []
                for a in range(len(beta[i])):
                    value = 1 if model.eval(beta[i][a], model_completion=True) else 0
                    beta_row.append(value)
                beta_values.append(beta_row)
            self.result_model.update({"beta": beta_values})

            # Store gamma (left shift selectors)
            gamma_values = []
            for i in range(len(gamma)):
                gamma_row = []
                for k in range(self.adder_wordlength - 1):
                    value = 1 if model.eval(gamma[i][k], model_completion=True) else 0
                    gamma_row.append(value)
                gamma_values.append(gamma_row)
            self.result_model.update({"gamma": gamma_values})

            # Store delta selectors
            delta_values = []
            for i in range(len(delta)):
                value = 1 if model.eval(delta[i], model_completion=True) else 0
                delta_values.append(value)
            self.result_model.update({"delta": delta_values})

            # Store epsilon selectors
            epsilon_values = []
            for i in range(len(epsilon)):
                value = 1 if model.eval(epsilon[i], model_completion=True) else 0
                epsilon_values.append(value)
            self.result_model.update({"epsilon": epsilon_values})

            # Store zeta (right shift selectors)
            zeta_values = []
            for i in range(len(zeta)):
                zeta_row = []
                for k in range(self.adder_wordlength - 1):
                    value = 1 if model.eval(zeta[i][k], model_completion=True) else 0
                    zeta_row.append(value)
                zeta_values.append(zeta_row)
            self.result_model.update({"zeta": zeta_values})


            # Store theta array
            theta_values = []
            for i in range(len(theta)):
                theta_row = []
                for m in range(half_order + 1):
                    value = 1 if model.eval(theta[i][m], model_completion=True) else 0
                    theta_row.append(value)
                theta_values.append(theta_row)
            self.result_model.update({"theta": theta_values})

            # Store iota array
            iota_values = []
            for m in range(len(iota)):
                value = 1 if model.eval(iota[m], model_completion=True) else 0
                iota_values.append(value)
            self.result_model.update({"iota": iota_values})

         

            
            
                      

        else:
            # print(f"unsat core {solver.unsat_core()}")
            print("Unsatisfiable")
            end_time = time.time()

        print("solver stopped")
        duration = end_time - start_time
        print(f"Duration: {duration} seconds")

        # for item in self.result_model:
        #     print(f"result of {item} is {self.result_model[item]}")

        print(f"\n************Z3 Report****************")
        print(f"Total number of variables            : ")
        print(f"Total number of constraints (clauses): {len(solver.assertions())}\n" )

        return self.result_model , satisfiability

    def int_to_signed_binary_list(value, word_length):
        # Calculate the range for the word length
        min_val = -(1 << (word_length - 1))
        max_val = (1 << (word_length - 1)) - 1
        
        # Check if the value fits in the given word length
        if value < min_val or value > max_val:
            raise ValueError(f"Value {value} out of range for a {word_length}-bit signed integer")
        
        # Handle negative values using two's complement
        if value < 0:
            value = (1 << word_length) + value

        # Get the binary representation and pad with leading zeros if necessary
        bin_representation = format(value, f'0{word_length}b')
        
        print(bin_representation)
        
        # Convert the binary representation to a list of integers (0 or 1)
        bin_list = [int(bit) for bit in bin_representation]
        
        return bin_list





if __name__ == '__main__':
    wordlength = 6 #min wordlength would be 2
    bit = bitshift(wordlength, verbose=True)
    res, sat = bit.runsolver()
    print(res)
