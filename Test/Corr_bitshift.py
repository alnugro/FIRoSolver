import numpy as np
from z3 import *
import matplotlib.pyplot as plt
import time

class bitshift():
    def __init__(self, out, wordlength, verbose=False):
        self.wordlength = wordlength
        self.out = out
        self.A_M = 15
        self.verbose = verbose
        solver = Solver()

        c_a = [Int(f'c_a{a}') for a in range(self.A_M + 1)]
        solver.add(c_a[0] == 1)

        c_sh_sg_a_i = [[Int(f'c_sh_sg_a_i{a}{i}') for i in range(2)] for a in range(self.A_M)]
        for a in range(self.A_M):
            solver.add(c_a[a + 1] == c_sh_sg_a_i[a][0] + c_sh_sg_a_i[a][1])

        c_a_i = [[Int(f'c_a_i{a}{i}') for i in range(2)] for a in range(self.A_M)]
        c_a_i_k = [[[Bool(f'c_a_i_k{a}{i}{k}') for k in range(self.A_M + 1)] for i in range(2)] for a in range(self.A_M)]

        for a in range(self.A_M):
            for i in range(2):
                for k in range(a + 1):
                    solver.add(Implies(c_a_i_k[a][i][k], c_a_i[a][i] == c_a[k]))
                solver.add(PbEq([(c_a_i_k[a][i][k], 1) for k in range(a + 1)], 1))

        c_sh_a_i = [[Int(f'c_sh_a_i{a}{i}') for i in range(2)] for a in range(self.A_M)]
        sh_a_i_s = [[[Bool(f'sh_a_i_s{a}{i}{s}') for s in range(2 * self.wordlength + 1)] for i in range(2)] for a in range(self.A_M)]

        for a in range(self.A_M):
            for i in range(2):
                for s in range(2 * self.wordlength + 1):
                    if s > self.wordlength and i == 0:
                        solver.add(sh_a_i_s[a][i][s] == False)
                    if s < self.wordlength and i == 0:
                        solver.add(sh_a_i_s[a][0][s] == sh_a_i_s[a][1][s])
                    shift = s - self.wordlength
                    solver.add(Implies(sh_a_i_s[a][i][s], c_sh_a_i[a][i] == (2 ** shift) * c_a_i[a][i]))
                solver.add(PbEq([(sh_a_i_s[a][i][s], 1) for s in range(2 * self.wordlength + 1)], 1))

        sg_a_i = [[Bool(f'sg_a_i{a}{i}') for i in range(2)] for a in range(self.A_M)]

        for a in range(self.A_M):
            solver.add(sg_a_i[a][0] + sg_a_i[a][1] <= 1)
            for i in range(2):
                solver.add(Implies(sg_a_i[a][i], -1 * c_sh_a_i[a][i] == c_sh_sg_a_i[a][i]))
                solver.add(Implies(Not(sg_a_i[a][i]), c_sh_a_i[a][i] == c_sh_sg_a_i[a][i]))

        o_a_m_s_sg = [[[[Bool(f'o_a_m_s_sg{a}{i}{s}{sg}') for sg in range(2)] for s in range(2 * self.wordlength + 1)] for i in range(len(self.out))] for a in range(self.A_M + 1)]

        for i in range(len(self.out)):
            for a in range(self.A_M + 1):
                for s in range(2 * self.wordlength + 1):
                    shift = s - self.wordlength
                    for sg in range(2):
                        solver.add(Implies(o_a_m_s_sg[a][i][s][sg], (-1 ** sg) * (2 ** shift) * c_a[a] == self.out[i]))
            solver.add(PbEq([(o_a_m_s_sg[a][i][s][sg], 1) for a in range(self.A_M + 1) for s in range(2 * self.wordlength + 1) for sg in range(2)], 1))

        print("solver Running")
        start_time = time.time()  # Start timing


        if solver.check() == sat:
            elapsed_time = time.time() - start_time  # Calculate elapsed time
            print(f"Solving time: {elapsed_time:.4f} seconds")
            print("Problem sat")
            model = solver.model()
            self.validate(model, c_a, c_a_i, c_a_i_k, c_sh_a_i, sg_a_i, c_sh_sg_a_i, o_a_m_s_sg, sh_a_i_s)
        else:
            print("No solution")
            raise ValueError("Solver found no solution")

    def _print(self, msg):
        if self.verbose:
            print(msg)

    def validate(self, model, c_a, c_a_i, c_a_i_k, c_sh_a_i, sg_a_i, c_sh_sg_a_i, o_a_m_s_sg, sh_a_i_s):
        for a in range(0, self.A_M):
            c_a_val = model.eval(c_a[a + 1]).as_long()
            self._print(f"Adder {a + 1}: output of adder {a + 1} is {c_a_val}")

            for i in range(2):
                # Determine the connection
                input_value = 0
                for k in range(a + 1):
                    if is_true(model.eval(c_a_i_k[a][i][k])):
                        input_value = model.eval(c_a[k]).as_long()
                        if i == 0:
                            self._print(f"  Left shifter of Adder {a + 1} is connected to input {k} with a value of {input_value}")
                        else:
                            self._print(f"  Right shifter of Adder {a + 1} is connected to input {k} with a value of {input_value}")

                # Determine the shift
                shift = 0
                for s in range(2 * self.wordlength + 1):
                    if is_true(model.eval(sh_a_i_s[a][i][s])):
                        shift = s - self.wordlength
                        if i == 0:
                            self._print(f"  Left Shifter of Adder {a + 1} shifted by {shift} bits, therefore it's multiplied by {2**shift}")
                        else:
                            self._print(f"  Right Shifter of Adder {a + 1} shifted by {shift} bits, therefore it's multiplied by {2**shift}")

                # Determine the sign
                sign = 0
                if is_true(model.eval(sg_a_i[a][i])):
                    sign = -1
                    if i == 0:
                        self._print(f"  Left Shifter of Adder {a + 1} has negative sign")
                    else:
                        self._print(f"  Right Shifter of Adder {a + 1} has negative sign")
                else:
                    sign = 1
                    if i == 0:
                        self._print(f"  Left Shifter of Adder {a + 1} has positive sign")
                    else:
                        self._print(f"  Right Shifter of Adder {a + 1} has positive sign")

                c_sh_sg_val = model.eval(c_sh_sg_a_i[a][i]).as_long()
                if i == 0:
                    self._print(f"  Left Shifter of Adder {a + 1} end value is {c_sh_sg_val}")
                else:
                    self._print(f"  Right Shifter of Adder {a + 1} end value is {c_sh_sg_val}")

                # Validate left or right shift
                c_sh_sg_val_calc = (2**shift) * sign * input_value
                if c_sh_sg_val_calc != c_sh_sg_val:
                    if i == 0:
                        self._print(f" Validation failed for Left Shifter of Adder {a + 1}: Expected: {c_sh_sg_val}, Calculated: {c_sh_sg_val_calc}")
                        raise ValueError(f"Validation failed for Left Shifter of Adder {a + 1}: Expected: {c_sh_sg_val}, Calculated: {c_sh_sg_val_calc}")
                    else:
                        self._print(f" Validation failed for Right Shifter of Adder {a + 1}: Expected: {c_sh_sg_val}, Calculated: {c_sh_sg_val_calc}")
                        raise ValueError(f"Validation failed for Right Shifter of Adder {a + 1}: Expected: {c_sh_sg_val}, Calculated: {c_sh_sg_val_calc}")

                # Validate adder Sums
                c_a_val_calc = sum([model.eval(c_sh_sg_a_i[a][i]).as_long() for i in range(2)])
                if c_a_val != c_a_val_calc:
                    self._print(f"Validation failed for Adder output c_a[{a + 1}] in adder {a + 1}: Model: {c_a_val}, Calculated: {c_a_val_calc}")
                    raise ValueError(f"Validation failed for Adder output c_a[{a + 1}] in adder {a + 1}: Model: {c_a_val}, Calculated: {c_a_val_calc}")

        for i in range(len(self.out)):
            for a in range(self.A_M + 1):
                for s in range(2 * self.wordlength + 1):
                    shift = s - self.wordlength
                    for sg in range(2):
                        if is_true(model.eval(o_a_m_s_sg[a][i][s][sg])):
                            c_a_val = model.eval(c_a[a]).as_long()
                            calculated_value = (-1 ** sg) * (2 ** shift) * c_a_val
                            if sg == 0:
                                sign = 1
                            else:
                                sign = -1
                            self._print(f'Output[{i}] is connected to adder {a + 1} with a value of {c_a_val} multiplied by {2**shift} with sign of {sign}')
                            self._print(f'Output[{i}]: Expected: {self.out[i]}, Calculated: {calculated_value}')
                            if calculated_value != self.out[i]:
                                self._print(f"Validation failed for output[{i}]: Expected: {self.out[i]}, Calculated: {calculated_value}")
                                raise ValueError(f"Validation failed for output[{i}]: Expected: {self.out[i]}, Calculated: {calculated_value}")
                            
        print("Validation Completed with no Error")

if __name__ == '__main__':
    hm = (25, 23, 11,25,75)
    wordlength = 10
    bitshift(hm, wordlength, verbose=True)
