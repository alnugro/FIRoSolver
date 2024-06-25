from z3 import *

h_int = [Real(f'h_int_{i}') for i in range(10)]

h_int_sum = 0

solver = Solver()

for i in range(10):
    h_int_sum += h_int[i]
    solver.add(h_int[i] > 0)

solver.add(h_int_sum <= 10)

if solver.check() == sat:
    print("solver sat")
    model = solver.model()
    for i in range(10):
        print(f'h_int_{i} = {model[h_int[i]]}')
else:
    print("Unsatisfiable")
