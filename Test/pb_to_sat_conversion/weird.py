from z3 import *


a = [Bool(f'a{i}') for i in range(6)]
b = [Bool(f'b{i}') for i in range(6)]

sa = [Bool(f'sa{i}') for i in range(10)]
couta = [Bool(f'couta{i}') for i in range(10)]

sb = [Bool(f'sb{i}') for i in range(10)]
coutb = [Bool(f'coutb{i}') for i in range(10)]

coutz = [Bool(f'coutz{i}') for i in range(10)]
z = [Bool(f'z{i}') for i in range(10)]

cinz = Bool('cinz')

solver = Solver()


solver.add(a[0])

if solver.check() == sat:
    print("solver sat")
    model = solver.model()
    # Print the model in an ordered way
    for v in a + b + sa + couta + sb + coutb + coutz + z + [cinz]:
        if v in model:
            print(f"{v} = {model[v]}")
        else:
            print(f"{v} = False")
else:
    print("unsat")
