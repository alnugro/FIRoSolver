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
solver.add(Not(a[5]))
solver.add(Not(b[5]))
solver.add(a[4])
solver.add(Not(b[4]))


solver.add(Not(sa[0]))
solver.add(Not(sa[1]))
solver.add(sa[2] == a[0])
solver.add(sa[3] == a[1])
solver.add(Not(couta[3]))
solver.add(sa[4] == Xor(couta[4-1], a[0], a[2]))
solver.add(sa[5] == Xor(couta[5-1], a[1], a[3]))
solver.add(sa[6] == Xor(couta[6-1], a[2], a[4]))
solver.add(sa[7] == Xor(couta[7-1], a[3], a[5]))
solver.add(sa[8] == Xor(couta[8-1], a[4], a[5]))
solver.add(sa[9] == Xor(couta[9-1], a[5], a[5]))

solver.add(couta[4] == Or(And(a[0], a[2]), And(a[0], couta[3]), And(a[2], couta[3])))
solver.add(couta[5] == Or(And(a[1], a[3]), And(a[1], couta[4]), And(a[3], couta[4])))
solver.add(couta[6] == Or(And(a[2], a[4]), And(a[2], couta[5]), And(a[4], couta[5])))
solver.add(couta[7] == Or(And(a[3], a[5]), And(a[3], couta[6]), And(a[5], couta[6])))
solver.add(couta[8] == Or(And(a[4], a[5]), And(a[4], couta[7]), And(a[5], couta[7])))
solver.add(couta[9] == Or(And(a[5], a[5]), And(a[5], couta[8]), And(a[5], couta[8])))

solver.add(sb[0])
solver.add(Not(sb[1]))
solver.add(sb[2] == b[0])
solver.add(sb[3] == b[1])
solver.add(Not(coutb[3]))
solver.add(sb[4] == Xor(coutb[4-1], b[0], b[2]))
solver.add(sb[5] == Xor(coutb[5-1], b[1], b[3]))
solver.add(sb[6] == Xor(coutb[6-1], b[2], b[4]))
solver.add(sb[7] == Xor(coutb[7-1], b[3], b[5]))
solver.add(sb[8] == Xor(coutb[8-1], b[4], b[5]))
solver.add(sb[9] == Xor(coutb[9-1], b[5], b[5]))

solver.add(coutb[4] == Or(And(Not(b[0]), b[2]), And(Not(b[0]), coutb[3]), And(b[2], coutb[3])))
solver.add(coutb[5] == Or(And(Not(b[1]), b[3]), And(Not(b[1]), coutb[4]), And(b[3], coutb[4])))
solver.add(coutb[6] == Or(And(Not(b[2]), b[4]), And(Not(b[2]), coutb[5]), And(b[4], coutb[5])))
solver.add(coutb[7] == Or(And(Not(b[3]), b[5]), And(Not(b[3]), coutb[6]), And(b[5], coutb[6])))
solver.add(coutb[8] == Or(And(Not(b[4]), b[5]), And(Not(b[4]), coutb[7]), And(b[5], coutb[7])))
solver.add(coutb[9] == Or(And(Not(b[5]), b[5]), And(Not(b[5]), coutb[8]), And(b[5], coutb[8])))

solver.add(cinz)
solver.add(z[0] == Xor(sa[0], Not(sb[0]), cinz))

for i in range(1, 10):
    solver.add(z[i] == Xor(sa[i], Not(sb[i]), coutz[i-1]))

solver.add(coutz[0] == Or(And(Not(sb[0]), sa[0]), And(Not(sb[0]), cinz), And(sa[0], cinz)))
for i in range(1, 10):
    solver.add(coutz[i] == Or(And(Not(sb[i]), sa[i]), And(Not(sb[i]), coutz[i-1]), And(sa[i], coutz[i-1])))

for i in range(0, 9):
    solver.add(Or(z[9], Not(z[i])))

# Assert that z[1] is True
solver.add(a[1] == True)

if solver.check() == sat:
    print("solver sat")
    model = solver.model()
    # Print the model in an ordered way
    variables = a + b + sa + couta + sb + coutb + coutz + z + [cinz]
    for v in variables:
        print(f"{v} = {model.evaluate(v)}")
else:
    print("unsat")
