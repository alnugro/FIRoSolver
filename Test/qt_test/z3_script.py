import z3

x = z3.Int('x')
s = z3.Solver()
s.add(x > 5)
if s.check() == z3.sat:
    m = s.model()
    print(f"x = {m[x]}")
else:
    print("unsat")
