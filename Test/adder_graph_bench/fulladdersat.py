from z3 import *

# Create a solver instance
solver = Solver()

# Define boolean variables for the full adder inputs and outputs
a = Bool('a')
b = Bool('b')
cin = Bool('cin')
sum = Bool('sum')
cout = Bool('cout')

# Add the CNF clauses for sum = a ⊕ b ⊕ cin
solver.add(Or(a, b, cin, Not(sum)))
solver.add(Or(a, b, Not(cin), sum))
solver.add(Or(a, Not(b), cin, sum))
solver.add(Or(Not(a), b, cin, sum))
solver.add(Or(Not(a), Not(b), Not(cin), sum))
solver.add(Or(Not(a), Not(b), cin, Not(sum)))
solver.add(Or(Not(a), b, Not(cin), Not(sum)))
solver.add(Or(a, Not(b), Not(cin), Not(sum)))

# Add the CNF clauses for cout = (a AND b) OR (cin AND (a ⊕ b))
solver.add(Or(Not(a), Not(b), cout))
solver.add(Or(a, b, Not(cout)))
solver.add(Or(Not(a), Not(cin), cout))
solver.add(Or(a, cin, Not(cout)))
solver.add(Or(Not(b), Not(cin), cout))
solver.add(Or(b, cin, Not(cout)))
solver.add(Not(a))
solver.add(Not(b))
solver.add(Not(cin))



# Check if the clauses are satisfiable
if solver.check() == sat:
    print("The CNF clauses are satisfiable.")
    model = solver.model()
    print("Model:")
    print(f"a = {model[a]}")
    print(f"b = {model[b]}")
    print(f"cin = {model[cin]}")
    print(f"sum = {model[sum]}")
    print(f"cout = {model[cout]}")
else:
    print("The CNF clauses are not satisfiable.")
