from pysat.pb import PBEnc, EncType
from pysat.solvers import Solver

# Function to get the value of a specific variable in the model
def get_var_value_in_model(model, var):
    return var in model

# Example 1: Encode an at-most-one constraint
# (x1 + 2*x2 + 3*x3 <= 3)
lits = [1, 2, 3]
weights = [1, 2, 3]
bound = 3
cnf = PBEnc.atmost(lits=lits, weights=weights, bound=bound)

# Print the CNF clauses for the at-most constraint
print("CNF clauses for at-most constraint:")
print(cnf.clauses)

# Solve the CNF with pysat
solver = Solver(name='g3')
for clause in cnf.clauses:
    solver.add_clause(clause)

if solver.solve():
    model = solver.get_model()
    print("SATISFIABLE")
    print("Model:", model)
    var_value = get_var_value_in_model(model, 1)
    print("Variable 1 is", "True" if var_value else "False")
else:
    print("UNSATISFIABLE")

solver.delete()

# Example 2: Encode an equality constraint
# (x1 + 2*x2 + 3*x3 == 3)
cnf = PBEnc.equals(lits=lits, weights=weights, bound=bound, encoding=EncType.bdd)

# Print the CNF clauses for the equality constraint
print("\nCNF clauses for equality constraint:")
print(cnf.clauses)

# Solve the CNF with pysat
solver = Solver(name='g3')
for clause in cnf.clauses:
    solver.add_clause(clause)

if solver.solve():
    model = solver.get_model()
    print("SATISFIABLE")
    print("Model:", model)
    var_value = get_var_value_in_model(model, 1)
    print("Variable 1 is", "True" if var_value else "False")
else:
    print("UNSATISFIABLE")

solver.delete()
