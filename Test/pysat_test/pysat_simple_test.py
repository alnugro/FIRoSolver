from pysat.formula import CNF, IDPool
from pysat.pb import PBEnc
from pysat.solvers import Solver



# Define literals, weights, and the bound
lits = [1, 2, 3 ,4]
weights = [3, 3, 3, 3]
upperbound = 5
lowerbound = 4
inverted_wights = [-w for w in weights]

# Solve the CNF using CaDiCaL
solver = Solver(name='Cadical195')
solver.activate_atmost()


solver.add_atmost(lits=lits, k=upperbound, weights=weights)
solver.add_atmost(lits=lits, k=-lowerbound, weights=inverted_wights)

is_sat = solver.solve()

# Solve the problem
is_sat = solver.solve()

if is_sat:
    model = solver.get_model()
    print("SAT")
    print("Model:", model)
else:
    print("UNSAT")

# Delete the solver to free up resources
solver.delete()

solver2 = Solver()
cnf = PBEnc.atmost(lits=lits, weights=weights, bound = upperbound,top_id=5)
cnf = PBEnc.atleast(lits=lits, weights=weights,bound = lowerbound, top_id=5)

for clause in cnf.clauses:
    solver2.add_clause(clause)

is_sat = solver2.solve()

if is_sat:
    model = solver2.get_model()
    print("SAT")
    print("Model:", model)
else:
    print("UNSAT")

solver2.delete()

pool = IDPool(start_from=5)
solver3 = Solver(name='minisat22')
cnf = PBEnc.atmost(lits=lits, weights=weights, bound=upperbound,vpool=pool)
cnf = PBEnc.atleast(lits=lits, weights=weights,bound = lowerbound, vpool=pool)

for clause in cnf.clauses:
    solver3.add_clause(clause)

is_sat = solver3.solve()

if is_sat:
    model = solver3.get_model()
    print("SAT")
    print("Model:", model)
else:
    print("UNSAT")

solver3.delete()


