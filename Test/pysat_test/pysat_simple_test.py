from pysat.formula import CNF, IDPool
from pysat.pb import PBEnc
from pysat.solvers import Solver

# Define literals, weights, and the bound
lits = [1, 2, 3, 4]
weights = [3*10**50, 3*10**40, 3*10**40, 3*10**40]
upperbound = 5*10**40
lowerbound = 4*10**40
inverted_weights = [-w for w in weights]

solver = Solver(name='Cadical195')
solver.activate_atmost()

solver.add_atmost(lits=lits, k=upperbound, weights=weights)
solver.add_atmost(lits=[-l for l in lits], k=sum(weights)-lowerbound, weights=weights)

is_sat = solver.solve()

if is_sat:
    model = solver.get_model()
    print("SAT")
    print("Model:", model)
else:
    print("UNSAT")

solver.delete()

solver2 = Solver()
cnf = PBEnc.atmost(lits=lits, weights=weights, bound=upperbound, top_id=5)
cnf.extend(PBEnc.atleast(lits=lits, weights=weights, bound=lowerbound, top_id=5).clauses)

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
cnf = PBEnc.atmost(lits=lits, weights=weights, bound=upperbound, vpool=pool)
cnf.extend(PBEnc.atleast(lits=lits, weights=weights, bound=lowerbound, vpool=pool).clauses)

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
