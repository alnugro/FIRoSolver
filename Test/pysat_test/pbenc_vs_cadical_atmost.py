import time
import random
from pysat.pb import PBEnc
from pysat.solvers import Solver

# Function to measure time
def measure_time(func):
    start_time = time.time()
    result = func()
    end_time = time.time()
    return result, end_time - start_time

# Function to solve using PBEnc
def solve_with_pbenc(lits, weights, bound):
    cnf = PBEnc.atleast(lits=lits, weights=weights, bound=bound)
    solver = Solver(name='cadical195')
    for clause in cnf.clauses:
        solver.add_clause(clause)
    is_sat = solver.solve()
    model = solver.get_model() if is_sat else None
    solver.delete()
    return is_sat, model

# Function to solve using native add_atmost in CaDiCaL
def solve_with_native(lits, weights, bound):
    solver = Solver(name='cadical195')
    solver.activate_atmost()  # Ensure atmost is activated
    solver.add_atmost(lits, bound, weights)
    is_sat = solver.solve()
    model = solver.get_model() if is_sat else None
    solver.delete()
    return is_sat, model

# Benchmark length
bench_length = 150
step = 10
length = 1

# Write header to the result file
with open('pbenc_vs_cadical_atmost.txt', 'w') as f:
    f.write("Length, PBEnc method SAT, PBEnc method Time (s), Native method SAT, Native method Time (s)\n")

# Loop over different lengths
while length <= bench_length:
    lits = list(range(1, length + 1))
    weights = list(range(1, length + 1))
    max_bound = sum(weights) // 2
    bound = random.randint(1, max_bound if max_bound > 0 else 1)
    print(bound)  # Randomize the bound
    
    # Measure time for PBEnc method
    result_pbenc, time_pbenc = measure_time(lambda: solve_with_pbenc(lits, weights, bound))
    
    # Measure time for native method
    result_native, time_native = measure_time(lambda: solve_with_native(lits, weights, bound))
    
    # Write results to file
    with open('pbenc_vs_cadical_atmost.txt', 'a') as f:
        f.write("{}, {}, {:.4f}, {}, {:.4f}\n".format(length, result_pbenc[0], time_pbenc, result_native[0], time_native))
    print("Length={} completed.".format(length))
    
    length += step

print("Benchmarking completed and results written to pbenc_vs_cadical_atmost.txt")
