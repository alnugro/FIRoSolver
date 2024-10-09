import z3
import time

# Define a medium-sized problem: Let's solve a simple SMT problem where we try to find
# values of 5000 integer variables that satisfy some random linear constraints.

def create_problem(num_vars=3000, ctx=None):
    # Create an array of integer variables with context
    x = [z3.Int(f'x{i}', ctx) for i in range(num_vars)]
    
    solver = z3.Solver(ctx=ctx)
    
    # Add some arbitrary linear constraints
    for i in range(num_vars - 1):
        solver.add(x[i] + x[i + 1] <= 50)  # sum of consecutive variables <= 50
        solver.add(x[i] - x[i + 1] >= 5)   # difference of consecutive variables >= 5
    
    # Add more constraints to increase the complexity
    solver.add(z3.Sum(x) >= 100)  # sum of all variables >= 100
    solver.add(z3.Sum([v*v for v in x]) <= 10000)  # sum of squares of variables <= 10000

    return solver

# Solve without threading
ctx1 = z3.Context()  # Create a separate context for solver1
solver1 = create_problem(ctx=ctx1)
solver1.set("smt.threads", 1)  # No parallelism, single thread
solver1.set("random_seed", 0)  # Set random seed
start_time = time.time()
result1 = solver1.check()
duration1 = time.time() - start_time
print(f"Result without parallelism (random_seed=42): {result1}")
print(f"Duration without parallelism: {duration1:.4f} seconds")

# Solve with threading enabled and max 4 threads
ctx2 = z3.Context()  # Create a separate context for solver2
solver2 = create_problem(ctx=ctx2)
# solver2.set("smt.threads", 4)  # Enable parallelism with 4 threads
solver2.set("random_seed", 42)  # Set random seed for parallel solver
start_time = time.time()
result2 = solver2.check()
duration2 = time.time() - start_time
print(f"Result with parallelism (random_seed=42, 4 threads): {result2}")
print(f"Duration with parallelism: {duration2:.4f} seconds")
