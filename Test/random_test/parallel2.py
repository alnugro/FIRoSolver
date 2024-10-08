import concurrent.futures
import time
import z3

# Function to run Z3 solver with a given random seed
def solve_with_seed(seed):
    # Create a new context and define the problem
    ctx = z3.Context()
    x = z3.Int('x', ctx=ctx)
    y = z3.Int('y', ctx=ctx)
    problem = [
        x + y == 2500,
        x - y == 212131,
    ]
    solver = z3.Solver(ctx=ctx)
    solver.set("random_seed", seed)
    solver.add(problem)

    try:
        if solver.check() == z3.sat:
            return solver.model()
        else:
            return None
    except z3.Z3Exception as e:
        print(f"Solver with seed {seed} generated an exception: {e}")
        return f"Solver with seed {seed} generated an exception: {e}"

# Number of parallel solver instances
num_instances = 8
# Random seeds for different instances
random_seeds = [i for i in range(num_instances)]

# Function to run multiple solver instances in parallel
def parallel_solve(random_seeds):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(solve_with_seed, seed): seed for seed in random_seeds}

        for future in concurrent.futures.as_completed(futures):
            seed = futures[future]
            try:
                result = future.result()
                if result and isinstance(result, z3.ModelRef):
                    print(f"Solution found with seed {seed}: {result}")
                    # Cancel other futures if a solution is found
                    for f in futures:
                        f.cancel()
                    return result
                elif isinstance(result, str):
                    print(result)  # Print exception message
            except Exception as exc:
                print(f"Solver with seed {seed} generated an exception: {exc}")

    return None

# Measure time for parallel run
start_time_parallel = time.time()
parallel_solution = parallel_solve(random_seeds)
end_time_parallel = time.time()

# Print results
if parallel_solution and isinstance(parallel_solution, z3.ModelRef):
    print(f"Parallel solution: {parallel_solution}")
else:
    print("Parallel instances found no solution or encountered errors.")

print(f"Time taken (parallel instances): {end_time_parallel - start_time_parallel} seconds")
