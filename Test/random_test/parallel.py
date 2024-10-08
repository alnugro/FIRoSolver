from z3 import *
import multiprocessing as mp

# Define Problem 1
def problem1():
    try:
        # Create a Z3 context for Problem 1
        ctx = Context()

        # Create integer variables within the context
        x = Int('x', ctx=ctx)
        y = Int('y', ctx=ctx)

        # Create a solver within the context
        solver = Solver(ctx=ctx)

        # Define constraints one by one within the context
        solver.add(x >= 0)
        solver.add(y >= 0)
        solver.add(x + y == 10)

        # Print debug info
        print(f"Problem 1: Constraints added")

        # Check satisfiability
        if solver.check() == sat:
            model = solver.model()
            print(f"Solution to Problem 1: x = {model[x]}, y = {model[y]}")
        else:
            print("Problem 1 is unsatisfiable")

        # Clean up the context for Problem 1 (optional)
        # ctx.exit()  # No need to call ctx.exit() for Z3 Context

    except Exception as e:
        # Print and propagate the exception
        print(f"Error in Problem 1: {e}")
        raise

# Define Problem 2
def problem2():
    try:
        # Create a Z3 context for Problem 2
        ctx = Context()

        # Create boolean variables within the context
        p = Bool('p', ctx=ctx)
        q = Bool('q', ctx=ctx)

        # Create a solver within the context
        solver = Solver(ctx=ctx)

        # Define constraints one by one within the context
        solver.add(Or(p, q))
        solver.add(Not(And(p, q)))

        # Print debug info
        print(f"Problem 2: Constraints added")

        # Check satisfiability
        if solver.check() == sat:
            model = solver.model()
            print(f"Solution to Problem 2: p = {model[p]}, q = {model[q]}")
        else:
            print("Problem 2 is unsatisfiable")

        # Clean up the context for Problem 2 (optional)
        # ctx.exit()  # No need to call ctx.exit() for Z3 Context

    except Exception as e:
        # Print and propagate the exception
        print(f"Error in Problem 2: {e}")
        raise

# Main program to solve both problems concurrently
if __name__ == "__main__":
    try:
        # Create a process pool
        pool = mp.Pool()

        # Submit each problem to the pool
        result1 = pool.apply_async(problem1)
        result2 = pool.apply_async(problem2)

        # Wait for all processes to finish
        result1.get()
        result2.get()

        # Close the pool
        pool.close()
        pool.join()

    except Exception as e:
        print(f"Error in multiprocessing: {e}")

# No need to exit the context at the end of the script, as Z3 handles its own resource management
