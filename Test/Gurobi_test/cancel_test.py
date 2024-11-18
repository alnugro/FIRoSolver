import gurobipy as gp
import z3
import time
from pebble import ProcessPool, ProcessExpired
from concurrent.futures import TimeoutError, CancelledError

class GurobiTestBench:
    def __init__(self, timeout=5):
        self.timeout = timeout

    def create_model(self):
        """
        Create a more complex Gurobi model for testing.
        """
        model = gp.Model("test")
        variables = []
        for i in range(1000000000):  # Create 1000 variables to make the problem harder
            variables.append(model.addVar(name=f"x_{i}", lb=0))
        
        # Set objective
        model.setObjective(gp.quicksum((i + 1) * variables[i] for i in range(1000)), gp.GRB.MAXIMIZE)
        
        # Add constraints
        for i in range(1000000000):
            model.addConstr(gp.quicksum((j + 1) * variables[j] for j in range(1000)) <= 1000 * (i + 1), f"c_{i}")
        
        return model

    def run_model(self):
        """
        Run the model to prove that it cannot be canceled once started.
        """
        model = self.create_model()
        model.optimize()
        time.sleep(10)  # Simulate long-running process
        return model.status

    def run_with_pebble(self):
        """
        Run the model using Pebble ProcessPool.
        """
        with ProcessPool(max_workers=1) as pool:
            future = pool.schedule(self.run_model, timeout=self.timeout)
            try:
                time.sleep(3)
                future.cancel()
                result = future.result()
                print(f"Model completed with status: {result}")
            except CancelledError:
                print("Future was successfully canceled.")
            except TimeoutError:
                print("Timeout reached. Model is still running and cannot be canceled.")
            except ProcessExpired as error:
                print(f"Process expired: {error}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

class Z3TestBench:
    def __init__(self, timeout=5):
        self.timeout = timeout

    def create_solver(self):
        """
        Create a simple Z3 solver for testing.
        """
        solver = z3.Solver()
        x = z3.Int('x')
        y = z3.Int('y')
        solver.add(x + 2 * y <= 10)
        solver.add(3 * x + y <= 15)
        return solver

    def run_solver(self):
        """
        Run the Z3 solver to prove that it cannot be canceled once started.
        """
        solver = self.create_solver()
        solver.check()  # This can take a long time depending on the problem
        time.sleep(10)  # Simulate long-running process
        return solver.model()

    def run_with_pebble(self):
        """
        Run the solver using Pebble ProcessPool.
        """
        with ProcessPool(max_workers=1) as pool:
            future = pool.schedule(self.run_solver, timeout=self.timeout)
            try:
                time.sleep(0.01)
                future.cancel()
                result = future.result()
                print(f"Solver completed with result: {result}")
            except CancelledError:
                print("Future was successfully canceled.")
            except TimeoutError:
                print("Timeout reached. Solver is still running and cannot be canceled.")
            except ProcessExpired as error:
                print(f"Process expired: {error}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    print("Testing Gurobi Model:")
    bench = GurobiTestBench()
    bench.run_with_pebble()

    print("\nTesting Z3 Solver:")
    z3_bench = Z3TestBench()
    z3_bench.run_with_pebble()
