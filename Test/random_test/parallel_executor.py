from pebble import ProcessPool, ProcessExpired
from concurrent.futures import TimeoutError, CancelledError, wait, ALL_COMPLETED
import traceback
import time


class TaskExecutor:
    def __init__(self, gurobi_thread, z3_thread, pysat_thread, solver_timeout=10):
        self.gurobi_thread = gurobi_thread
        self.z3_thread = z3_thread
        self.pysat_thread = pysat_thread
        self.solver_timeout = solver_timeout

    def execute_parallel_tasks(self):
        pools = []  # To store active pools for cleanup
        futures_gurobi = []  # List to store Gurobi futures
        futures_z3 = []  # List to store Z3 futures
        futures_pysat = []  # List to store PySAT futures

        try:
            # Conditionally create the Gurobi pool
            if self.gurobi_thread > 0:
                pool_gurobi = ProcessPool(max_workers=self.gurobi_thread)
                pools.append(pool_gurobi)
                for i in range(self.gurobi_thread):
                    future = pool_gurobi.schedule(self.gurobi_task, args=(i + 1,), timeout=self.solver_timeout)
                    futures_gurobi.append(future)
                    future.add_done_callback(self.task_done('Gurobi', futures_gurobi))
            else:
                pool_gurobi = None

            # Conditionally create the Z3 pool
            if self.z3_thread > 0:
                pool_z3 = ProcessPool(max_workers=self.z3_thread)
                pools.append(pool_z3)
                for i in range(self.z3_thread):
                    future = pool_z3.schedule(self.solve_z3_with_seed, args=(i + 1,), timeout=self.solver_timeout)
                    futures_z3.append(future)
                    future.add_done_callback(self.task_done('Z3', futures_z3))
            else:
                pool_z3 = None

            # Conditionally create the PySAT pool
            if self.pysat_thread > 0:
                pool_pysat = ProcessPool(max_workers=self.pysat_thread)
                pools.append(pool_pysat)
                for i in range(self.pysat_thread):
                    future = pool_pysat.schedule(self.solve_pysat_with_seed, args=(i + 1,), timeout=self.solver_timeout)
                    futures_pysat.append(future)
                    future.add_done_callback(self.task_done('PySAT', futures_pysat))
            else:
                pool_pysat = None

            # Wait for all futures to complete, handling timeouts as well
            all_futures = futures_gurobi + futures_z3 + futures_pysat
            done, not_done = wait(all_futures, return_when=ALL_COMPLETED)


        finally:
            # Ensure all pools are properly cleaned up
            for pool in pools:
                pool.stop()
                pool.join()

    def task_done(self, solver_name, futures):
        def callback(future):
            try:
                result = future.result()  # blocks until results are ready
                print(f"{solver_name} task done with result: {result}")

                # Cancel all other processes for this solver (only within the same group)
                for f in futures:
                    if f is not future and not f.done():  # Check if `f` is a `Future`
                        f.cancel()
                        print(f"Cancelled another {solver_name} process")

                # Handle the result (custom logic depending on the solver)
                if isinstance(result, list) and result and result[0] == "unsat":
                    print(f"{solver_name} task returned unsat.")
                elif isinstance(result, list) and result and result[0] == "unknown":
                    print(f"{solver_name} task returned unknown.")
                else:
                    print(f"Solution found for {solver_name}: {result}")

            except CancelledError:
                print(f"{solver_name} task was cancelled.")
            except TimeoutError:
                print(f"{solver_name} task timed out.1")
            except ProcessExpired as error:
                print(f"{solver_name} process {error.pid} expired.")
            except Exception as error:
                print(f"{solver_name} task raised an exception: {error}")
                traceback.print_exc()  # Print the full traceback to get more details

        return callback

    def get_solver_name(self, future, futures_gurobi, futures_z3, futures_pysat):
        """Helper function to identify which solver a future belongs to."""
        if future in futures_gurobi:
            return "Gurobi"
        elif future in futures_z3:
            return "Z3"
        elif future in futures_pysat:
            return "PySAT"
        return "Unknown"

    def gurobi_task(self, seed):
        print(f"gurobi called with seed {seed}")
        # Simulated Gurobi task
        time.sleep(1)  # Simulate computation based on seed
        print(f"gurobi called again with seed {seed}")

        return f"Gurobi result with seed {seed}"

    def solve_z3_with_seed(self, seed):
        print(f"z3 called with seed {seed}")
        # Simulated Z3 task with seed
        time.sleep(seed+1)  # Simulate computation based on seed
        print(f"z3 called again with seed {seed}")

        return f"Z3 result with seed {seed}"

    def solve_pysat_with_seed(self, seed):
        print(f"pysat called with seed {seed}")
        # Simulated PySAT task with seed
        time.sleep(3)  # Simulate computation based on seed
        print(f"pysat called again with seed {seed}")

        return f"PySAT result with seed {seed}"


# Example usage
executor = TaskExecutor(gurobi_thread=1, z3_thread=5, pysat_thread=6, solver_timeout=2)
executor.execute_parallel_tasks()
