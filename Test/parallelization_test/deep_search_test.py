import threading
import time
from pebble import ProcessPool, ProcessExpired, ThreadPool
from concurrent.futures import TimeoutError, CancelledError, wait, ALL_COMPLETED
import traceback

class DeepSearchSolver:
    def __init__(self, worker, gurobi_thread, order_upperbound, timeout=30):
        # Instance variables
        self.worker = worker
        self.gurobi_thread = gurobi_thread
        self.order_upperbound = order_upperbound
        self.timeout = timeout  # Set a timeout for tasks

        # Initialize variables used in deep_search
        self.h_zero_search_space = []
        self.adder_m_cost = []
        self.new_sat_found = False
        self.target_result_best = None
        self.h_zero_best = None
        self.best_adderm = None
        # Do not include threading locks or events in instance variables

    def dummy_solver(self, presolve_result, am_cost, h_zero, thread, idx):
        # Simulate computation time
        print(f"Running {idx} for h_zero {h_zero}, adder_m_cost {am_cost}")
        time.sleep(am_cost%10)
        print(f" {idx} done")

        # For demonstration, assume that if am_cost is even and >= 0, we return 'sat', else 'unsat'
        if am_cost + h_zero % 2 > 10 and am_cost >= 0:
            satisfiability = 'sat'
            target_result = {'some': 'result'}
        else:
            satisfiability = 'unsat'
            target_result = None

        return {
            'satisfiability': satisfiability,
            'target_result': target_result,
            'h_zero': h_zero,
            'adder_m_cost': am_cost,
            'index': idx
        }
    

    def deep_search(self, presolve_result, input_data_dict):
        sat_found_event = threading.Event()
        task_lock = threading.RLock()  # Use RLock to prevent deadlocks
        failed_cancel = threading.Event()
        # Unpacking variables from the dictionary
        total_adder_s = input_data_dict['total_adder_s']
        adder_s_h_zero_best = input_data_dict['adder_s_h_zero_best']

        print(f"presolve result {presolve_result}")
        print(f"input data dict {input_data_dict}")
        
        # Initialize h_zero_search_space and adder_m_cost
        self.h_zero_search_space = [val for val in range(adder_s_h_zero_best)]
        self.adder_m_cost = [
            total_adder_s - self.order_upperbound + (2 * val) - 1
            for val in self.h_zero_search_space
        ]
        print(f"Initial adder_m_cost: {self.adder_m_cost}")

        if all(val < 0 for val in self.adder_m_cost):
            print("@MSG@ : All adder_m_cost values are negatives. No Better solution possible.")
            return None, None, None

        print(f"h_zero_search_space is {self.h_zero_search_space}")
        print(f"adder_m_cost is {self.adder_m_cost}")

        # Initialize vars
        self.target_result_best = None
        self.h_zero_best = None
        self.best_adderm = None
        done = False

        # Create pools
        pools = [ProcessPool(max_workers=1) for _ in range(self.worker)]

        try:
            while not done:
                if failed_cancel.is_set():
                    # Cancel failed, need to restart the loop wait a bit
                    print("Failed cancel, waiting before restarting loop.")
                    print(f"pool status: {pools}")
                    time.sleep(5)
                    print(f"pool status: {pools}")
                    failed_cancel.clear()

                print("\n\nStarting new iteration\n\n")
                # Check if all adder_m_cost values are negative
                if all(val < 0 for val in self.adder_m_cost):
                    print("All adder_m_cost values are negative. Ending search.")
                    break

                print("Searching for adder_m_cost >= 0")

                # Get indices where adder_m_cost >= 0
                indices_to_process = [
                    i for i, val in enumerate(self.adder_m_cost) if val >= 0
                ]
                print(f"indices_to_process: {indices_to_process}")

                if not indices_to_process:
                    print("No adder_m_cost values >= 0. Ending search.")
                    break

                # Initialize variables for task management
                threads_per_worker = self.gurobi_thread // self.worker

                # Dictionary to keep track of running tasks
                futures_dict = {}
                futures = [None for _ in range(min(self.worker, len(indices_to_process)))]
                
                # Function to submit new tasks
                def submit_task(idx):
                    with task_lock:
                        h_zero_val = self.h_zero_search_space[idx]
                        target_adderm = self.adder_m_cost[idx]

                        if target_adderm < 0:
                            return  # Skip if adder_m_cost is negative
                        future_index = None
                        for i, future in enumerate(futures):
                            if future is None:
                                future_index = i
                        
                        

                        if all(future is not None for future in futures):
                            for future in futures:
                                if future.done():
                                    future_index = futures.index(future)
                                    break



                        print(f"pool_index: {future_index}, idx: {idx}, h_zero: {h_zero_val}, adder_m_cost: {target_adderm}")
                        future = pools[future_index].schedule(
                            self.dummy_solver,
                            args=(presolve_result, target_adderm, h_zero_val, threads_per_worker, idx,),
                            timeout=self.timeout
                        )
                        futures_dict[idx] = future
                        futures[future_index] = future
                        future.add_done_callback(
                            self.task_done_deep_search(
                                idx, futures_dict, indices_to_process, task_lock, submit_task, sat_found_event, pools, failed_cancel, futures
                            )
                        )

                        print(f"running futures {futures}")


                # Submit initial tasks to fill up the worker slots
                with task_lock:
                    for _ in range(min(self.worker, len(indices_to_process))):
                        idx = indices_to_process.pop(0)
                        submit_task(idx)

                # Wait for sat_found_event or until all tasks are done
                while not sat_found_event.is_set() and futures_dict:
                    time.sleep(0.1)  # Sleep briefly to avoid busy waiting

                if sat_found_event.is_set():
                    # Reduce adder_m_cost by 1 for all h_zero's
                    self.adder_m_cost = [
                        val - 1 if val >= 0 else val for val in self.adder_m_cost
                    ]
                    print(f"Reduced adder_m_cost: {self.adder_m_cost}")
                    sat_found_event.clear()
                    # Need to restart the loop
                    continue
                else:
                    # No 'sat' found, and all tasks are done
                    done = True
                    break

        except Exception as e:
            print(f"Exception occurred: {e}")
            traceback.print_exc()

        finally:
            for pool in pools:
                pool.stop()
                pool.join()

        if self.target_result_best is None:
            print("No solution from deep search")
        else:
            print(f"min_total_adder is {self.target_result_best}")
            print(f"best adderm is {self.best_adderm}")
            print(f"h_zero_best is {self.h_zero_best}")
        return self.target_result_best, self.best_adderm, self.h_zero_best

    def task_done_deep_search(self, idx, futures_dict, indices_to_process, task_lock, submit_task, sat_found_event, pools, failed_cancel, futures):
        def callback(future):
            try:
                result = future.result()
                satisfiability_loc = result['satisfiability']
                target_result = result['target_result']
                h_zero_val = result['h_zero']
                adder_m_cost_result = result['adder_m_cost']

                future_index = futures.index(future)
                futures[future_index] = None

                with task_lock:
                    # Remove the completed task from futures_dict
                    del futures_dict[idx]

                if satisfiability_loc == 'sat':
                    # Found a satisfiable solution
                    print(f"worker {future_index}: Found SAT for h_zero {h_zero_val}, adder_m_cost {adder_m_cost_result}")
                    self.target_result_best = target_result
                    self.h_zero_best = h_zero_val
                    self.best_adderm = adder_m_cost_result
                    sat_found_event.set()

                    # Cancel other running tasks
                    with task_lock:
                        for other_idx, other_future in list(futures_dict.items()):
                            if not other_future.done():
                                if not other_future.cancel():
                                    failed_cancel.set()
                                    print(f"Cancelling task for index {other_idx}.")
                                    pools[other_idx].stop()
                                    pools[other_idx].join()
                                    pools[other_idx] = ProcessPool(max_workers=1)
                                            
                        futures_dict.clear()

                else:
                    # 'unsat', set adder_m_cost[idx] = -1
                    print(f"worker {future_index}: Found UNSAT for h_zero {h_zero_val}, adder_m_cost {adder_m_cost_result}")
                    self.adder_m_cost[idx] = -1

                    with task_lock:
                        # If there are more indices to process, submit a new task
                        if indices_to_process:
                            next_idx = indices_to_process.pop(0)
                            submit_task(next_idx)
                        elif not futures_dict:
                            # No more tasks running and no more indices, proceed to next iteration
                            sat_found_event.set()

            except CancelledError:
                print(f"Task at index {idx} was cancelled.")
                future_index = futures.index(future)
                futures[future_index] = None
            except TimeoutError:
                print(f"Task at index {idx} timed out.")
            except ProcessExpired as error:
                print(f"Task at index {idx} raised a ProcessExpired error: {error}")
            except Exception as error:
                print(f"Task raised an exception: {error}")
                traceback.print_exc()  # Print the full traceback to get more details
        return callback

def main():
    # Create an instance of the class
    worker = 2             # Number of workers
    gurobi_thread = 4      # Total number of threads available
    order_upperbound = 40  # Example value for order_upperbound

    solver = DeepSearchSolver(worker, gurobi_thread, order_upperbound, timeout=30)

    # User assigns h_zero_search_space and adder_m_cost manually
    input_data_dict = {
        'best_adderm_from_s': 10,
        'total_adder_s': 50,
        'adder_s_h_zero_best': 5
    }

    presolve_result = {
        'min_adderm_without_zero': 2  # Dummy presolve_result
    }

    # Call deep_search
    target_result_best, best_adderm, h_zero_best = solver.deep_search(presolve_result, input_data_dict)

    # Print results
    print(f"Target result best: {target_result_best}")
    print(f"Best adderm: {best_adderm}")
    print(f"h_zero best: {h_zero_best}")

if __name__ == "__main__":
    main()
