import argparse
import json
from pebble import ProcessPool, ProcessExpired
from concurrent.futures import TimeoutError, CancelledError, wait, ALL_COMPLETED
import multiprocessing
import time
import traceback
from filelock import FileLock
import math
class SATFinder:
    def __init__(self, workers=4, step_size=5):
        self.workers = 2
        self.step_size = 4
        self.sat_list = [None for _ in range(self.workers)]
        self.JSON_FILE = 'search_space.json'
        self.LOCK_FILE = 'search_space.lock'
        self.initialize_json_file()

    def initialize_json_file(self):
        """Initialize the JSON file with default values if it doesn't exist."""
        data = {
            'search_space': [],
            'results': {}
        }
        with FileLock(self.LOCK_FILE):
            with open(self.JSON_FILE, 'w') as f:
                json.dump(data, f)

    def update_json_file(self, number=None, result=None, search_space=None):
        """Update the JSON file with new search space or results."""
        with FileLock(self.LOCK_FILE):
            with open(self.JSON_FILE, 'r') as f:
                data = json.load(f)
            if search_space is not None:
                data['search_space'].extend(search_space)
            if number is not None and result is not None:
                data['results'][str(number)] = result
            with open(self.JSON_FILE, 'w') as f:
                json.dump(data, f)

    def read_json_file(self):
        """Read data from the JSON file."""
        with FileLock(self.LOCK_FILE):
            with open(self.JSON_FILE, 'r') as f:
                data = json.load(f)
        return data

    def satisfiability_solver(self, number):
        """Dummy satisfiability solver function."""
        # Simulate computation time
        sleep_time = abs(11 - number)
        print(f"Processing number {number}, sleeping for {sleep_time} seconds...")
        time.sleep(sleep_time)
        # For demonstration, let's say numbers >= 21 are SAT
        if number >= 4:
            print(f"Finished processing number {number}, result: SAT")
            return True
        else:
            print(f"Finished processing number {number}, result: UNSAT")
            return False

       

    def search_sat_number(self):
        """Main function to search for the smallest SAT number."""
        current_step = 0
        found_sat = None
        prev_unsat = -1
        max_number = 100  # Define an upper limit for the search
        sat_found = multiprocessing.Event()
        

        # Create a list of pools, one for each worker
        pools = [ProcessPool(max_workers=1) for _ in range(self.workers)]
        print(f"workers: {self.workers}, step_size: {self.step_size}")
        try:
            while current_step <= max_number and not found_sat:
                self.sat_list = [None for _ in range(self.workers)]
                print(f"Current step: {current_step}")
                self.sat_list = [None for _ in range(self.workers)]
                numbers_to_check = [current_step + self.step_size * i for i in range(self.workers)]
                print(f"Checking numbers: {numbers_to_check}")
                # Ensure we don't exceed the max_number
                numbers_to_check = [n for n in numbers_to_check if n <= max_number]
                self.update_json_file(search_space=numbers_to_check)

                futures = []
                for idx, number in enumerate(numbers_to_check):
                    future = pools[idx].schedule(self.satisfiability_solver, args=(number,))
                    futures.append(future)

                # Wait for any future to complete
                for future in futures:
                    future.add_done_callback(self.task_done(futures, pools, numbers_to_check))

                print(f"futures: {futures}")

                done, not_done = wait(futures, return_when=ALL_COMPLETED)

                #sleep to avoid race condition with task_done
                time.sleep(0.5)
                min_sat = min(i for i, x in enumerate(self.sat_list) if x == "sat")
                for future in futures:
                    try:
                        result = future.result()
                        completed_index = futures.index(future)
                        number_index = completed_index + numbers_to_check.index(current_step)
                        number = numbers_to_check[number_index]
                        print(f"number: {number}, result: {result}")
                        self.update_json_file(number=number, result=result)
                        if result:
                                if completed_index == min_sat:
                                    found_sat = number
                                    prev_unsat = found_sat - self.step_size if found_sat-self.step_size >= 0 else 0
            
                    except CancelledError:
                        completed_index = futures.index(future)
                        number_index = completed_index + numbers_to_check.index(current_step)
                        number = numbers_to_check[number_index]
                        print(f"number: {number}, result: cancelled")

                        self.update_json_file(number=number, result="Cancelled")
                    except TimeoutError:
                        pass
                    except Exception as e:
                        # Handle other exceptions if necessary
                        print(f"Task raised an exception: {e}")
                        traceback.print_exc()

                if not found_sat:
                    current_step += self.step_size * self.workers

            # Stop all pools
            for idx, pool in enumerate(pools):
                pool.stop()
                print(f"Stopped pool {idx}")

            # Refinement step
            if found_sat is not None:
                print(f"Found SAT at {found_sat}, refining search...")
                lower_bound = prev_unsat + 1
                upper_bound = found_sat - 1
                refine_search_sat = self.refine_search(lower_bound, upper_bound)
                if refine_search_sat is not None:
                    found_sat = refine_search_sat

        finally:
            # Ensure all pools are properly closed
            for pool in pools:
                pool.close()
                pool.join()

        print(f"The smallest SAT number is: {found_sat}")

    def task_done(self, futures, pools, numbers_to_check):
        def callback(future):
            try:
                result = future.result()  # blocks until results are ready
                print(f"Task done: {result}")
                # Find the index of the current future
                completed_index = futures.index(future)
                if result:
                    self.sat_list[completed_index] = "sat"
                    print(f"sat_list: {self.sat_list}")

                    # If the task is satisfiable, check if the previous task was unsatisfiable
                    if completed_index != 0:
                        # If the previous task was unsatisfiable, stop all pools
                        if self.sat_list[completed_index - 1] == "unsat":
                            print(f"Stopping all pools")
                            for idx, f in enumerate(futures):
                                if not f.done():
                                    if not f.cancel():
                                        # Stop the pool associated with this future
                                        print(f"Stopping pool for future at index {idx}.")
                                        pools[idx].stop()

                    # Cancel all futures beyond the completed one
                    for idx, f in enumerate(futures[completed_index + 1:], start=completed_index + 1):
                        if not f.done():
                            if not f.cancel():
                                # Stop the pool associated with this future
                                print(f"Stopping pool for future at index {idx}.")
                                pools[idx].stop()
                else:
                    # If the task is unsatisfiable, update the previous unsat number
                    self.sat_list[completed_index] = "unsat"
                    if completed_index != self.workers - 1:
                        if self.sat_list[completed_index + 1] == "sat":
                            print(f"Stopping all pools")
                            for idx, f in enumerate(futures):
                                if not f.done():
                                    if not f.cancel():
                                        # Stop the pool associated with this future
                                        print(f"Stopping pool for future at index {idx}.")
                                        pools[idx].stop()

                    # Cancel all futures before the completed one
                    for idx, f in enumerate(futures[:completed_index - 1], start=0):
                        if not f.done():
                            if not f.cancel():
                                # Stop the pool associated with this future
                                print(f"Stopping pool for future at index {idx}.")
                                pools[idx].stop()

            except CancelledError:
                completed_index = futures.index(future)
                print(f"Task at index {completed_index} was cancelled.")
            except TimeoutError:
                completed_index = futures.index(future)
                print(f"Task at index {completed_index} timed out.")
            except ProcessExpired as error:
                completed_index = futures.index(future)
                print(f"Task at index {completed_index} raised a ProcessExpired error: {error}")
            except Exception as error:
                print(f"Task raised an exception: {error}")
                traceback.print_exc()  # Print the full traceback to get more details

        return callback

    def refine_search(self, lower_bound, upper_bound):
        """Refine the search between lower_bound and upper_bound using binary search with n parallel workers."""
        found_sat = None
        # Use a single pool with max_workers=self.workers
        pools = [ProcessPool(max_workers=1) for _ in range(self.workers)]
        res = [None for i in range(lower_bound, upper_bound+1)]
        original_lower_bound = lower_bound
        print(f"Refining search between {lower_bound} and {upper_bound}...")
        try:
            while lower_bound <= upper_bound:
                self.sat_list = [None for _ in range(self.workers)]
                used_worker = 0
                # Compute midpoints
                midpoints = [lower_bound + (upper_bound - lower_bound) * (i + 1) / (self.workers + 1) for i in range(self.workers)]
                midpoints = list(map(math.floor, midpoints))                
                # Remove duplicates and ensure midpoints are within bounds
                midpoints = sorted(set(midpoints))
                midpoints = [m for m in midpoints if lower_bound <= m <= upper_bound]
                print(f"midpoints: {midpoints}")
                print(f"Testing midpoints: {midpoints}")
                if not midpoints:
                    break  # No midpoints to test

                self.update_json_file(search_space=midpoints)

                # Schedule tasks
                futures = []
                for idx, number in enumerate(midpoints):
                    used_worker += 1
                    future = pools[idx].schedule(self.satisfiability_solver, args=(number,))
                    futures.append((future))

                print(f"Used worker: {used_worker}")

                for future in futures:
                    future.add_done_callback(self.task_done(futures, pools, midpoints))

                # Wait for all futures to complete
                done, not_done = wait(futures, return_when=ALL_COMPLETED)

                # Collect results
                sat_numbers = []
                unsat_numbers = []
                for future in futures:
                    try:
                        result = future.result()
                        number = midpoints[futures.index(future)]
                        print(f"number: {number}, result: {result}")
                        res_index = number - original_lower_bound
                        res[res_index] = result
                        self.update_json_file(number=number, result=result)
                        if result:
                            sat_numbers.append(number)
                        else:
                            unsat_numbers.append(number)
                            
                    except CancelledError:
                        number = midpoints[futures.index(future)]
                        res_index = number - original_lower_bound
                        res[res_index] = "Cancelled"
                        self.update_json_file(number=number, result="Cancelled")

                    except Exception as e:
                        print(f"Task raised an exception for number {number}: {e}")
                        traceback.print_exc()

                if sat_numbers:
                    # Found SAT at midpoints
                    min_sat = min(sat_numbers)
                    found_sat = min_sat
                    # Adjust upper_bound
                    upper_bound = min_sat - 1

                if unsat_numbers:
                    max_unsat = max(unsat_numbers)
                    lower_bound = max_unsat + 1

                if not sat_numbers and not unsat_numbers:
                    # No results found, break the loop
                    raise Exception("Somehow no results found in the search space.")
                
                for res_idx, res_val in enumerate(res):
                    print(f"res_idx: {res_idx}, res_val: {res_val}")
                    if res_val == True:
                        if res[res_idx-1] == False:
                            found_sat = res_idx + original_lower_bound
                            upper_bound = lower_bound - 1
                            print(f"Found SAT at {found_sat}, breaking the loop")
                            break
                        break
                        

                # Check if lower_bound > upper_bound
                if lower_bound > upper_bound:
                    break

        finally:
            # Ensure the pool is properly closed
            for pool in pools:
                pool.close()
                pool.join()

        return found_sat



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAT Number Finder")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes")
    parser.add_argument("--step", type=int, default=5, help="Step size for searching")
    args = parser.parse_args()

    finder = SATFinder(workers=args.workers, step_size=args.step)
    finder.search_sat_number()
