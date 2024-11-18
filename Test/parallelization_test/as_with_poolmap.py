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
            return "SAT", number
        else:
            print(f"Finished processing number {number}, result: UNSAT")
            return "UNSAT", number
        
    def task_done_as_step(self, pool, futures, sat_found):
        def callback(future):
            try:
                result, number = future.result()
                if result == "SAT":
                    print(f"SAT found: {number}")
                    sat_found.set()
                    future_index = futures.index(future)
                    # Cancel all other futures once a SAT is found
                    for f in futures[future_index+1:]:
                        if not f.done():
                            if not f.cancel():
                                print(f"failed to cancel future: {f}, stopping pool")
                                pool.stop()
                else:
                    print(f"UNSAT: {number}")
            except CancelledError:
                print(f"Future was cancelled: {future}")
            except ProcessExpired as error:
                print(f"Process expired: {error}")
            except Exception as error:
                print(f"Error: {error}")
                traceback.print_exc()
        return callback

       

    def search_as_step(self):
        """Main function to search for the smallest SAT number."""
        current_step = 0
        found_sat = None
        prev_unsat = -1
        max_number = 60  # Define an upper limit for the search
        sat_found = multiprocessing.Event()

        am_to_check = list(range(0, max_number+1, self.step_size))

        with ProcessPool(max_workers=4) as pool:
            futures = [pool.schedule(self.satisfiability_solver, args=[am_count]) for am_count in am_to_check]

            for future in futures:
                future.add_done_callback(self.task_done_as_step(pool, futures, sat_found))
        
            done, not_done = wait(futures, return_when=ALL_COMPLETED)
            sat_list = []
            unsat_list = []
            for future in futures:
                    try:
                        result, number = future.result()
                        print(f"number: {number}, result: {result}")
                        self.update_json_file(number=number, result=result)
                        if result == "SAT":
                            sat_list.append(number)
                        else:
                            unsat_list.append(number)
                    except CancelledError:
                        completed_index = futures.index(future)
                        number = am_to_check[completed_index]
                        print(f"number: {number}, result: cancelled")
                        self.update_json_file(number=number, result="Cancelled")
                    except TimeoutError:
                        pass
                    except Exception as e:
                        # Handle other exceptions if necessary
                        print(f"Task raised an exception: {e}")
                        traceback.print_exc()
            

            # Refinement step
            if sat_found.is_set():
                print(f"Found SAT at {found_sat}, refining search...")
                lower_bound = max(unsat_list) + 1
                upper_bound = min(sat_list) - 1
                print(f"Refining search between {lower_bound} and {upper_bound}...")
                refine_search_sat = self.refine_search(lower_bound, upper_bound)
                if refine_search_sat is not None:
                    found_sat = refine_search_sat

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
    finder.search_as_step()
