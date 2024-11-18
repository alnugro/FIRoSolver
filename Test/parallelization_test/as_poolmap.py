from pebble import ProcessPool, ProcessExpired
from concurrent.futures import TimeoutError, CancelledError
import time
import traceback
import os

# Dummy solver function to check if a number is satisfiable (SAT)
def dummy_solver(number):
    print(f"Processing number: {number}")
    worker_pid = os.getpid()
    print(f"Worker PID: {worker_pid} - Processing number: {number}")
    time.sleep(abs(number - 8))  # Simulate some processing time
    if number > 38:
        print(f"number: {number} is SAT")
        return "SAT", number
    else:
        print(f"number: {number} is UNSAT")
        return "UNSAT", number

# List of numbers to evaluate
numbers = list(range(0, 41, 4))  # [0, 4, 8, 12, ..., 40]

# Callback function to handle results
def handle_result(pool):
    def callback(future):
        try:
            result, number = future.result()
            if result == "SAT":
                print(f"SAT found: {number}")
                # Cancel all other futures once a SAT is found
                for f in futures:
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

if __name__ == "__main__":
    # Create a process pool
    with ProcessPool(max_workers=4) as pool:
        # Submit tasks using map, getting a future for each task
        futures = [pool.schedule(dummy_solver, args=[number]) for number in numbers]
        print("Processing...")
        # Add done callback to each future
        for future in futures:
            future.add_done_callback(handle_result(pool))

        # Wait for the process pool to complete all tasks or be stopped
        pool.close()
        pool.join()

    print("Processing complete.")
