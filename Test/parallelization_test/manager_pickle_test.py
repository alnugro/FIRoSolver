from pebble import ProcessPool
from multiprocessing import Manager
import time

class Example:
    def __init__(self, flag_value):
        # Use Manager to create a shared Value for flag
        manager = Manager()
        self.flag = manager.Value('b', flag_value)  # 'b' denotes a boolean

    def worker_task(self, flag):
        while True:
            print("Worker sees flag:", flag.value)
            if not flag.value:
                print("Worker stopping due to flag.")
                break
            time.sleep(1)  # Simulate ongoing work

    def start_worker(self):
        with ProcessPool() as pool:
            future = pool.schedule(self.worker_task, args=(self.flag,))
            print("Worker started. You can change the flag now.")

            # Modify the flag after scheduling
            time.sleep(5)
            self.flag.value = False  # This change is visible to the worker
            future.result()  # Block until the task completes

# Usage
if __name__ == '__main__':
    example = Example(flag_value=True)
    example.start_worker()
