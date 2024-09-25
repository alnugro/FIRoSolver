from pebble import ThreadPool
import random
import time

class ParallelRunner:
    def __init__(self, num_workers):
        self.num_workers = num_workers
        self.results = {}
        
    def worker_func(self, input_value):
        """Function to be executed in parallel."""
        print(f"running with input :{input_value}")
        time.sleep(input_value)  # Wait for input_value seconds
        result = random.randint(1, 100)  # Return a random integer
        print(f"input is done:{input_value}")

        return (input_value, result)
    
    def run(self, inputs):
        """Run the worker function in parallel over the input list."""
        with ThreadPool(self.num_workers) as pool:
            future = pool.map(self.worker_func, inputs)
            iterator = future.result()
            try:
                for res in iterator:
                    input_value, result = res
                    self.results[input_value] = result
            except Exception as error:
                print("Error:", error)
    
    def get_results(self):
        """Get the results dictionary."""
        return self.results

if __name__ == '__main__':
    test_cases = [
        # {'inputs': [0, 1, 2, 3, 4], 'num_workers': 2},
        # {'inputs': [1, 2, 3, 4, 5], 'num_workers': 3},
        {'inputs': [2, 4, 6, 8, 10], 'num_workers': 2},
    ]

    for idx, test_case in enumerate(test_cases):
        print(f"Test case {idx+1}: Inputs={test_case['inputs']}, Num Workers={test_case['num_workers']}")
        runner = ParallelRunner(num_workers=test_case['num_workers'])
        start_time = time.time()
        runner.run(test_case['inputs'])
        end_time = time.time()
        results = runner.get_results()
        print("Results:")
        for input_value in sorted(results):
            print(f"Input {input_value}: Result {results[input_value]}")
        print(f"Execution time: {end_time - start_time:.2f} seconds\n")
