import json
import multiprocessing
from pebble import ProcessPool
from concurrent.futures import TimeoutError

# Create a lock for synchronizing file access across processes
file_lock = multiprocessing.Lock()

# Step 1: Create a dictionary inside a dictionary
data = {
    "person_1": {
        "name": "Alice",
        "age": 25,
        "city": "New York"
    },
    "person_2": {
        "name": "Bob",
        "age": 30,
        "city": "Los Angeles"
    },
    "person_3": {
        "name": "Charlie",
        "age": 35,
        "city": "Chicago"
    }
}

# Step 2: Save the initial dictionary to a JSON file with locking
def save_data_with_lock(data, filename='nested_dict.json'):
    with file_lock:  # Acquire lock before file access
        with open(filename, 'w') as json_file:
            json.dump(data, json_file, indent=4)

# Step 2.5: Function to update the dictionary with new data
def update_json_with_lock(new_data, filename='nested_dict.json'):
    with file_lock:  # Acquire lock before file access
        # Load current data from the file
        with open(filename, 'r') as json_file:
            current_data = json.load(json_file)

        # Update the current dictionary with new data
        current_data.update(new_data)

        # Save the updated dictionary back to the file
        with open(filename, 'w') as json_file:
            json.dump(current_data, json_file, indent=4)

# Step 3: Function to load and print data with locking
def load_and_print_with_lock(filename='nested_dict.json'):
    with file_lock:  # Acquire lock before file access
        with open(filename, 'r') as json_file:
            loaded_data = json.load(json_file)

        # Iterate through the outer dictionary and print
        for person, details in loaded_data.items():
            print(f"\n{person}:")
            for key, value in details.items():
                print(f"  {key}: {value}")

# Example worker function for processes
def worker(new_data):
    update_json_with_lock(new_data)
    load_and_print_with_lock()

# Simulating multiprocessing with Pebble
if __name__ == "__main__":
    # Save initial data first
    save_data_with_lock(data)

    # Create new data for two workers
    new_data_1 = {
        "person_4": {
            "name": "Diana",
            "age": 28,
            "city": "San Francisco"
        }
    }

    new_data_2 = {
        "person_5": {
            "name": "Eve",
            "age": 40,
            "city": "Boston"
        }
    }

    # Using Pebble to manage processes
    with ProcessPool(max_workers=2) as pool:
        future1 = pool.schedule(worker, args=(new_data_1,))
        future2 = pool.schedule(worker, args=(new_data_2,))

        # Wait for processes to complete
        try:
            result1 = future1.result(timeout=10)
            result2 = future2.result(timeout=10)
        except TimeoutError:
            print("A process took too long to complete")
