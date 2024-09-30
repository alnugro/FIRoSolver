import json

# Step 1: Create a dictionary inside a dictionary
data = {
    1: {
        "name": "Alice",
        "age": 25,
        "city": "New York"
    },
    # "person_2": {
    #     "name": "Bob",
    #     "age": 30,
    #     "city": "Los Angeles"
    # },
    # "person_3": {
    #     "name": "Charlie",
    #     "age": 35,
    #     "city": "Chicago"
    # }
}

# Step 2: Save the initial dictionary to a JSON file
with open('nested_dict.json', 'w') as json_file:
    json.dump(data, json_file, indent=4)

# **New Step (Between Step 2 and Step 3):**
# Step 2.5: Load the existing data, update it with a new dictionary, and save it back

# New dictionary to be added
new_data = {
    2: {
        "name": "Diana",
        "age": 28,
        "city": "San Francisco"
    }
}

# Load the current JSON data
with open('nested_dict.json', 'r') as json_file:
    current_data = json.load(json_file)

# Update the current dictionary with the new dictionary
current_data.update(new_data)

# Save the updated dictionary back to the JSON file
with open('nested_dict.json', 'w') as json_file:
    json.dump(current_data, json_file, indent=4)

# Step 3: Define a function to load and iterate through the dictionary
def load_and_print(filename):
    with open(filename, 'r') as json_file:
        loaded_data = json.load(json_file)

    # Iterate through the outer dictionary
    for person, details in loaded_data.items():
        print(f"\n{person}:")
        
        # Iterate through the inner dictionary
        for key, value in details.items():
            print(f"  {key}: {value}")

# Step 4: Load and print the saved data
load_and_print('nested_dict.json')
