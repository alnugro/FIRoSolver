import time
import string

# Custom Base62 encoding alphabet (alphanumeric characters only)
ALPHABET = string.ascii_letters + string.digits

# Function to convert a number to a custom base (like Base62)
def base_encode(number, base=len(ALPHABET)):
    if number == 0:
        return ALPHABET[0]
    encoded = []
    while number:
        number, rem = divmod(number, base)
        encoded.append(ALPHABET[rem])
    return ''.join(reversed(encoded))

# Get the current time in microseconds and encode it
def generate_unique_id():
    timestamp = int(time.time())
    print(timestamp)
    return base_encode(timestamp)

# Usage
unique_id = generate_unique_id()
print(unique_id)
