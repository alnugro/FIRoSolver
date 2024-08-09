import platform
import struct
import sys

def check_python_bitness():
    # Check using the struct module
    bitness_struct = struct.calcsize("P") * 8

    # Check using the platform module
    bitness_platform = platform.architecture()[0]

    # Python executable bitness
    bitness_sys = "64-bit" if sys.maxsize > 2**32 else "32-bit"

    print(f"Python bitness (struct): {bitness_struct}-bit")
    print(f"Python bitness (platform): {bitness_platform}")
    print(f"Python executable bitness: {bitness_sys}")

if __name__ == "__main__":
    check_python_bitness()
