def twos_complement(val, nbits):
    """Compute the 2's complement of int value val"""
    if val < 0:
        val = (1 << nbits) + val
    else:
        if (val & (1 << (nbits - 1))) != 0:
            # If sign bit is set.
            # compute negative value.
            val = val - (1 << nbits)
    return val

# Examples
examples = [(5, 4), (-3, 4), (8, 4)]

for val, nbits in examples:
    result = twos_complement(val, nbits)
    print(f"2's complement of {val} in {nbits}-bits is {result}")
