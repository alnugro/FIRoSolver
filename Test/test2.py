from z3 import *

# Function to decompose large coefficients
def decompose_coefficients(literal, coeff, max_coeff_value):
    coeffs = []
    while coeff > 0:
        if coeff >= max_coeff_value:
            coeffs.append((literal, max_coeff_value))
            coeff -= max_coeff_value
        else:
            coeffs.append((literal, coeff))
            coeff = 0
    return coeffs

# Define variables and large coefficients
variables = [Bool(f'hm{i}') for i in range(5)]
large_coeffs = [6, 7, 8, 9, 10]

# Define maximum coefficient value
max_coeff_value = 4

# Decompose coefficients and construct pairs
pb_pairs = []
for var, coeff in zip(variables, large_coeffs):
    pb_pairs.extend(decompose_coefficients(var, coeff, max_coeff_value))

print(large_coeffs)
print(pb_pairs)

# Example bound for the constraint
bound = 20

# Create the solver and add the pseudo-boolean constraint
solver = Solver()
solver.add(PbGe(pb_pairs, bound))

# Check satisfiability
if solver.check() == sat:
    print("SAT")
    print(solver.model())
else:
    print("UNSAT")
