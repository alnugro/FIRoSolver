from z3 import *

# Initialize the Z3 solver
solver = Solver()

# Define 4-bit decision variables
bit0 = Bool('bit0')
bit1 = Bool('bit1')
bit2 = Bool('bit2')
bit3 = Bool('bit3')

solver.add(And(Not(bit3),Not(bit2),Not(bit1),bit0))


# Check if the constraints are satisfiable
if solver.check() == sat:
    model = solver.model()
    print(model)
    print("Satisfiable solution found:")

else:
    print("No satisfiable solution found.")
