# compare_formulations.py
import time
from z3 import *

class BitShiftBitLevel:
    def __init__(self, out, wordlength, N=4, verbose=False):
        """
        Bit-Level Formulation using individual Boolean variables.

        Parameters:
        - out (tuple): Output tuple (not used in current constraints).
        - wordlength (int): Number of bits in each word.
        - N (int): Number of XOR gates.
        - verbose (bool): If True, prints the model.
        """
        self.wordlength = wordlength
        self.out = out
        self.N = N  # Number of XOR gates
        self.verbose = verbose

        self.solver = Solver()

        # Define Boolean variables
        self.w = [[Bool(f'w{i}{w}') for w in range(self.wordlength)] for i in range(1, self.N + 1)]
        self.epsilon = [Bool(f'epsilon{i}') for i in range(1, self.N + 1)]
        self.y = [[Bool(f'y{i}{w}') for w in range(self.wordlength)] for i in range(1, self.N + 1)]

        # Add constraints to set exactly half of the bits to True
        half_bits = self.wordlength // 2
        for i in range(self.N):
            # Count the number of True bits in self.w[i]
            bits = self.w[i]
            # Create a list of If expressions: 1 if bit is True, else 0
            bit_values = [If(bit, 1, 0) for bit in bits]
            # Add the constraint that the sum equals half_bits
            self.solver.add(Sum(bit_values) == half_bits)

            # Similarly, set epsilon[i] to True
            self.solver.add(self.epsilon[i] == True)

        # XOR constraints: y = w XOR epsilon
        for i in range(self.N):
            for w in range(self.wordlength):
                # y = w XOR epsilon is equivalent to:
                # (y ∨ ¬w ∨ ¬epsilon) ∧ (¬y ∨ w ∨ ¬epsilon) ∧ (¬y ∨ ¬w ∨ epsilon) ∧ (y ∨ w ∨ epsilon)
                self.solver.add(Or(Not(self.w[i][w]), Not(self.epsilon[i]), Not(self.y[i][w])))
                self.solver.add(Or(self.w[i][w], Not(self.epsilon[i]), Not(self.y[i][w])))
                self.solver.add(Or(Not(self.w[i][w]), self.epsilon[i], Not(self.y[i][w])))
                self.solver.add(Or(self.w[i][w], self.epsilon[i], Not(self.y[i][w])))

    def run(self):
        """
        Runs the solver and measures the runtime.

        Returns:
        - sat_status (str): "SAT" if satisfiable, else "No solution".
        - elapsed (float): Time taken to solve in seconds.
        """
        start_time = time.perf_counter()
        result = self.solver.check()
        end_time = time.perf_counter()
        elapsed = end_time - start_time

        if result == sat:
            sat_status = "SAT"
            if self.verbose:
                model = self.solver.model()
                self.print_model(model)
        else:
            sat_status = "No solution"

        return sat_status, elapsed

    def print_model(self, model):
        """
        Prints the model in a readable format.

        Parameters:
        - model (z3.ModelRef): The model returned by the solver.
        """
        print("Bit-Level Model:")
        for i in range(len(self.w)):
            for w in range(self.wordlength):
                print(f'w[{i + 1}][{w}] = {model[self.w[i][w]]}')
        for i in range(len(self.epsilon)):
            print(f'epsilon[{i + 1}] = {model[self.epsilon[i]]}')
        for i in range(len(self.y)):
            for w in range(self.wordlength):
                print(f'y[{i + 1}][{w}] = {model[self.y[i][w]]}')
        print("\n")


class BitShiftBitVector:
    def __init__(self, out, wordlength, N=4, verbose=False):
        """
        Bit-Vector Formulation using Z3's BitVec variables.

        Parameters:
        - out (tuple): Output tuple (not used in current constraints).
        - wordlength (int): Number of bits in each word.
        - N (int): Number of XOR gates.
        - verbose (bool): If True, prints the model.
        """
        self.wordlength = wordlength
        self.out = out
        self.N = N  # Number of XOR gates
        self.verbose = verbose

        self.solver = Solver()

        # Define BitVec variables
        self.w = [BitVec(f'w{i}', self.wordlength) for i in range(1, self.N + 1)]
        self.epsilon = [Bool(f'epsilon{i}') for i in range(1, self.N + 1)]
        self.y = [BitVec(f'y{i}', self.wordlength) for i in range(1, self.N + 1)]

        # Add constraints to set exactly half of the bits to 1
        half_bits = self.wordlength // 2
        for i in range(self.N):
            # Extract each bit and sum them
            bits = [Extract(k, k, self.w[i]) for k in range(self.wordlength)]
            bit_values = [If(bits[k] == 1, 1, 0) for k in range(self.wordlength)]
            # Add the constraint that the sum equals half_bits
            self.solver.add(Sum(bit_values) == half_bits)

            # Similarly, set epsilon[i] to True
            self.solver.add(self.epsilon[i] == True)

        # XOR constraints: y = w XOR epsilon
        for i in range(self.N):
            # Convert epsilon (Bool) to BitVec
            epsilon_bitvec = If(self.epsilon[i], BitVecVal(1, self.wordlength), BitVecVal(0, self.wordlength))
            self.solver.add(self.y[i] == (self.w[i] ^ epsilon_bitvec))

    def run(self):
        """
        Runs the solver and measures the runtime.

        Returns:
        - sat_status (str): "SAT" if satisfiable, else "No solution".
        - elapsed (float): Time taken to solve in seconds.
        """
        start_time = time.perf_counter()
        result = self.solver.check()
        end_time = time.perf_counter()
        elapsed = end_time - start_time

        if result == sat:
            sat_status = "SAT"
            if self.verbose:
                model = self.solver.model()
                self.print_model(model)
        else:
            sat_status = "No solution"

        return sat_status, elapsed

    def print_model(self, model):
        """
        Prints the model in a readable format.

        Parameters:
        - model (z3.ModelRef): The model returned by the solver.
        """
        print("Bit-Vector Model:")
        for i in range(len(self.w)):
            w_val = model.evaluate(self.w[i], model_completion=True).as_long()
            # Convert to binary string with leading zeros
            w_bin = bin(w_val)[2:].zfill(self.wordlength)
            print(f'w[{i + 1}] = {w_bin}')
        for i in range(len(self.epsilon)):
            epsilon_val = model.evaluate(self.epsilon[i], model_completion=True)
            print(f'epsilon[{i + 1}] = {epsilon_val}')
        for i in range(len(self.y)):
            y_val = model.evaluate(self.y[i], model_completion=True).as_long()
            y_bin = bin(y_val)[2:].zfill(self.wordlength)
            print(f'y[{i + 1}] = {y_bin}')
        print("\n")


def main():
    """
    Main function to compare Bit-Level and Bit-Vector formulations with increasing wordlength and N.
    """
    # Starting values
    wordlength = 5  # Start at 5 to satisfy constraints accessing bits 2, 3, 4
    N = 1
    out = (25, 23, 11, 25, 75)  # Not used in current constraints
    verbose = False  # Set to True if you want to see models (may slow down the process)

    # Print header with semicolon separators
    header = "WordLength;N;Bit-Level Status;Bit-Level Time (s);Bit-Vector Status;Bit-Vector Time (s);Faster Formulation"
    print(header)

    try:
        while True:
            # Bit-Level Formulation
            bitshift_bit_level = BitShiftBitLevel(out, wordlength, N=N, verbose=verbose)
            status_bit_level, time_bit_level = bitshift_bit_level.run()

            # Bit-Vector Formulation
            bitshift_bit_vector = BitShiftBitVector(out, wordlength, N=N, verbose=verbose)
            status_bit_vector, time_bit_vector = bitshift_bit_vector.run()

            # Determine which formulation is faster
            if time_bit_level < time_bit_vector:
                faster = f"Bit-Level by {time_bit_vector - time_bit_level:.6f}s"
            else:
                faster = f"Bit-Vector by {time_bit_level - time_bit_vector:.6f}s"

            # Print the results with semicolon separator
            print(f"{wordlength};{N};{status_bit_level};{time_bit_level:.6f};{status_bit_vector};{time_bit_vector:.6f};{faster}")

            # Increment wordlength and N
            wordlength += 1
            N += 1

    except KeyboardInterrupt:
        print("\nInfinite testing stopped by user.")


if __name__ == '__main__':
    main()
