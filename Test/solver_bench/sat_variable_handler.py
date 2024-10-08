class VariableMapper:
    def __init__(self, half_order, wordlength, adder_wordlength, max_adder, adder_depth):
        self.variables = self._initialize_variables(half_order, wordlength, adder_wordlength, max_adder, adder_depth)
        self.var_to_int = {var: i + 1 for i, var in enumerate(self.variables.values())}  # +1 to avoid using 0
        self.int_to_var = {i + 1: var for i, var in enumerate(self.variables.values())}
        self.max_int_value = max(self.var_to_int.values())
        
    def _initialize_variables(self, half_order, wordlength, adder_wordlength, max_adder, adder_depth):
        variables = {
            ('h', a, w): f'h_{a}_{w}' for a in range(half_order + 1) for w in range(wordlength)
        }
        variables.update({
            ('gain', g): f'gain{g}' for g in range(wordlength)
        })
        variables.update({
            ('c', i, w): f'c_{i}_{w}' for i in range(max_adder + 2) for w in range(adder_wordlength)
        })
        variables.update({
            ('l', i, w): f'l_{i}_{w}' for i in range(1, max_adder + 1) for w in range(adder_wordlength)
        })
        variables.update({
            ('r', i, w): f'r_{i}_{w}' for i in range(1, max_adder + 1) for w in range(adder_wordlength)
        })
        variables.update({
            ('alpha', i, a): f'alpha_{i}_{a}' for i in range(1, max_adder + 1) for a in range(i)
        })
        variables.update({
            ('beta', i, a): f'beta_{i}_{a}' for i in range(1, max_adder + 1) for a in range(i)
        })
        variables.update({
            ('gamma', i, k): f'gamma_{i}_{k}' for i in range(1, max_adder + 1) for k in range(adder_wordlength - 1)
        })
        variables.update({
            ('s', i, w): f's{i}{w}' for i in range(1, max_adder + 1) for w in range(adder_wordlength)
        })
        variables.update({
            ('delta', i): f'delta_{i}' for i in range(1, max_adder + 1)
        })
        variables.update({
            ('u', i, w): f'u_{i}_{w}' for i in range(1, max_adder + 1) for w in range(adder_wordlength)
        })
        variables.update({
            ('x', i, w): f'x_{i}_{w}' for i in range(1, max_adder + 1) for w in range(adder_wordlength)
        })
        variables.update({
            ('epsilon', i): f'epsilon{i}' for i in range(1, max_adder + 1)
        })
        variables.update({
            ('y', i, w): f'y_{i}_{w}' for i in range(1, max_adder + 1) for w in range(adder_wordlength)
        })
        variables.update({
            ('z', i, w): f'z_{i}_{w}' for i in range(1, max_adder + 1) for w in range(adder_wordlength)
        })
        variables.update({
            ('cout', i, w): f'cout_{i}_{w}' for i in range(1, max_adder + 1) for w in range(adder_wordlength)
        })
        variables.update({
            ('zeta', i, k): f'zeta_{i}_{k}' for i in range(1, max_adder + 1) for k in range(adder_wordlength - 1)
        })
        variables.update({
            ('theta', i, m): f'theta_{i}_{m}' for i in range(max_adder + 2) for m in range(half_order + 1)
        })
        variables.update({
            ('iota', m): f'iota_{m}' for m in range(half_order + 1)
        })
        variables.update({
            ('t', m, w): f't_{m}_{w}' for m in range(half_order + 1) for w in range(adder_wordlength)
        })
        variables.update({
            ('h_ext', m, w): f'h_ext_{m}_{w}' for m in range(half_order + 1) for w in range(adder_wordlength)
        })
        variables.update({
            ('phi', m, k): f'phi_{m}_{k}' for m in range(half_order + 1) for k in range(adder_wordlength - 1)
        })
        variables.update({
            ('psi_alpha', i, d): f'psi_alpha_{i}_{d}' for i in range(1, max_adder + 1) for d in range(adder_depth)
        })
        variables.update({
            ('psi_beta', i, d): f'psi_beta_{i}_{d}' for i in range(1, max_adder + 1) for d in range(adder_depth)
        })
        return variables
    
    def var_name_to_int(self, var_name):
        return self.var_to_int[var_name]

    def int_to_var_name(self, var_int):
        return self.int_to_var[var_int]

    def tuple_to_int(self, var_tuple):
        var_name = self.variables[var_tuple]
        return self.var_to_int[var_name]


if __name__ == "__main__":
    # Example parameters
    half_order = 5
    wordlength = 10
    adder_wordlength = wordlength + 2
    max_adder = 5
    adder_depth = 2
    
    # Create a global instance of VariableMapper
    var_mapper = VariableMapper(half_order, wordlength, adder_wordlength, max_adder, adder_depth)

    # Simplified functions to convert between variable names and integers
    def v2i(var_tuple):
        return var_mapper.tuple_to_int(var_tuple)

    def i2v(var_int):
        return var_mapper.int_to_var_name(var_int)

    # Example usage
    example_struct_var = ('h', 0, 0)
    example_struct_int = v2i(example_struct_var)
    print(f'Variable {example_struct_var} is mapped to integer {example_struct_int}')
    print(f'Integer {example_struct_int} is mapped back to variable {i2v(example_struct_int)}')

    # Accessing variable name using integer
    int_example = 4
    accessed_var_name = i2v(int_example)
    print(f'Accessed variable name for integer {int_example} is {accessed_var_name}')

    # Print the maximum integer value
    print(var_mapper.max_int_value)
