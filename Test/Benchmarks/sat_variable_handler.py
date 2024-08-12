class VariableMapper:
    def __init__(self, half_order, wordlength, adder_count):
        self.variables = self._initialize_variables(half_order, wordlength, adder_count)
        self.var_to_int = {var: i + 1 for i, var in enumerate(self.variables.values())}  # +1 to avoid using 0
        self.int_to_var = {i + 1: var for i, var in enumerate(self.variables.values())}
        self.max_int_value = max(self.var_to_int.values())
        
    def _initialize_variables(self, half_order, wordlength, adder_count):
        variables = {
            ('h', a, w): f'h{a}_{w}' for a in range(half_order + 1) for w in range(wordlength)
        }
        variables.update({
            ('gain', g): f'gain{g}' for g in range(wordlength)
        })
        variables.update({
            ('c', i, w): f'c{i}{w}' for i in range(adder_count + 1) for w in range(wordlength)
        })
        variables.update({
            ('l', i, w): f'l{i}{w}' for i in range(1, adder_count + 1) for w in range(wordlength)
        })
        variables.update({
            ('r', i, w): f'r{i}{w}' for i in range(1, adder_count + 1) for w in range(wordlength)
        })
        variables.update({
            ('alpha', i, a): f'alpha{i}{a}' for i in range(1, adder_count + 1) for a in range(i)
        })
        variables.update({
            ('Beta', i, a): f'Beta{i}{a}' for i in range(1, adder_count + 1) for a in range(i)
        })
        variables.update({
            ('gamma', i, k): f'gamma{i}{k}' for i in range(1, adder_count + 1) for k in range(wordlength - 1)
        })
        variables.update({
            ('s', i, w): f's{i}{w}' for i in range(1, adder_count + 1) for w in range(wordlength)
        })
        variables.update({
            ('delta', i): f'delta{i}' for i in range(1, adder_count + 1)
        })
        variables.update({
            ('u', i, w): f'u{i}{w}' for i in range(1, adder_count + 1) for w in range(wordlength)
        })
        variables.update({
            ('x', i, w): f'x{i}{w}' for i in range(1, adder_count + 1) for w in range(wordlength)
        })
        variables.update({
            ('epsilon', i): f'epsilon{i}' for i in range(1, adder_count + 1)
        })
        variables.update({
            ('y', i, w): f'y{i}{w}' for i in range(1, adder_count + 1) for w in range(wordlength)
        })
        variables.update({
            ('z', i, w): f'z{i}{w}' for i in range(1, adder_count + 1) for w in range(wordlength)
        })
        variables.update({
            ('cout', i, w): f'cout{i}{w}' for i in range(1, adder_count + 1) for w in range(wordlength)
        })
        variables.update({
            ('zeta', i, k): f'zeta{i}{k}' for i in range(1, adder_count + 1) for k in range(wordlength - 1)
        })
        variables.update({
            ('h0', m): f'h0{m}' for m in range(half_order + 1)
        })
        variables.update({
            ('t', i, m): f't{i}_{m}' for i in range(1, adder_count + 1) for m in range(half_order + 1)
        })
        variables.update({
            ('e', m): f'e{m}' for m in range(half_order + 1)
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
    # Create a global instance of VariableMapper
    half_order = 3
    wordlength = 4
    wordlength = 6
    adder_count = 5
    var_mapper = VariableMapper(half_order, wordlength, wordlength, adder_count)

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

    example_struct_var = ('h', 0, 0)
    example_struct_int = v2i(('h', 0, 0))
