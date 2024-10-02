from enum import Enum
import math

class DataType(Enum):
    UNSIGNED = 0
    SIGNED = 1


##########################
# list of useful modules:
##########################
# i/o
##########################
# + input
# + output
# + constant
##########################
# arithmetic
##########################
# + add
# + sub
# + mult
# + comparator
##########################
# data path control
##########################
# + register
# + multiplexer
##########################
# number format
##########################
# + shift left/right
# + truncate msb
##########################


class Module:
    def __init__(self, name) -> None:
        self.inputs = []
        self.output = None
        self.input_ranges = []
        self.data_type = None
        self.name = name
        self.input_modules = {}
    
    def get_output(self, time_step):
        val = self.compute_output(time_step)
        o_range = self.output_range()
        if val < o_range[0] or val > o_range[1]:
            raise Exception(f"{self.name}.compute_output failed: value '{val}' out of bounds, allowed range {o_range[0]} <= value <= {o_range[1]}")
        return val
    
    def compute_output(self, time_step):
        raise Exception("Never call Module.compute_output -> only call overloaded versions by derived classes")
    
    def output_word_size(self):
        o_range = self.output_range()
        o_min = o_range[0]
        o_max = o_range[1]
        if self.data_type is DataType.UNSIGNED:
            return int(math.ceil(math.log2(o_max+1)))
        else:
            val_min = -o_min if o_min < 0 else o_min+1
            val_max = -o_max if o_max < 0 else o_max+1
            w_min_val = int(math.ceil(math.log2(val_min)))
            w_max_val = int(math.ceil(math.log2(val_max)))
            return max(w_min_val, w_max_val) + 1

    def output_range(self):
        raise Exception("Never call Module.output_range -> only call overloaded versions by derived classes")
    
    def is_valid(self):
        return False, "Please don't instantiate the base class!"
    

class Input(Module):
    def __init__(self, input_ranges, data_type, name) -> None:
        super().__init__(name)
        if len(input_ranges) != 1:
            raise Exception("Please provide input ranges for Input modules as 1-dimensional lists, e.g., '[[0, 255]]' for an unsigned 8-bit input")
        self.inputs = [name]
        self.input_ranges = input_ranges
        self.data_type = data_type
        self.input_data = [0]

    def output_range(self):
        return self.input_ranges[0]
    
    def define_input_data(self, input_data):
        self.input_data = input_data
    
    def compute_output(self, time_step):
        idx = time_step % len(self.input_data)
        return self.input_data[idx]
    
    def is_valid(self):
        valid = len(self.input_modules) == 0
        reason = "" if valid else "Input modules cannot have inputs!"
        return valid, reason


class Output(Module):
    def __init__(self, input_ranges, data_type, name) -> None:
        super().__init__(name)
        if len(input_ranges) != 1:
            raise Exception("Please provide input ranges for Output modules as 1-dimensional lists, e.g., '[[0, 255]]' for an unsigned 8-bit output")
        self.output = name
        self.input_ranges = input_ranges
        self.data_type = data_type

    def output_range(self):
        return self.input_ranges[0]

    def compute_output(self, time_step):
        return self.input_modules[0].get_output(time_step)
    
    def is_valid(self):
        valid = len(self.input_modules) == 1 and 0 in self.input_modules
        reason = "" if valid else "Output modules must have exactly 1 input!"
        return valid, reason
    

class Constant(Module):
    def __init__(self, value, data_type, name) -> None:
        super().__init__(name)
        self.output = name
        self.input_ranges = [[value, value]]
        self.data_type = data_type
        self.value = value

    def output_range(self):
        return self.input_ranges[0]
    
    def compute_output(self, time_step):
        return self.value
    
    def is_valid(self):
        valid = len(self.input_modules) == 0
        reason = "" if valid else "Constants cannot have inputs!"
        return valid, reason


class Register(Module):
    def __init__(self, input_ranges, data_type, name) -> None:
        super().__init__(name)
        if len(input_ranges) != 1:
            raise Exception("Please provide input ranges for Register modules as 1-dimensional lists, e.g., '[[0, 255]]' for an unsigned 8-bit output")
        self.inputs = ["X"]
        self.output = "Y"
        self.input_ranges = input_ranges
        self.data_type = data_type
        self.value_map = {0: 0}  # t -> value at time t
    
    def reset(self, reset_value):
        self.value_map = {0: reset_value}

    def output_range(self):
        return self.input_ranges[0]
    
    def compute_output(self, time_step):
        if time_step not in self.value_map:
            self.value_map[time_step] = self.input_modules[0].get_output(time_step-1)
        return self.value_map[time_step]
    
    def is_valid(self):
        valid = len(self.input_modules) == 1 and 0 in self.input_modules
        reason = "" if valid else "Registers must have exactly 1 input!"
        return valid, reason


class Add(Module):
    def __init__(self, input_ranges, data_type, name) -> None:
        super().__init__(name)
        if len(input_ranges) != 2:
            raise Exception("Please provide input ranges for Adder modules as 2-dimensional lists, e.g., '[[0, 255], [0, 255]]' for unsigned 8-bit inputs")
        self.inputs = ["X0", "X1"]
        self.output = "Y"
        self.input_ranges = input_ranges
        if type(data_type) is not DataType:
            raise Exception("Invalid data type provided; it needs to be of type 'DataType'")
        self.data_type = data_type
        if self.data_type == DataType.UNSIGNED:
            if any(r[0] < 0 for r in self.input_ranges):
                raise Exception("Data type (UNSIGNED) incompatible with provided input ranges")
            
    def output_range(self):
        input_range_0 = self.input_ranges[0]
        input_range_1 = self.input_ranges[1]
        min_val = input_range_0[0] + input_range_1[0]
        max_val = input_range_0[1] + input_range_1[1]
        return [min_val, max_val]

    def compute_output(self, time_step):
        x_0 = self.input_modules[0].get_output(time_step)
        x_1 = self.input_modules[1].get_output(time_step)
        return x_0 + x_1
    
    def is_valid(self):
        valid = len(self.input_modules) == 2 and 0 in self.input_modules and 1 in self.input_modules
        reason = "" if valid else "Adders must have exactly 2 inputs!"
        return valid, reason


class Sub(Module):
    def __init__(self, input_ranges, data_type, name) -> None:
        super().__init__(name)
        if len(input_ranges) != 2:
            raise Exception("Please provide input ranges for Subtract modules as 2-dimensional lists, e.g., '[[0, 255], [0, 255]]' for unsigned 8-bit inputs")
        self.inputs = ["X0", "X1"]
        self.output = "Y"
        self.input_ranges = input_ranges
        if type(data_type) is not DataType:
            raise Exception("Invalid data type provided; it needs to be of type 'DataType'")
        self.data_type = data_type
        if self.data_type == DataType.UNSIGNED:
            if any(r[0] < 0 for r in self.input_ranges):
                raise Exception("Data type (UNSIGNED) incompatible with provided input ranges")
            
    def output_range(self):
        input_range_0 = self.input_ranges[0]
        input_range_1 = self.input_ranges[1]
        min_val = input_range_0[0] - input_range_1[1]
        max_val = input_range_0[1] - input_range_1[0]
        return [min_val, max_val]

    def compute_output(self, time_step):
        x_0 = self.input_modules[0].get_output(time_step)
        x_1 = self.input_modules[1].get_output(time_step)
        return x_0 - x_1
    
    def is_valid(self):
        reason = ""
        inputs_valid = len(self.input_modules) == 2 and 0 in self.input_modules and 1 in self.input_modules
        if not inputs_valid:
            reason = "Subtracters must have exactly 2 inputs!"
        data_type_valid = self.data_type == DataType.SIGNED or self.output_range[0] >= 0
        if not inputs_valid:
            reason = "Subtracters must have signed data type if they can produce negative values"
        valid = inputs_valid and data_type_valid
        return valid, reason

class Mult(Module):
    def __init__(self, input_ranges, data_type, name) -> None:
        super().__init__(name)
        if len(input_ranges) != 2:
            raise Exception("Please provide input ranges for Multiplier modules as 2-dimensional lists, e.g., '[[0, 255], [0, 255]]' for unsigned 8-bit inputs")
        self.inputs = ["X0", "X1"]
        self.output = "Y"
        self.input_ranges = input_ranges
        if type(data_type) is not DataType:
            raise Exception("Invalid data type provided; it needs to be of type 'DataType'")
        self.data_type = data_type
        if self.data_type == DataType.UNSIGNED:
            if any(r[0] < 0 for r in self.input_ranges):
                raise Exception("Data type (UNSIGNED) incompatible with provided input ranges")

    def compute_output(self, time_step):
        x_0 = self.input_modules[0].get_output(time_step)
        x_1 = self.input_modules[1].get_output(time_step)
        return x_0 * x_1
            
    def output_range(self):
        input_range_0 = self.input_ranges[0]
        input_range_1 = self.input_ranges[1]
        mul_a = input_range_0[0] * input_range_1[0]
        mul_b = input_range_0[0] * input_range_1[1]
        mul_c = input_range_0[1] * input_range_1[0]
        mul_d = input_range_0[1] * input_range_1[1]
        min_val = min(mul_a, mul_b, mul_c, mul_d)
        max_val = max(mul_a, mul_b, mul_c, mul_d)
        return [min_val, max_val]
    
    def is_valid(self):
        valid = len(self.input_modules) == 2 and 0 in self.input_modules and 1 in self.input_modules
        reason = "" if valid else "Multipliers must have exactly 2 inputs!"
        return valid, reason


class ComparatorType(Enum):
    LE  = 0  # <
    LEQ = 1  # <=
    EQ  = 2  # =
    GEQ = 3  # >=
    GE  = 4  # >


class Comparator(Module):
    def __init__(self, input_ranges, name, comparator_type) -> None:
        super().__init__(name)
        if len(input_ranges) != 2:
            raise Exception("Please provide input ranges for Comparator modules as 2-dimensional lists, e.g., '[[0, 255], [0, 255]]' for unsigned 8-bit inputs")
        self.inputs = ["X0", "X1"]
        self.output = "Y"
        self.input_ranges = input_ranges
        self.data_type = DataType.UNSIGNED  # output is always an unsigned 1-bit number
        if type(comparator_type) is not ComparatorType:
            raise Exception("Invalid comparator type provided; it needs to be of type 'ComparatorType'")
        self.comparator_type = comparator_type
            
    def output_range(self):
        return [0, 1]

    def compute_output(self, time_step):
        x_0 = self.input_modules[0].get_output(time_step)
        x_1 = self.input_modules[1].get_output(time_step)
        result_dict = {
            ComparatorType.LE: x_0 < x_1,
            ComparatorType.LEQ: x_0 <= x_1,
            ComparatorType.EQ: x_0 == x_1,
            ComparatorType.GEQ: x_0 >= x_1,
            ComparatorType.GE: x_0 > x_1
        }
        return 1 if result_dict[self.comparator_type] else 0
    
    def is_valid(self):
        valid = len(self.input_modules) == 2 and 0 in self.input_modules and 1 in self.input_modules
        reason = "" if valid else "Comparators must have exactly 2 inputs!"
        return valid, reason


class Mux(Module):
    def __init__(self, input_ranges, data_type, name) -> None:
        super().__init__(name)
        num_data_inputs = len(input_ranges)-1
        self.inputs = [f"X{i}" for i in range(num_data_inputs)] + ["S"]
        self.output = "Y"
        self.input_ranges = input_ranges
        if type(data_type) is not DataType:
            raise Exception("Invalid data type provided; it needs to be of type 'DataType'")
        self.data_type = data_type
        if self.data_type == DataType.UNSIGNED:
            if any(r[0] < 0 for r in self.input_ranges):
                raise Exception("Data type (UNSIGNED) incompatible with provided input ranges")
        self.num_data_inputs = num_data_inputs
            
    def output_range(self):
        min_val = min(x[0] for x in self.input_ranges)
        max_val = max(x[1] for x in self.input_ranges)
        return [min_val, max_val]

    def compute_output(self, time_step):
        input_data = {i: self.input_modules[i].get_output(time_step) for i in range(self.num_data_inputs)}
        select_value = self.input_modules[self.num_data_inputs].get_output(time_step)
        return input_data[select_value]
    
    def is_valid(self):
        reason = ""
        correct_num_inputs = len(self.input_modules) == self.num_data_inputs+1
        if not correct_num_inputs:
            reason = "Invalid number of inputs"
        correct_input_ports = all(x in self.input_modules for x in range(self.num_data_inputs+1))
        if not correct_input_ports:
            reason = "Inputs are connected to invalid input ports"
        correct_select_range = all(x in self.input_ranges[self.num_data_inputs] for x in range(self.num_data_inputs))
        if not correct_select_range:
            reason = "Invalid select range for the given input count"
        correct_select_data_type = self.input_modules[self.num_data_inputs].data_type == DataType.UNSIGNED
        if not correct_select_data_type:
            reason = "Select data type must be UNSIGNED"
        valid = correct_num_inputs and correct_input_ports and correct_select_range and correct_select_data_type
        return valid, reason


class ShiftLeft(Module):
    def __init__(self, input_ranges, data_type, name, shift_length) -> None:
        super().__init__(name)
        if len(input_ranges) != 1:
            raise Exception("Please provide input ranges for Shift modules as 1-dimensional lists, e.g., '[[0, 255]]' for an unsigned 8-bit input")
        if shift_length < 0:
            raise Exception("Shift length must be larger than or equal to 0")
        self.inputs = ["X"]
        self.output = "Y"
        self.input_ranges = input_ranges
        self.data_type = data_type
        self.shift_length = shift_length

    def output_range(self):
        scaling_factor = 2**self.shift_length
        return [scaling_factor * self.input_ranges[0][0], scaling_factor * self.input_ranges[0][1]]
    
    def compute_output(self, time_step):
        return self.input_modules[0].get_output(time_step) * (2**self.shift_length)
    
    def is_valid(self):
        valid = len(self.input_modules) == 1 and 0 in self.input_modules
        reason = "" if valid else "Shift modules must have exactly 1 input!"
        return valid, reason


class ShiftRight(Module):
    def __init__(self, input_ranges, data_type, name, shift_length) -> None:
        super().__init__(name)
        if len(input_ranges) != 1:
            raise Exception("Please provide input ranges for Shift modules as 1-dimensional lists, e.g., '[[0, 255]]' for an unsigned 8-bit input")
        if shift_length < 0:
            raise Exception("Shift length must be larger than or equal to 0")
        self.inputs = ["X"]
        self.output = "Y"
        self.input_ranges = input_ranges
        self.data_type = data_type
        self.shift_length = shift_length

    def output_range(self):
        scaling_factor = 1 / (2**self.shift_length)
        return [int(math.floor(scaling_factor * self.input_ranges[0][0])), int(math.floor(scaling_factor * self.input_ranges[0][1]))]
    
    def compute_output(self, time_step):
        return int(math.floor(self.input_modules[0].get_output(time_step) / (2**self.shift_length)))
    
    def is_valid(self):
        valid = len(self.input_modules) == 1 and 0 in self.input_modules
        reason = "" if valid else "Shift modules must have exactly 1 input!"
        return valid, reason


class TruncateMSBs(Module):
    def __init__(self, input_ranges, data_type, name, how_many) -> None:
        super().__init__(name)
        if len(input_ranges) != 1:
            raise Exception("Please provide input ranges for MSB truncation modules as 1-dimensional lists, e.g., '[[0, 255]]' for an unsigned 8-bit input")
        if how_many <= 0:
            raise Exception("Number of bits must be larger than 0")
        self.inputs = ["X"]
        self.output = "Y"
        self.input_ranges = input_ranges
        self.data_type = data_type
        self.how_many = how_many

    def output_range(self):
        input_range = self.input_ranges[0]
        i_min = input_range[0]
        i_max = input_range[1]
        if self.data_type is DataType.UNSIGNED:
            num_bits_in = int(math.ceil(math.log2(i_max+1)))
            min_val = 0
            max_val = 2**(num_bits_in-self.how_many) - 1
            max_val = max(0, max_val)
        else:
            val_min = -i_min if i_min < 0 else i_min+1
            val_max = -i_max if i_max < 0 else i_max+1
            w_min_val = int(math.ceil(math.log2(val_min)))
            w_max_val = int(math.ceil(math.log2(val_max)))
            num_bits_in = max(w_min_val, w_max_val) + 1
            min_val = -2**(num_bits_in-1-self.how_many)
            max_val = -min_val-1
        return [min_val, max_val]
    
    def compute_output(self, time_step):
        input_module = self.input_modules[0]
        input_value = input_module.get_output(time_step)
        input_bit_width = input_module.output_word_size()
        if input_value >= 0 or self.data_type == DataType.UNSIGNED:
            input_value_str = bin(input_value)[2:].zfill(input_bit_width)
        else:
            input_value_str = bin(2**input_bit_width + input_value)[2:]
        output_value_str = input_value_str[self.how_many:]
        output_bit_width = self.output_word_size()
        output_value = 0
        for idx, digit in enumerate(output_value_str):
            weighted_digit = 2**(output_bit_width-1-idx) * int(digit)
            if idx == 0 and self.data_type == DataType.SIGNED:
                output_value -= weighted_digit
            else:
                output_value += weighted_digit
        return output_value
    
    def is_valid(self):
        reason = ""
        correct_num_inputs = len(self.input_modules) == 1 and 0 in self.input_modules
        if not correct_num_inputs:
            reason = "MSB truncation modules must have exactly 1 input!"
        not_too_many_bits_cut_off = self.input_modules[0].output_word_size() > self.how_many
        if not not_too_many_bits_cut_off:
            reason = "Cut off too many bits (double-check data ranges please)"
        valid = correct_num_inputs and not_too_many_bits_cut_off
        return valid, reason