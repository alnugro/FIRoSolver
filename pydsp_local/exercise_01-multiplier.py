from circuit import Circuit
from modules import *
from io_utility import *


def main():
    # create a circuit for a truncated multiplication
    c = Circuit()
    # define data type used within the circuit => 8 bit signed numbers with 4 fractional bits
    dat = DataType.SIGNED
    num_bits = 8
    fract_bits = 4
    integer_bits = num_bits - fract_bits
    i_min = -(2**(num_bits-1))
    i_max = -i_min-1
    # create some inputs for our circuit
    i0 = c.add(Input(input_ranges=[[i_min, i_max]], data_type=dat, name="X0"))
    i1 = c.add(Input(input_ranges=[[i_min, i_max]], data_type=dat, name="X1"))
    # create multiplier -> it will produce twice the number of bits with twice the number of fractional bits
    m0 = c.add(Mult(input_ranges=[i0.output_range(), i1.output_range()], data_type=dat, name="Mult0"))
    # create the output
    o0 = c.add(Output(input_ranges=[m0.output_range()], data_type=dat, name="Y0"))
    # connect the modules 
    # => inputs -> multiplier
    c.connect(src_module=i0, dst_module=m0, dst_port=0)
    c.connect(src_module=i1, dst_module=m0, dst_port=1)
    # => multiplier -> output
    c.connect(src_module=m0, dst_module=o0, dst_port=0)
    # validate circuit (raise an exception for, e.g., open input ports, invalid connections, etc.)
    c.validate()
    # print info
    c.print_info()
    # run simulation for 10 random data samples
    num_data_samples = 10
    i0.define_input_data(get_random_values(min_val=i_min, max_val=i_max, how_many=num_data_samples))
    i1.define_input_data(get_random_values(min_val=i_min, max_val=i_max, how_many=num_data_samples))
    for time_step in range(num_data_samples):
        raw_i0_val = i0.get_output(time_step)
        i0_val = to_fixed_point(raw_i0_val, fract_bits)
        raw_i1_val = i1.get_output(time_step)
        i1_val = to_fixed_point(raw_i1_val, fract_bits)
        raw_o0_val = o0.get_output(time_step)
        o0_val = to_fixed_point(raw_o0_val, 2*fract_bits)  # note that o0 has twice the number of fractional bits due to the multiplication
        print(f"t={time_step} => i0={i0_val} i1={i1_val} o0={o0_val}")
        if i0_val * i1_val != o0_val:
            raise Exception(f"Multiplication error in time step t={time_step}")


if __name__ == '__main__':
    main()
