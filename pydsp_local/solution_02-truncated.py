from circuit import Circuit
from modules import *
from io_utility import *
from file_writers import *


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
    q0 = c.add(ShiftRight(input_ranges=[m0.output_range()], data_type=dat, name="Q0", shift_length=fract_bits))
    q1 = c.add(TruncateMSBs(input_ranges=[q0.output_range()], data_type=dat, name="Q2", how_many=integer_bits))
    # create the output
    o0 = c.add(Output(input_ranges=[q1.output_range()], data_type=dat, name="Y0"))
    # connect the modules 
    # => inputs -> multiplier
    c.connect(src_module=i0, dst_module=m0, dst_port=0)
    c.connect(src_module=i1, dst_module=m0, dst_port=1)
    # => quantization
    c.connect(src_module=m0, dst_module=q0, dst_port=0)
    c.connect(src_module=q0, dst_module=q1, dst_port=0)
    # => multiplier -> output
    c.connect(src_module=q1, dst_module=o0, dst_port=0)
    # validate circuit (raise an exception for, e.g., open input ports, invalid connections, etc.)
    c.validate()
    # print info
    c.print_info()
    # export dot graph
    GraphvizWriter.write(file_path="truncated_mult.dot", obj=c)
    # run simulation for some hand-crafted data samples
    num_data_samples = 9
    i0.define_input_data(from_fixed_point([0.50,  1.50, 2.50, -8.00, 3.250,  2.000, 1.125, -0.125,  3.125], fractional_bits=fract_bits))
    i1.define_input_data(from_fixed_point([3.75, -1.25, 1.25,  7.75, 5.125, -7.875, 3.875,  0.125, -0.250], fractional_bits=fract_bits))
    for time_step in range(num_data_samples):
        raw_i0_val = i0.get_output(time_step)
        i0_val = to_fixed_point(raw_i0_val, fract_bits)
        raw_i1_val = i1.get_output(time_step)
        i1_val = to_fixed_point(raw_i1_val, fract_bits)
        raw_m0_val = m0.get_output(time_step)
        m0_val = to_fixed_point(raw_m0_val, 2*fract_bits)
        raw_o0_val = o0.get_output(time_step)
        o0_val = to_fixed_point(raw_o0_val, fract_bits)
        print(f"t={time_step} => i0={i0_val} i1={i1_val} m0={m0_val} o0={o0_val}")


if __name__ == '__main__':
    main()
