from circuit import Circuit
from modules import *
from io_utility import *


def main():
    # create a circuit for a truncated multiplication
    c = Circuit()
    # define data type used within the circuit => 8 bit signed numbers with 4 fractional bits
    dat = DataType.SIGNED
    num_bits = 16
    fract_bits = 12
    i_min = -(2**(num_bits-1))
    i_max = -i_min-1
    # create an input for our circuit
    i0 = c.add(Input(input_ranges=[[i_min, i_max]], data_type=dat, name="X0"))
    # create a register => note that we have to manually set the input word size since we will use it within a feed-back loop!
    r0 = c.add(Register(input_ranges=[[i_min, i_max]], data_type=dat, name="R0"))
    # create a constant value
    c0 = c.add(Constant(value=3, data_type=dat, name="C0"))
    # create the data path
    u0 = c.add(Sub(input_ranges=[i0.output_range(), r0.output_range()], data_type=dat, name="Sub0"))
    m0 = c.add(Mult(input_ranges=[u0.output_range(), c0.output_range()], data_type=dat, name="Mult0"))
    s0 = c.add(ShiftRight(input_ranges=[m0.output_range()], shift_length=3, data_type=dat, name="Shift0"))
    # create the output
    o0 = c.add(Output(input_ranges=[u0.output_range()], data_type=dat, name="Y0"))
    # connect the modules 
    c.connect(src_module=i0, dst_module=u0, dst_port=0)
    c.connect(src_module=r0, dst_module=u0, dst_port=1)  # this port gets subtracted
    c.connect(src_module=u0, dst_module=m0, dst_port=0)
    c.connect(src_module=c0, dst_module=m0, dst_port=1)
    c.connect(src_module=m0, dst_module=s0, dst_port=0)
    c.connect(src_module=s0, dst_module=r0, dst_port=0)
    c.connect(src_module=u0, dst_module=o0, dst_port=0)
    # validate circuit (raise an exception for, e.g., open input ports, invalid connections, etc.)
    c.validate()
    # print info
    c.print_info()
    # run simulation for some random data samples
    num_data_samples = 10
    i0.define_input_data(get_random_values(min_val=i_min, max_val=i_max, how_many=num_data_samples))
    for time_step in range(num_data_samples):
        raw_i0_val = i0.get_output(time_step)
        i0_val = to_fixed_point(raw_i0_val, fract_bits)
        raw_r0_val = r0.get_output(time_step)
        r0_val = to_fixed_point(raw_r0_val, fract_bits)
        raw_o0_val = o0.get_output(time_step)
        o0_val = to_fixed_point(raw_o0_val, fract_bits)
        print(f"t={time_step} => input={i0_val} register={r0_val} output={o0_val}")


if __name__ == '__main__':
    main()
