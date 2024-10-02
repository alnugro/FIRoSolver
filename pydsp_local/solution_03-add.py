from circuit import Circuit
from modules import *
from io_utility import *
from file_writers import *


def main():
    # create a circuit for an adder tree
    c = Circuit()
    # define data type used within the circuit
    dat = DataType.SIGNED
    num_bits = 8
    fract_bits = 0
    integer_bits = num_bits - fract_bits
    i_min = -(2**(num_bits-1))
    i_max = -i_min-1
    # create some inputs for our circuit
    i0 = c.add(Input(input_ranges=[[i_min, i_max]], data_type=dat, name="X0"))
    i1 = c.add(Input(input_ranges=[[i_min, i_max]], data_type=dat, name="X1"))
    i2 = c.add(Input(input_ranges=[[i_min, i_max]], data_type=dat, name="X2"))
    i3 = c.add(Input(input_ranges=[[i_min, i_max]], data_type=dat, name="X3"))
    i4 = c.add(Input(input_ranges=[[i_min, i_max]], data_type=dat, name="X4"))
    i5 = c.add(Input(input_ranges=[[i_min, i_max]], data_type=dat, name="X5"))
    i6 = c.add(Input(input_ranges=[[i_min, i_max]], data_type=dat, name="X6"))
    i7 = c.add(Input(input_ranges=[[i_min, i_max]], data_type=dat, name="X7"))
    # create adders
    a0 = c.add(Add(input_ranges=[i0.output_range(), i1.output_range()], data_type=dat, name="Add0"))
    a1 = c.add(Add(input_ranges=[i2.output_range(), i3.output_range()], data_type=dat, name="Add1"))
    a2 = c.add(Add(input_ranges=[i4.output_range(), i5.output_range()], data_type=dat, name="Add2"))
    a3 = c.add(Add(input_ranges=[i6.output_range(), i7.output_range()], data_type=dat, name="Add3"))
    a4 = c.add(Add(input_ranges=[a0.output_range(), a1.output_range()], data_type=dat, name="Add4"))
    a5 = c.add(Add(input_ranges=[a2.output_range(), a3.output_range()], data_type=dat, name="Add5"))
    a6 = c.add(Add(input_ranges=[a4.output_range(), a5.output_range()], data_type=dat, name="Add6"))
    # create the output
    o0 = c.add(Output(input_ranges=[a6.output_range()], data_type=dat, name="Y"))
    # connect the modules 
    c.connect(src_module=i0, dst_module=a0, dst_port=0)
    c.connect(src_module=i1, dst_module=a0, dst_port=1)
    c.connect(src_module=i2, dst_module=a1, dst_port=0)
    c.connect(src_module=i3, dst_module=a1, dst_port=1)
    c.connect(src_module=i4, dst_module=a2, dst_port=0)
    c.connect(src_module=i5, dst_module=a2, dst_port=1)
    c.connect(src_module=i6, dst_module=a3, dst_port=0)
    c.connect(src_module=i7, dst_module=a3, dst_port=1)
    c.connect(src_module=a0, dst_module=a4, dst_port=0)
    c.connect(src_module=a1, dst_module=a4, dst_port=1)
    c.connect(src_module=a2, dst_module=a5, dst_port=0)
    c.connect(src_module=a3, dst_module=a5, dst_port=1)
    c.connect(src_module=a4, dst_module=a6, dst_port=0)
    c.connect(src_module=a5, dst_module=a6, dst_port=1)
    c.connect(src_module=a6, dst_module=o0, dst_port=0)
    # validate circuit (raise an exception for, e.g., open input ports, invalid connections, etc.)
    c.validate()
    # print info
    c.print_info()
    # export dot graph
    GraphvizWriter.write(file_path="adder_tree.dot", obj=c)
    # run simulation for some random data and check against python's "+" operator
    num_data_samples = 100_000
    i0.define_input_data(get_random_values(min_val=i_min, max_val=i_max, how_many=num_data_samples))
    i1.define_input_data(get_random_values(min_val=i_min, max_val=i_max, how_many=num_data_samples))
    i2.define_input_data(get_random_values(min_val=i_min, max_val=i_max, how_many=num_data_samples))
    i3.define_input_data(get_random_values(min_val=i_min, max_val=i_max, how_many=num_data_samples))
    i4.define_input_data(get_random_values(min_val=i_min, max_val=i_max, how_many=num_data_samples))
    i5.define_input_data(get_random_values(min_val=i_min, max_val=i_max, how_many=num_data_samples))
    i6.define_input_data(get_random_values(min_val=i_min, max_val=i_max, how_many=num_data_samples))
    i7.define_input_data(get_random_values(min_val=i_min, max_val=i_max, how_many=num_data_samples))
    for time_step in range(num_data_samples):
        i0_val = to_fixed_point(i0.get_output(time_step), fract_bits)
        i1_val = to_fixed_point(i1.get_output(time_step), fract_bits)
        i2_val = to_fixed_point(i2.get_output(time_step), fract_bits)
        i3_val = to_fixed_point(i3.get_output(time_step), fract_bits)
        i4_val = to_fixed_point(i4.get_output(time_step), fract_bits)
        i5_val = to_fixed_point(i5.get_output(time_step), fract_bits)
        i6_val = to_fixed_point(i6.get_output(time_step), fract_bits)
        i7_val = to_fixed_point(i7.get_output(time_step), fract_bits)
        o0_val = to_fixed_point(o0.get_output(time_step), fract_bits)
        if sum([i0_val, i1_val, i2_val, i3_val, i4_val, i5_val, i6_val, i7_val]) != o0_val:
            raise Exception(f"Found error in addition: {i0_val}+{i1_val}+{i2_val}+{i3_val}+{i4_val}+{i5_val}+{i6_val}+{i7_val} != {o0_val}")


if __name__ == '__main__':
    main()
