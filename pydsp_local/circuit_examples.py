from circuit import Circuit
from modules import *
from io_utility import *
from file_writers import GraphvizWriter


def example_1():
    # create a circuit that contains all the basic modules
    c = Circuit()
    # define data type used within the circuit => 8 bit signed numbers
    dat = DataType.SIGNED
    num_bits = 8
    i_min = -(2**(num_bits-1))
    i_max = -i_min-1
    # create some inputs for our circuit
    i0 = c.add(Input(input_ranges=[[i_min, i_max]], data_type=dat, name="X0"))
    i1 = c.add(Input(input_ranges=[[i_min, i_max]], data_type=dat, name="X1"))
    i2 = c.add(Input(input_ranges=[[i_min, i_max]], data_type=dat, name="X2"))
    i3 = c.add(Input(input_ranges=[[0, 1]], data_type=DataType.UNSIGNED, name="X3"))  # this one is special since it's used as a 2:1 MUX select input
    # create shifters
    s0 = c.add(ShiftLeft(input_ranges=[i0.output_range()], data_type=dat, name="Shift0", shift_length=1))
    s1 = c.add(ShiftRight(input_ranges=[i1.output_range()], data_type=dat, name="Shift1", shift_length=2))
    # create a register
    r0 = c.add(Register(input_ranges=[i2.output_range()], data_type=dat, name="Reg0"))
    # create a constant
    c0 = c.add(Constant(value=-2, data_type=dat, name="C0"))
    # create arithmetic units
    m0 = c.add(Mult(input_ranges=[s0.output_range(), s1.output_range()], data_type=dat, name="Mult0"))
    a0 = c.add(Add(input_ranges=[m0.output_range(), r0.output_range()], data_type=dat, name="Add0"))
    u0 = c.add(Sub(input_ranges=[a0.output_range(), c0.output_range()], data_type=dat, name="Sub0"))
    # create a multiplexer
    x0 = c.add(Mux(input_ranges=[u0.output_range(), a0.output_range(), i3.output_range()], data_type=dat, name="Mux0"))
    # create an msb truncation
    t0 = c.add(TruncateMSBs(input_ranges=[x0.output_range()], data_type=dat, name="Trunc0", how_many=4))
    # create the output
    o0 = c.add(Output(input_ranges=[t0.output_range()], data_type=dat, name="Y0"))
    # connect them
    # -> s0
    c.connect(src_module=i0, dst_module=s0, dst_port=0)
    # -> s1
    c.connect(src_module=i1, dst_module=s1, dst_port=0)
    # -> m0
    c.connect(src_module=s0, dst_module=m0, dst_port=0)
    c.connect(src_module=s1, dst_module=m0, dst_port=1)
    # -> r0
    c.connect(src_module=i2, dst_module=r0, dst_port=0)
    # -> a0
    c.connect(src_module=m0, dst_module=a0, dst_port=0)
    c.connect(src_module=r0, dst_module=a0, dst_port=1)
    # -> u0
    c.connect(src_module=a0, dst_module=u0, dst_port=0)
    c.connect(src_module=c0, dst_module=u0, dst_port=1)
    # -> x0
    c.connect(src_module=u0, dst_module=x0, dst_port=0)
    c.connect(src_module=a0, dst_module=x0, dst_port=1)
    c.connect(src_module=i3, dst_module=x0, dst_port=2)
    # -> t0
    c.connect(src_module=x0, dst_module=t0, dst_port=0)
    # -> o0
    c.connect(src_module=t0, dst_module=o0, dst_port=0)
    # validate circuit (e.g., open input ports, invalid connections, etc.)
    c.validate()
    # print info
    c.print_info()
    # run simulation
    i0.define_input_data([-128, 122, 59])
    i1.define_input_data([-1, 55, 127]) 
    i2.define_input_data([33, -47, 7])
    i3.define_input_data([0, 1, 0])
    for time_step in range(10):
        o0_val = o0.get_output(time_step)
        print(f"t={time_step} => o0={o0_val}")
    # write graphviz .dot file
    GraphvizWriter.write("example.dot", c)


def example_2():
    # create a circuit for a truncated multiplication
    c = Circuit()
    # define data type used within the circuit => 16 bit signed numbers with 8 fractional bits
    dat = DataType.SIGNED
    num_bits = 16
    fract_bits = 8
    integer_bits = num_bits - fract_bits
    i_min = -(2**(num_bits-1))
    i_max = -i_min-1
    # create some inputs for our circuit
    i0 = c.add(Input(input_ranges=[[i_min, i_max]], data_type=dat, name="X0"))
    i1 = c.add(Input(input_ranges=[[i_min, i_max]], data_type=dat, name="X1"))
    # create multiplier -> it will produce twice the number of bits with twice the number of fractional bits
    m0 = c.add(Mult(input_ranges=[i0.output_range(), i1.output_range()], data_type=dat, name="Mult0"))
    # shift right to truncate excess LSBs
    s0 = c.add(ShiftRight(input_ranges=[m0.output_range()], data_type=dat, name="Shift1", shift_length=fract_bits))
    # truncate excess MSBs
    t0 = c.add(TruncateMSBs(input_ranges=[s0.output_range()], data_type=dat, name="Trunc0", how_many=integer_bits))
    # create the output
    o0 = c.add(Output(input_ranges=[t0.output_range()], data_type=dat, name="Y0"))
    # connect the modules
    c.connect(src_module=i0, dst_module=m0, dst_port=0)
    c.connect(src_module=i1, dst_module=m0, dst_port=1)
    c.connect(src_module=m0, dst_module=s0, dst_port=0)
    c.connect(src_module=s0, dst_module=t0, dst_port=0)
    c.connect(src_module=t0, dst_module=o0, dst_port=0)
    # validate circuit (e.g., open input ports, invalid connections, etc.)
    c.validate()
    # print info
    c.print_info()
    # run simulation
    i0.define_input_data(from_fixed_point([-1.5, 7.3, 1.1], fract_bits))
    i1.define_input_data(from_fixed_point([3.5, -1.5, 1.3], fract_bits))
    for time_step in range(10):
        raw_o0_val = o0.get_output(time_step)
        o0_val = to_fixed_point(raw_o0_val, fract_bits)
        print(f"t={time_step} => o0={o0_val}")


def example_3():
    # create a circuit for a simple constant multiplication
    c = Circuit()
    # define data type used within the circuit => 8 bit signed numbers
    dat = DataType.SIGNED
    num_bits = 8
    i_min = -(2**(num_bits-1))
    i_max = -i_min-1
    # create the input for our circuit
    i0 = c.add(Input(input_ranges=[[i_min, i_max]], data_type=dat, name="X0"))
    # a0 = (x0 << 3) - x0 = 7*x0
    s0 = c.add(ShiftLeft(input_ranges=[i0.output_range()], data_type=dat, name="Shift0", shift_length=3))
    a0 = c.add(Sub(input_ranges=[s0.output_range(), i0.output_range()], data_type=dat, name="Sub0"))
    # a1 = (a0 << 2) + a0 = 5*a0 = 35*x0
    s1 = c.add(ShiftLeft(input_ranges=[a0.output_range()], data_type=dat, name="Shift1", shift_length=2))
    a1 = c.add(Add(input_ranges=[s1.output_range(), a0.output_range()], data_type=dat, name="Add1"))
    # y = a1
    o0 = c.add(Output(input_ranges=[a1.output_range()], data_type=dat, name="Y0"))
    # connect the modules
    c.connect(src_module=i0, dst_module=s0, dst_port=0)

    c.connect(src_module=s0, dst_module=a0, dst_port=0)
    c.connect(src_module=i0, dst_module=a0, dst_port=1)
    
    c.connect(src_module=a0, dst_module=s1, dst_port=0)
    c.connect(src_module=s1, dst_module=a1, dst_port=0)
    c.connect(src_module=a0, dst_module=a1, dst_port=1)
    c.connect(src_module=a1, dst_module=o0, dst_port=0)
    # validate circuit (e.g., open input ports, invalid connections, etc.)
    c.validate()
    # print info
    c.print_info()
    # run simulation
    i0.define_input_data(get_random_values(i_min, i_max, 100))
    for time_step in range(100):
        i0_val = i0.get_output(time_step)
        o0_val = o0.get_output(time_step)
        if i0_val*35 != o0_val:
            raise Exception(f"Wrong value for i={i0_val} at time step {time_step}")
        print(f"t={time_step} => o0={o0_val} = {i0_val}*35")


def example_4():
    # create a circuit for a mac unit with a cycle that computes y[i] = y[i-1] + x_0[i]*x_1[i]
    # here we have to truncate the adder result before feeding it back
    # hence, we actually compute y[i] = truncate(y[i-1]) + x_0[i]*x_1[i]
    # due to the recurrence, we also have to manually define the data range of one of the adder inputs
    # this means that we might produce overflows if we accumulate too many values with the same sign over time
    c = Circuit()
    # define data type used within the circuit => 8 bit signed numbers
    dat = DataType.SIGNED
    i_min_8 = -(2**(8-1))
    i_max_8 = -i_min_8-1
    i_min_16 = -(2**(16-1))
    i_max_16 = -i_min_16-1
    # create the input for our circuit
    i0 = c.add(Input(input_ranges=[[i_min_8, i_max_8]], data_type=dat, name="X0"))
    i1 = c.add(Input(input_ranges=[[i_min_8, i_max_8]], data_type=dat, name="X1"))
    # data path
    m0 = c.add(Mult(input_ranges=[i0.output_range(), i1.output_range()], data_type=dat, name="Mult0"))
    a0 = c.add(Add(input_ranges=[m0.output_range(), [i_min_16, i_max_16]], data_type=dat, name="Add0"))
    t0 = c.add(TruncateMSBs(input_ranges=[a0.output_range()], data_type=dat, name="Trunc0", how_many=1))
    r0 = c.add(Register(input_ranges=[t0.output_range()], data_type=dat, name="Reg0"))
    # output
    o0 = c.add(Output(input_ranges=[a0.output_range()], data_type=dat, name="Y0"))
    # connections
    c.connect(src_module=i0, dst_module=m0, dst_port=0)
    c.connect(src_module=i1, dst_module=m0, dst_port=1)
    c.connect(src_module=m0, dst_module=a0, dst_port=0)
    c.connect(src_module=r0, dst_module=a0, dst_port=1)
    c.connect(src_module=a0, dst_module=t0, dst_port=0)
    c.connect(src_module=t0, dst_module=r0, dst_port=0)
    c.connect(src_module=a0, dst_module=o0, dst_port=0)
    # validate circuit (e.g., open input ports, invalid connections, etc.)
    c.validate()
    # print info
    c.print_info()
    # run simulation
    # -> this choice of input data produces overflows because all the values have the same signs (which is expected/normal behaviour)!
    i0.define_input_data([127, 122, 59, 0, 123, 44, 19])
    i1.define_input_data([0, 123, 44, 19, 127, 122, 59])
    for time_step in range(7):
        o0_val = o0.get_output(time_step)
        print(f"t={time_step} => o0={o0_val}")


def example_5():
    # create a circuit for y = |a-b| using basic elements
    c = Circuit()
    # define data type used within the circuit => 8 bit signed numbers
    dat = DataType.SIGNED
    i_min = -(2**(8-1))
    i_max = -i_min-1
    # inputs
    i0 = c.add(Input(input_ranges=[[i_min, i_max]], data_type=dat, name="X0"))
    i1 = c.add(Input(input_ranges=[[i_min, i_max]], data_type=dat, name="X1"))
    # subtracters
    s0 = c.add(Sub(input_ranges=[i0.output_range(), i1.output_range()], data_type=dat, name="Sub0"))
    s1 = c.add(Sub(input_ranges=[i1.output_range(), i0.output_range()], data_type=dat, name="Sub1"))
    # comparator and mux
    c0 = c.add(Comparator(input_ranges=[i0.output_range(), i1.output_range()], name="Comp0", comparator_type=ComparatorType.LEQ))
    m0 = c.add(Mux(input_ranges=[s0.output_range(), s1.output_range(), c0.output_range()], data_type=dat, name="Mux0"))
    # output
    o0 = c.add(Output(input_ranges=[m0.output_range()], data_type=dat, name="Y0"))
    # connections
    c.connect(src_module=i0, dst_module=s0, dst_port=0)
    c.connect(src_module=i1, dst_module=s0, dst_port=1)
    c.connect(src_module=i0, dst_module=s1, dst_port=1)
    c.connect(src_module=i1, dst_module=s1, dst_port=0)
    c.connect(src_module=i0, dst_module=c0, dst_port=0)
    c.connect(src_module=i1, dst_module=c0, dst_port=1)
    c.connect(src_module=s0, dst_module=m0, dst_port=0)
    c.connect(src_module=s1, dst_module=m0, dst_port=1)
    c.connect(src_module=c0, dst_module=m0, dst_port=2)
    c.connect(src_module=m0, dst_module=o0, dst_port=0)
    # validate circuit (e.g., open input ports, invalid connections, etc.)
    c.validate()
    # print info
    c.print_info()
    # run simulation
    i0.define_input_data([127, -122, 59, 0, -123, 44, -19])
    i1.define_input_data([0, 123, 44, -19, 127, 122, 59])
    for time_step in range(7):
        o0_val = o0.get_output(time_step)
        print(f"t={time_step} => o0={o0_val}")


def main():
    # example_1()
    #example_2()
    example_3()
    #example_4()
    #example_5()


if __name__ == '__main__':
    main()