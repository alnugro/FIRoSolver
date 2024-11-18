from circuit import Circuit
from modules import *
from io_utility import *


def main():
    # create a circuit for a truncated multiplication
    c = Circuit()
    # define data type used within the circuit => 8 bit signed numbers with 4 fractional bits
    dat = DataType.SIGNED
    num_bits = 5
    i_min = -(2**(num_bits-1))
    i_max = -i_min-1
    # create an input for our circuit
    # i0 = c.add(Input(input_ranges=[[i_min, i_max]], data_type=dat, name="X0"))
    i0 = c.add(Constant(value=0, data_type=dat, name="X0"))
    i1 = c.add(Constant(value=1, data_type=dat, name="X1"))
    i2 = c.add(Constant(value=2, data_type=dat, name="X2"))

    sel = c.add(Input(input_ranges=[[0, 2]], data_type=dat, name="sel"))
    c.connect(src_module=o1, dst_module=out, dst_port=0)


    o1 = c.add(Mux(input_ranges=[i0.output_range(), i1.output_range(), i2.output_range(), sel.output_range()], data_type=dat, name="Mux0"))

    # #a0 = (i0 << 2) - i0
    # s0 = c.add(ShiftLeft(input_ranges=[i0.output_range()], data_type=dat, name="Shift0", shift_length=2))
    # c.connect(src_module=i0, dst_module=s0, dst_port=0)

    # a0 = c.add(Sub(input_ranges=[s0.output_range(), i0.output_range()], data_type=dat, name="Add1")) 
    # c.connect(src_module=s0, dst_module=a0, dst_port=0)
    # c.connect(src_module=i0, dst_module=a0, dst_port=1)

    # #a1 is time delayed result of a0
    # a1 = c.add(Register(input_ranges=[a0.output_range()], data_type=dat, name="Reg0"))
    # c.connect(src_module=a0, dst_module=a1, dst_port=0)
    # #a0 + a1
    # o1 = c.add(Add(input_ranges=[a0.output_range(), a1.output_range()], data_type=dat, name="Add2")) 
    # c.connect(src_module=a0, dst_module=o1, dst_port=0)
    # c.connect(src_module=a1, dst_module=o1, dst_port=1)

    
    
    out = c.add(Output(input_ranges=[o1.output_range()], data_type=dat, name="Y0"))
    

    c.connect(src_module=o1, dst_module=out, dst_port=0)


    # validate circuit (e.g., open input ports, invalid connections, etc.)
    c.validate()
    # print info
    c.print_info()
    # run simulation
    i0.define_input_data(get_random_values(0, 100, 5))
    i0_val_bef = None
    for time_step in range(10):
        i0_val = i0.get_output(time_step)
        out_val = out.get_output(time_step)
        print(f"t={time_step} => out={out_val} , i0 ={i0_val}")

        if i0_val_bef:
            print(f"out = {i0_val * 4 - i0_val +  i0_val_bef * 4 - i0_val_bef}")
        
        i0_val_bef = i0_val
        
        # print(f"a0 = (i0 << 2) + i0 + a0(-1) = (i0 << 2) + i0 +")
        # print(f"out = i0 * 4 + i0 +  i0(-1) * 4 + i0(-1)")

def Mux_test():
    # create a circuit for a truncated multiplication
    c = Circuit()
    # define data type used within the circuit => 8 bit signed numbers with 4 fractional bits
    dat = DataType.UNSIGNED
    num_bits = 5
    i_min = 0
    i_max = (2**(num_bits-1))
    # create an input for our circuit
    # i0 = c.add(Input(input_ranges=[[i_min, i_max]], data_type=dat, name="X0"))
    i0 = c.add(Constant(value=5, data_type=dat, name="C0"))
    i1 = c.add(Input(input_ranges=[[i_min, i_max]], data_type=dat, name="i1"))

    sel = c.add(Input(input_ranges=[[0, 1]], data_type=dat, name="sel"))
    
    o1 = c.add(Mux(input_ranges=[i0.output_range(), i1.output_range(), sel.output_range()], data_type=dat, name="Mux0"))
    

    out = c.add(Output(input_ranges=[o1.output_range()], data_type=dat, name="Y0"))
    
    c.connect(src_module=i0, dst_module=o1, dst_port=0)
    c.connect(src_module=i1, dst_module=o1, dst_port=1)
    c.connect(src_module=sel, dst_module=o1, dst_port=2)

    c.connect(src_module=o1, dst_module=out, dst_port=0)

    for connect in c.connections:
        print(connect)
        print("src_module",connect.src_module.name)
        print("dst_module",connect.dst_module.name)
        print("dst_port",connect.dst_port)

    # validate circuit (e.g., open input ports, invalid connections, etc.)
    # c.validate()
    # # print info
    # c.print_info()
    # # run simulation
    # i1.define_input_data(get_random_values(0, 1, 5))
    # i0_val_bef = None
    # for time_step in range(10):
    #     i0_val = i0.get_output(time_step)
    #     out_val = out.get_output(time_step)
    #     print(f"t={time_step} => out={out_val} , i0 ={i0_val}")

    #     if i0_val_bef:
    #         print(f"out = {i0_val * 4 - i0_val +  i0_val_bef * 4 - i0_val_bef}")
        
    #     i0_val_bef = i0_val
        
        # print(f"a0 = (i0 << 2) + i0 + a0(-1) = (i0 << 2) + i0 +")
        # print(f"out = i0 * 4 + i0 +  i0(-1) * 4 + i0(-1)")

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
        




if __name__ == '__main__':
    Mux_test()
