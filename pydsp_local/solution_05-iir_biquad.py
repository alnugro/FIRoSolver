from circuit import Circuit
from modules import *
from io_utility import *
from math import sin, pi
from file_writers import *


def get_sine_wave(amplitude, num_samples, sample_frequency, sine_frequency):
    return [amplitude*sin(2*pi*sine_frequency*i/sample_frequency) for i in range(num_samples)]


def main():
    # suppress debug outputs
    verbose = False
    # create a circuit for a truncated multiplication
    c = Circuit()
    # define data type used within the circuit => 8 bit signed numbers with 4 fractional bits
    dat = DataType.SIGNED
    fract_bits_input = 8
    num_bits_input = 12
    integer_bits_input = num_bits_input - fract_bits_input
    i_min = -(2**(num_bits_input-1))
    i_max = -i_min-1
    i_range = [i_min, i_max]
    fract_bits_reg = fract_bits_input + 4
    num_bits_reg = fract_bits_reg + integer_bits_input + 8  # that should be enough
    integer_bits_reg = num_bits_reg - fract_bits_reg
    r_min = -(2**(num_bits_reg-1))
    r_max = -r_min-1
    r_range = [r_min, r_max]
    fract_bits_constants = 12
    fract_bits_mult = fract_bits_constants + fract_bits_reg
    # compute filter coefficient representations
    filter_coeff_a1_raw = -1.8984098541537924 * -1
    filter_coeff_a2_raw = 0.9032371264734055 * -1
    filter_coeff_b0_raw = 0.0012068180799032807
    filter_coeff_b1_raw = 0.0024136361598065615
    filter_coeff_b2_raw = 0.0012068180799032807
    filter_coeff_a1 = from_fixed_point([filter_coeff_a1_raw], fract_bits_constants)[0]
    filter_coeff_a2 = from_fixed_point([filter_coeff_a2_raw], fract_bits_constants)[0]
    filter_coeff_b0 = from_fixed_point([filter_coeff_b0_raw], fract_bits_constants)[0]
    filter_coeff_b1 = from_fixed_point([filter_coeff_b1_raw], fract_bits_constants)[0]
    filter_coeff_b2 = from_fixed_point([filter_coeff_b2_raw], fract_bits_constants)[0]
    # create an input for our circuit
    i0 = c.add(Input(input_ranges=[i_range], data_type=dat, name="X"))
    # left shifter to align the comma position
    s0 = c.add(ShiftLeft(input_ranges=[i0.output_range()], data_type=dat, name="X0_shift", shift_length=fract_bits_mult-fract_bits_input))
    # create the registers
    r1 = c.add(Register(input_ranges=[r_range], data_type=dat, name="R0"))
    r2 = c.add(Register(input_ranges=[r_range], data_type=dat, name="R1"))
    # create the constants
    ca1 = c.add(Constant(value=filter_coeff_a1, data_type=dat, name="A1"))
    ca2 = c.add(Constant(value=filter_coeff_a2, data_type=dat, name="A2"))
    cb0 = c.add(Constant(value=filter_coeff_b0, data_type=dat, name="B0"))
    cb1 = c.add(Constant(value=filter_coeff_b1, data_type=dat, name="B1"))
    cb2 = c.add(Constant(value=filter_coeff_b2, data_type=dat, name="B2"))
    # create multipliers
    ma1 = c.add(Mult(input_ranges=[r_range, ca1.output_range()], data_type=dat, name="MultA1"))
    ma2 = c.add(Mult(input_ranges=[r_range, ca2.output_range()], data_type=dat, name="MultA2"))
    mb0 = c.add(Mult(input_ranges=[r_range, cb0.output_range()], data_type=dat, name="MultB0"))
    mb1 = c.add(Mult(input_ranges=[r_range, cb1.output_range()], data_type=dat, name="MultB1"))
    mb2 = c.add(Mult(input_ranges=[r_range, cb2.output_range()], data_type=dat, name="MultB2"))
    # create adders
    a0 = c.add(Add(input_ranges=[ma1.output_range(), ma2.output_range()], data_type=dat, name="Add0"))
    a1 = c.add(Add(input_ranges=[s0.output_range(), a0.output_range()], data_type=dat, name="Add1"))
    a2 = c.add(Add(input_ranges=[mb1.output_range(), mb2.output_range()], data_type=dat, name="Add2"))
    a3 = c.add(Add(input_ranges=[mb0.output_range(), a2.output_range()], data_type=dat, name="Add3"))
    # create quantization
    q0 = c.add(ShiftRight(input_ranges=[a1.output_range()], data_type=dat, name="Q0", shift_length=fract_bits_constants))
    word_size_a1 = a1.output_word_size()
    integer_bits_a1 = word_size_a1 - fract_bits_mult
    q1 = c.add(TruncateMSBs(input_ranges=[q0.output_range()], data_type=dat, name="Q1", how_many=integer_bits_a1-integer_bits_reg))
    # create the output
    o0 = c.add(Output(input_ranges=[a3.output_range()], data_type=dat, name="Y"))
    # connect the modules 
    c.connect(src_module=i0, dst_module=s0, dst_port=0)
    c.connect(src_module=a1, dst_module=q0, dst_port=0)
    c.connect(src_module=q0, dst_module=q1, dst_port=0)
    c.connect(src_module=q1, dst_module=r1, dst_port=0)
    c.connect(src_module=r1, dst_module=r2, dst_port=0)
    c.connect(src_module=q1, dst_module=mb0, dst_port=0)
    c.connect(src_module=r1, dst_module=ma1, dst_port=0)
    c.connect(src_module=r1, dst_module=mb1, dst_port=0)
    c.connect(src_module=r2, dst_module=ma2, dst_port=0)
    c.connect(src_module=r2, dst_module=mb2, dst_port=0)
    c.connect(src_module=cb0, dst_module=mb0, dst_port=1)
    c.connect(src_module=cb1, dst_module=mb1, dst_port=1)
    c.connect(src_module=cb2, dst_module=mb2, dst_port=1)
    c.connect(src_module=ca1, dst_module=ma1, dst_port=1)
    c.connect(src_module=ca2, dst_module=ma2, dst_port=1)
    c.connect(src_module=ma1, dst_module=a0, dst_port=0)
    c.connect(src_module=ma2, dst_module=a0, dst_port=1)
    c.connect(src_module=s0, dst_module=a1, dst_port=0)
    c.connect(src_module=a0, dst_module=a1, dst_port=1)
    c.connect(src_module=mb1, dst_module=a2, dst_port=0)
    c.connect(src_module=mb2, dst_module=a2, dst_port=1)
    c.connect(src_module=mb0, dst_module=a3, dst_port=0)
    c.connect(src_module=a2, dst_module=a3, dst_port=1)
    c.connect(src_module=a3, dst_module=o0, dst_port=0)
    # validate circuit (raise an exception for, e.g., open input ports, invalid connections, etc.)
    c.validate()
    # print info
    if verbose:
        c.print_info()
    # export .dot graph
    GraphvizWriter.write(file_path="iir_biqu.dot", obj=c)
    # experimentally validate that we do not produce any overflows by our quantization
    num_data_samples = 1000
    i0.define_input_data(get_random_values(min_val=i_min, max_val=i_max, how_many=num_data_samples))
    for time_step in range(num_data_samples):
        if verbose:
            # print values
            print(f"t={time_step}:")
            # inputs + registers
            print(f"  i0 = {to_fixed_point(i0.get_output(time_step), fract_bits_input)}")
            print(f"  r1 = {to_fixed_point(r1.get_output(time_step), fract_bits_reg)}")
            print(f"  r2 = {to_fixed_point(r2.get_output(time_step), fract_bits_reg)}")
            # shifter to align comma position
            print(f"  s0 = {to_fixed_point(s0.get_output(time_step), fract_bits_mult)}")
            # rounding
            print(f"  q0 = {to_fixed_point(q0.get_output(time_step), fract_bits_reg)}")
            print(f"  q1 = {to_fixed_point(q1.get_output(time_step), fract_bits_reg)}")
            # multipliers
            print(f"  ma1 = {to_fixed_point(ma1.get_output(time_step), fract_bits_mult)} = r1 * {filter_coeff_a1_raw}")
            print(f"  ma2 = {to_fixed_point(ma2.get_output(time_step), fract_bits_mult)} = r2 * {filter_coeff_a2_raw}")
            print(f"  mb0 = {to_fixed_point(mb0.get_output(time_step), fract_bits_mult)} = q1 * {filter_coeff_b0_raw}")
            print(f"  mb1 = {to_fixed_point(mb1.get_output(time_step), fract_bits_mult)} = r1 * {filter_coeff_b1_raw}")
            print(f"  mb2 = {to_fixed_point(mb2.get_output(time_step), fract_bits_mult)} = r2 * {filter_coeff_b2_raw}")
            # adders
            print(f"  a0 = {to_fixed_point(a0.get_output(time_step), fract_bits_mult)} = ma1 + ma2")
            print(f"  a1 = {to_fixed_point(a1.get_output(time_step), fract_bits_mult)} = a0 + s0")
            print(f"  a2 = {to_fixed_point(a2.get_output(time_step), fract_bits_mult)} = mb1 + mb2")
            print(f"  a3 = {to_fixed_point(a3.get_output(time_step), fract_bits_mult)} = a2 + mb0")
        # detect overflow
        a1_val = to_fixed_point(a1.get_output(time_step), fract_bits_mult)
        q1_val = to_fixed_point(q1.get_output(time_step), fract_bits_reg)
        if (a1_val < 0 and q1_val >= 0) or (a1_val < 0 and q1_val >= 0):
            raise Exception(f"Detected overflow in time step {time_step}")
    # the filter is a low-pass with a -3dB cutoff frequency at 500Hz assuming a sample rate of 44.1kHz
    # => frequencies below 500Hz should not pass through the filter (i.e., their amplitudes are attenuated)
    # => frequencies above 500Hz should pass through the filter without any distortion/attenuation
    # => create csv files with X(t) and Y(t)
    # => visualize them online: https://www.csvplot.com
    sample_frequency = 44100.0
    cutoff_frequency = 500.0
    passband_frequency = cutoff_frequency / 2.0
    stopband_frequency = cutoff_frequency * 2.0
    num_data_samples_passband = 1000
    num_data_samples_stopband = num_data_samples_passband
    # check pass band
    i0.define_input_data(from_fixed_point(get_sine_wave(amplitude=1.0, num_samples=num_data_samples_passband, sample_frequency=sample_frequency, sine_frequency=passband_frequency), fract_bits_input))
    r1.reset(0)  # reset registers between simulation runs!
    r2.reset(0)  # reset registers between simulation runs!
    i_vals = []
    o_vals = []
    for time_step in range(num_data_samples_passband):
        i_vals.append(to_fixed_point(i0.get_output(time_step), fract_bits_input))
        o_vals.append(to_fixed_point(o0.get_output(time_step), fract_bits_mult))
    CsvWriter.write(file_path="pass_band.csv", obj={"X": i_vals, "Y": o_vals}, idx_name="t [ms]", idx_scaling_factor=1000.0/sample_frequency)
    # check stop band
    i0.define_input_data(from_fixed_point(get_sine_wave(amplitude=1.0, num_samples=num_data_samples_stopband, sample_frequency=sample_frequency, sine_frequency=stopband_frequency), fract_bits_input))
    r1.reset(0)  # reset registers between simulation runs!
    r2.reset(0)  # reset registers between simulation runs!
    i_vals = []
    o_vals = []
    for time_step in range(num_data_samples_stopband):
        i_vals.append(to_fixed_point(i0.get_output(time_step), fract_bits_input))
        o_vals.append(to_fixed_point(o0.get_output(time_step), fract_bits_mult))
    CsvWriter.write(file_path="stop_band.csv", obj={"X": i_vals, "Y": o_vals}, idx_name="t [ms]", idx_scaling_factor=1000.0/sample_frequency)


if __name__ == '__main__':
    main()
