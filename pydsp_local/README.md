# pydsp-toolbox
This is a Python toolbox for developing basic DSP applications/circuits for FPGAs. It is intended to run stand-alone without any dependencies in a class room environment.
Note that the code was intended to be used for teaching purposes. Hence I tried to keep it as concise and simple as possible, so do not expect anything to be optimized for speed/efficiency.


## Getting started
Just copy the relevant source files into your project and follow the examples given in 'circuit_examples.py' and 'dfg_examples.py' for basic usage.
Write me a message/mail if you are interested in the PDF that describes the exercises corresponding to:
- exercise_*.py
- solution_*.py


## Features
Currently, you can create circuits for signed & unsigned integer & fixed point arithmetic.
All circuit elements (see 'modules.py') compute their output word sizes on their own via the given input data ranges.
Registers and truncations (LSBs & MSBs) can be used to create circuits with feed-back loops (e.g., IIR filters).
Once you have created a circuit, you can provide data for your inputs, simulate the circuit (in discrete time steps/clock cycles) and request output values from all modules.


## Future plans
1) VHDL/Verilog code generator to deploy circuits on FPGAs