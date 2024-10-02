import math

try:
    import modules
    from circuit import Circuit
    from dfg import Graph
except:
    import pydsp.modules as modules
    from pydsp.circuit import Circuit
    from pydsp.dfg import Graph





# write simulations to csv files so you can visualize them online at
# https://www.csvplot.com
class CsvWriter:
    @staticmethod
    def write(file_path: str, obj: dict, idx_name: str, idx_scaling_factor: float = 1.0):
        num_samples = None
        for _, v in obj.items():
            if num_samples is None:
                num_samples = len(v)
            elif len(v) != num_samples:
                raise Exception(f"Simulation values must have equal length for CSV export")
        with open(file_path, "w") as f:
            header = idx_name
            for name in obj:
                header = f"{header};{name}"
            f.write(f"{header}\n")
            for t in range(num_samples):
                line = f"{idx_scaling_factor * t}"
                for name in obj:
                    line = f"{line};{obj[name][t]}"
                f.write(f"{line}\n")


# display the .dot files generated with this class at
# https://dreampuf.github.io/GraphvizOnline/
class GraphvizWriter:
    color_dict = {
        modules.Input: "blue",
        modules.Output: "blue",
        modules.Constant: "blue",
        modules.Add: "black",
        modules.Sub: "black",
        modules.Mult: "black",
        modules.Comparator: "red",
        modules.Register: "red",
        modules.Mux: "red",
        modules.ShiftLeft: "green",
        modules.ShiftRight: "green",
        modules.TruncateMSBs: "green"
    }

    typename_dict = {
        modules.Input: "Input",
        modules.Output: "Output",
        modules.Constant: "Const",
        modules.Add: "Add",
        modules.Sub: "Sub",
        modules.Mult: "Mult",
        modules.Comparator: "Comp",
        modules.Register: "Reg",
        modules.Mux: "MUX",
        modules.ShiftLeft: "LL",
        modules.ShiftRight: "RR",
        modules.TruncateMSBs: "Trunc"
    }

    @staticmethod
    def __write_circuit(file_path, circuit):
        if type(circuit) is not Circuit:
            raise Exception(f"Invalid object type '{type(circuit)}'")
        with open(file_path, "w") as f:
            f.write("digraph G {\n")
            for m in circuit.modules:
                color = GraphvizWriter.color_dict.get(type(m), "black")
                typename = GraphvizWriter.typename_dict.get(type(m), "unknown")
                f.write(f"  {m.name}_{typename} [shape=rect, color={color}]\n")
           
            for c in circuit.connections:
                src_module = c.src_module
                dst_module = c.dst_module
                dst_port = c.dst_port
                typename_src = GraphvizWriter.typename_dict.get(type(src_module), "unknown")
                typename_dst = GraphvizWriter.typename_dict.get(type(dst_module), "unknown")
                f.write(f"  {src_module.name}_{typename_src} -> {dst_module.name}_{typename_dst} [headlabel=\"{dst_port}\"]\n")
            # footer
            f.write("}\n")

    @staticmethod
    def __write_graph(file_path, graph):
        if type(graph) is not Graph:
            raise Exception(f"Invalid object type '{type(graph)}'")
        with open(file_path, "w") as f:
            f.write("digraph G {\n")
            
            for e in graph.edges:
                src_vertex = e.src_vertex
                dst_vertex = e.dst_vertex
                dst_port = e.dst_port
                distance = e.distance
                taillabel_str = f", taillabel=\"d={distance}\"" if distance > 0 else ""
                f.write(f"  {src_vertex.name} -> {dst_vertex.name} [headlabel=\"{dst_port}\"{taillabel_str}]\n")
            # footer
            f.write("}\n")

    @staticmethod
    def write(file_path: str, obj):
        if not file_path.endswith(".dot"):
            raise Exception(f"file path must end with '.dot'")
        if type(obj) is Circuit:
            return GraphvizWriter.__write_circuit(file_path, obj)
        if type(obj) is Graph:
            return GraphvizWriter.__write_graph(file_path, obj)
        

class VHDLWriter:
    @staticmethod
    def write(file_path: str, circuit: Circuit):
        if not (file_path.endswith(".vhd") or file_path.endswith(".vhdl")):
            raise Exception(f"file path must end with '.vhd' or '.vhdl'")

        with open(file_path, "w") as f:
            f.write("library IEEE;\n")
            f.write("use IEEE.STD_LOGIC_1164.ALL;\n")
            f.write("use IEEE.NUMERIC_STD.ALL;\n\n")

            module_types = VHDLWriter.collect_unique_module_types(circuit)

            for module_type in module_types:
                VHDLWriter.write_module_definition(f, module_type)

            # Write the top module that connects all module instances
            VHDLWriter.write_top_module(f, circuit)

    @staticmethod
    def collect_unique_module_types(circuit: Circuit):
        unique_types = {}
        for module in circuit.modules:
            module_class = type(module)
            if module_class not in unique_types:
                unique_types[module_class] = module
        return unique_types.values()

    @staticmethod
    def write_module_definition(f, module: modules):
        module_entity = VHDLWriter.generate_module_entity(module)
        module_architecture = VHDLWriter.generate_module_architecture(module)

        f.write(module_entity)
        f.write(module_architecture)

    @staticmethod
    def generate_module_entity(module: modules):
        entity_name = type(module).__name__
        inputs = []
        outputs = []

        generics = []

        for idx, input_name in enumerate(module.inputs):
            # Inputs will be of generic width
            inputs.append((input_name, "DATA_WIDTH"))

        outputs.append((module.output, "DATA_WIDTH"))

        generics.append("DATA_WIDTH : integer := 8")  # Default width

        entity_str = f"entity {entity_name} is\n"
        if generics:
            entity_str += "    generic (\n"
            entity_str += ";\n".join(f"        {g}" for g in generics)
            entity_str += "\n    );\n"
        entity_str += "    Port (\n"

        port_lines = []
        for name, width in inputs:
            port_lines.append(f"        {name} : in std_logic_vector({width}-1 downto 0)")
        for name, width in outputs:
            port_lines.append(f"        {name} : out std_logic_vector({width}-1 downto 0)")

        entity_str += ";\n".join(port_lines)
        entity_str += "\n    );\n"
        entity_str += f"end {entity_name};\n\n"

        return entity_str

    @staticmethod
    def generate_module_architecture(module: modules):
        entity_name = type(module).__name__
        architecture_str = f"architecture Behavioral of {entity_name} is\n"

        architecture_str += "begin\n"

        # Generate the behavior based on module type
        if isinstance(module, modules.Input):
            # Input module logic is not needed
            architecture_str += "    -- Input module logic is handled externally\n"
        elif isinstance(module, modules.Output):
            # Output module logic is not needed
            architecture_str += "    -- Output module logic is handled externally\n"
        elif isinstance(module, modules.Constant):
            # Assign constant value to output
            width = "DATA_WIDTH"
            architecture_str += f"    {module.output} <= std_logic_vector(to_unsigned(VALUE, {width}));\n"
        elif isinstance(module, modules.Add):
            # Addition operation
            op_str = VHDLWriter.generate_arithmetic_operation_generic(module, '+')
            architecture_str += f"    {module.output} <= {op_str};\n"
        elif isinstance(module, modules.Sub):
            # Subtraction operation
            op_str = VHDLWriter.generate_arithmetic_operation_generic(module, '-')
            architecture_str += f"    {module.output} <= {op_str};\n"
        elif isinstance(module, modules.Mult):
            # Multiplication operation
            op_str = VHDLWriter.generate_arithmetic_operation_generic(module, '*')
            architecture_str += f"    {module.output} <= {op_str};\n"
        elif isinstance(module, modules.Comparator):
            raise NotImplementedError("Not implemented")
            # Comparator operation
            op_str = VHDLWriter.generate_comparator_operation_generic(module)
            architecture_str += f"    {module.output} <= {op_str};\n"
        elif isinstance(module, modules.Mux):
            raise NotImplementedError("Not implemented")

            # Multiplexer operation
            op_str = VHDLWriter.generate_mux_operation_generic(module)
            architecture_str += f"    {module.output} <= {op_str};\n"
        elif isinstance(module, modules.ShiftLeft):
            # Shift left operation
            op_str = VHDLWriter.generate_shift_operation_generic(module, 'sll')
            architecture_str += f"    {module.output} <= {op_str};\n"
        elif isinstance(module, modules.ShiftRight):
            # Shift right operation
            op_str = VHDLWriter.generate_shift_operation_generic(module, 'srl')
            architecture_str += f"    {module.output} <= {op_str};\n"
        elif isinstance(module, modules.TruncateMSBs):
            raise NotImplementedError("Not implemented")
            # Truncate MSBs operation
            op_str = VHDLWriter.generate_truncate_operation_generic(module)
            architecture_str += f"    {module.output} <= {op_str};\n"
        elif isinstance(module, modules.Register):
            # Register operation
            op_str = VHDLWriter.generate_register_operation_generic(module)
            architecture_str += op_str
        else:
            raise NotImplementedError("Not implemented")
            architecture_str += "    -- Unsupported module type\n"

        # End architecture body
        architecture_str += f"end Behavioral;\n\n"

        return architecture_str

    @staticmethod
    def generate_arithmetic_operation_generic(module: modules, operator: str):
        # Handle arithmetic operations with generics
        input_names = [name for name in module.inputs]
        if module.data_type == modules.DataType.UNSIGNED:
            op_str = f'std_logic_vector(unsigned({input_names[0]}) {operator} unsigned({input_names[1]}))'
        else:
            op_str = f'std_logic_vector(signed({input_names[0]}) {operator} signed({input_names[1]}))'
        return op_str

    

    @staticmethod
    def generate_mux_operation_generic(module: modules.Mux):
        data_inputs = module.inputs[:-1]
        select_input = module.inputs[-1]
        op_str = f"{data_inputs[0]}"
        for i in range(1, len(data_inputs)):
            op_str = f"{op_str} when {select_input} = {i} else {data_inputs[i]}"
        return op_str

    @staticmethod
    def generate_shift_operation_generic(module: modules.Module, shift_op: str):
        input_name = module.inputs[0]
        shift_length = module.shift_length
        if module.data_type == modules.DataType.UNSIGNED:
            op_str = f'std_logic_vector(unsigned({input_name}) {shift_op} {shift_length})'
        else:
            op_str = f'std_logic_vector(signed({input_name}) {shift_op} {shift_length})'
        return op_str


    @staticmethod
    def generate_register_operation_generic(module: modules.Register):
        input_name = module.inputs[0]
        output_name = module.output
        width = "DATA_WIDTH"
        process_str = f"""
    process(clk)
    begin
        if rising_edge(clk) then
            {output_name} <= {input_name};
        end if;
    end process;
"""
        return process_str

    @staticmethod
    def write_top_module(f, circuit: Circuit):
        # Write the top module entity
        entity_str = "entity TopModule is\n"
        entity_str += "    Port (\n"

        port_lines = []
        # Collect all inputs and outputs from Input and Output modules
        for module in circuit.modules:
            if isinstance(module, modules.Input):
                width = module.output_word_size()
                port_lines.append(f"        {module.name} : in std_logic_vector({width - 1} downto 0)")
            elif isinstance(module, modules.Output):
                width = module.output_word_size()
                port_lines.append(f"        {module.name} : out std_logic_vector({width - 1} downto 0)")

        has_register = any(isinstance(m, modules.Register) for m in circuit.modules)
        if has_register:
            port_lines.append("        clk : in std_logic")
            port_lines.append("        rst : in std_logic")

        entity_str += ";\n".join(port_lines)
        entity_str += "\n    );\n"
        entity_str += "end TopModule;\n\n"

        f.write(entity_str)

        architecture_str = "architecture Structural of TopModule is\n"

        for module in circuit.modules:
            if not isinstance(module, (modules.Input, modules.Output)):
                width = module.output_word_size()
                architecture_str += f"    signal {module.name}_sig : std_logic_vector({width - 1} downto 0);\n"

        architecture_str += "\nbegin\n"

        for module in circuit.modules:
            if isinstance(module, modules.Input):
                # Input signals are top-level ports
                architecture_str += f"    {module.name}_sig <= {module.name};\n"
            elif isinstance(module, modules.Output):
                # Output signals are top-level ports
                src_module = module.input_modules[0]
                src_signal = VHDLWriter.get_signal_name(src_module)
                architecture_str += f"    {module.name} <= {src_signal};\n"
            else:
                # Instantiate the module
                instance_name = module.name + "_inst"
                module_type_name = type(module).__name__
                width = module.output_word_size()

                # Map generics
                architecture_str += f"    {instance_name} : entity work.{module_type_name}\n"
                architecture_str += f"        generic map (\n"
                architecture_str += f"            DATA_WIDTH => {width}\n"
                architecture_str += "        )\n"

                architecture_str += "        port map (\n"

                # Map inputs
                port_mappings = []
                for idx, input_name in enumerate(module.inputs):
                    connected_module = module.input_modules[idx]
                    src_signal = VHDLWriter.get_signal_name(connected_module)
                    port_mappings.append(f"            {input_name} => {src_signal}")

                # Map outputs
                port_mappings.append(f"            {module.output} => {module.name}_sig")

                # Map clock and reset if needed
                if isinstance(module, modules.Register):
                    port_mappings.append("            clk => clk")
                    port_mappings.append("            rst => rst")

                architecture_str += ",\n".join(port_mappings)
                architecture_str += "\n        );\n"

        architecture_str += "end Structural;\n\n"

        f.write(architecture_str)

    @staticmethod
    def get_signal_name(module):
        if isinstance(module, modules.Input):
            return f"{module.name}_sig"
        elif isinstance(module, modules.Output):
            return f"{module.name}"
        else:
            return f"{module.name}_sig"
