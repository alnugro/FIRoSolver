import math

try:
    import modules
    from circuit import Circuit
    from dfg import Graph
except:
    import pydsp_local.modules as modules
    from pydsp_local.circuit import Circuit
    from pydsp_local.dfg import Graph





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
            # Collect unique module types
            module_types = VHDLWriter.collect_unique_module_types(circuit)

            # Write VHDL code for each unique module type
            for module_type in module_types:
                VHDLWriter.write_module_definition(f, module_type)

            # Write the top module that connects all module instances
            VHDLWriter.write_top_module(f, circuit)

    @staticmethod
    def entity_name_changer(entity_name: str):
        if entity_name == "Register":
            entity_name = "Regist"
        elif entity_name == "ShiftLeft":
            entity_name = "ShiftL"
        elif entity_name == "ShiftRight":
            entity_name = "ShiftR"
        elif entity_name == "TruncateMSBs":
            entity_name = "Trunc"
        elif entity_name == "Comparator":
            entity_name = "Comp"
        elif entity_name == "Mux":
            entity_name = "Multiplex"
        elif entity_name == "Mult":
            entity_name = "Multip"
        elif entity_name == "Add":
            entity_name = "Adder"
        elif entity_name == "Sub":
            entity_name = "Subtr"
        elif entity_name == "Constant":
            entity_name = "Consta"
        elif entity_name == "Input":
            entity_name = "Inp"
        elif entity_name == "Output":
            entity_name = "Outp"
        return entity_name

    @staticmethod
    def collect_unique_module_types(circuit: Circuit):
        unique_types = {}
        for module in circuit.modules:
            module_class = type(module)
            if module_class not in unique_types:
                unique_types[module_class] = module
        return unique_types.values()

    @staticmethod
    def write_module_definition(f, module: modules.Module):
        # Generate the entity and architecture for the module type
        module_entity = VHDLWriter.generate_module_entity(module)
        module_architecture = VHDLWriter.generate_module_architecture(module)

        # Write the entity and architecture to the file
        f.write(module_entity)
        f.write(module_architecture)

    @staticmethod
    def generate_module_entity(module: modules.Module):
        entity_name = type(module).__name__
        entity_name = VHDLWriter.entity_name_changer(entity_name)
        inputs = []
        outputs = []

        # Parameters (Generics)
        generics = []

        # Determine the inputs and outputs, and collect generics
        if isinstance(module, modules.Add) or isinstance(module, modules.Sub) or isinstance(module, modules.Mult):
            # Differentiate between input and output widths
            generics.append(f"    IN0_WIDTH : integer := 8")
            generics.append(f"    IN1_WIDTH : integer := 8")
            generics.append(f"    OUT_WIDTH : integer := 8")
            inputs.append((module.inputs[0], "IN0_WIDTH"))
            inputs.append((module.inputs[1], "IN1_WIDTH"))
            outputs.append((module.output, "OUT_WIDTH"))
        elif isinstance(module, modules.ShiftLeft) or isinstance(module, modules.ShiftRight):
            generics.append(f"    IN_WIDTH : integer := 8")
            generics.append(f"    OUT_WIDTH : integer := 8")
            generics.append(f"    SHIFT_AMOUNT : integer := {module.shift_length}")
            inputs.append((module.inputs[0], "IN_WIDTH"))
            outputs.append((module.output, "OUT_WIDTH"))
        elif isinstance(module, modules.Register):
            generics.append(f"    DATA_WIDTH : integer := 8")
            inputs.append((module.inputs[0], "DATA_WIDTH"))
            outputs.append((module.output, "DATA_WIDTH"))
            # Include clk in the port list
            inputs.append(("clk", ""))
        elif isinstance(module, modules.Constant):
            generics.append(f"    DATA_WIDTH : integer := {module.output_word_size()}")
            generics.append(f"    VALUE : integer := {module.value}")
            outputs.append((module.output, "DATA_WIDTH"))
        else:
            # For other modules, use DATA_WIDTH generic
            generics.append(f"    DATA_WIDTH : integer := 8")
            for idx, input_name in enumerate(module.inputs):
                inputs.append((input_name, "DATA_WIDTH"))
            outputs.append((module.output, "DATA_WIDTH"))
        
        entity_str = "library IEEE;\n"
        entity_str += "use IEEE.STD_LOGIC_1164.ALL;\n"
        entity_str += "use IEEE.NUMERIC_STD.ALL;\n\n"

        # Build the entity string
        entity_str += f"entity {entity_name} is\n"
        if generics:
            entity_str += "    generic (\n"
            entity_str += ";\n".join(generics)
            entity_str += "\n    );\n"
        entity_str += "    Port (\n"

        port_lines = []
        for name, width in inputs:
            if name == "clk":
                port_lines.append(f"        {name} : in std_logic")
            else:
                port_lines.append(f"        {name} : in std_logic_vector({width}-1 downto 0)")
        for name, width in outputs:
            port_lines.append(f"        {name} : out std_logic_vector({width}-1 downto 0)")

        entity_str += ";\n".join(port_lines)
        entity_str += "\n    );\n"
        entity_str += f"end {entity_name};\n\n"

        return entity_str

    @staticmethod
    def generate_module_architecture(module: modules.Module):
        entity_name = type(module).__name__
        entity_name = VHDLWriter.entity_name_changer(entity_name)
        architecture_str = f"architecture Behavioral of {entity_name} is\n"
        architecture_str += "begin\n"

        # Generate the behavior based on module type
        if isinstance(module, modules.Input):
            # Input module logic is handled externally
            architecture_str += "    -- Input module logic is handled externally\n"
        elif isinstance(module, modules.Output):
            # Output module logic is handled externally
            architecture_str += "    -- Output module logic is handled externally\n"
        elif isinstance(module, modules.Constant):
            architecture_str += f"    {module.output} <= std_logic_vector(to_signed(VALUE, DATA_WIDTH));\n"
        elif isinstance(module, modules.Add):
            op_str = VHDLWriter.generate_arithmetic_operation_generic(module, '+')
            architecture_str += f"    {module.output} <= {op_str};\n"
        elif isinstance(module, modules.Sub):
            op_str = VHDLWriter.generate_arithmetic_operation_generic(module, '-')
            architecture_str += f"    {module.output} <= {op_str};\n"
        elif isinstance(module, modules.ShiftLeft) or isinstance(module, modules.ShiftRight):
            op_str = VHDLWriter.generate_shift_operation_generic(module)
            architecture_str += f"    {module.output} <= {op_str};\n"
        elif isinstance(module, modules.Register):
            process_str = VHDLWriter.generate_register_operation_generic(module)
            architecture_str += process_str
        else:
            architecture_str += "    -- Unsupported module type\n"

        architecture_str += f"end Behavioral;\n\n"
        return architecture_str

    @staticmethod
    def generate_arithmetic_operation_generic(module: modules.Module, operator: str):
        x0 = module.inputs[0]
        x1 = module.inputs[1]
        op_str = f"std_logic_vector(resize(signed({x0}), OUT_WIDTH) {operator} resize(signed({x1}), OUT_WIDTH))"
        return op_str

    @staticmethod
    def generate_shift_operation_generic(module: modules.Module):
        input_name = module.inputs[0]
        shift_op = 'sll' if isinstance(module, modules.ShiftLeft) else 'srl'
        op_str = f"std_logic_vector(resize(signed({input_name}), OUT_WIDTH) {shift_op} SHIFT_AMOUNT)"
        return op_str

    @staticmethod
    def generate_register_operation_generic(module: modules.Register):
        input_name = module.inputs[0]
        output_name = module.output
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
        entity_str = "library IEEE;\n"
        entity_str += "use IEEE.STD_LOGIC_1164.ALL;\n"
        entity_str += "use IEEE.NUMERIC_STD.ALL;\n\n"
        entity_str += "entity TopModule is\n"
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

        # Add clock signal if necessary (e.g., for registers)
        has_register = any(isinstance(m, modules.Register) for m in circuit.modules)
        if has_register:
            port_lines.append("        clk : in std_logic")

        entity_str += ";\n".join(port_lines)
        entity_str += "\n    );\n"
        entity_str += "end TopModule;\n\n"

        f.write(entity_str)

        # Write the architecture of the top module
        architecture_str = "architecture Structural of TopModule is\n"

        # Signal declarations
        for module in circuit.modules:
            if not isinstance(module, modules.Output):
                width = module.output_word_size()
                architecture_str += f"    signal {module.name}_sig : std_logic_vector({width - 1} downto 0);\n"

        architecture_str += "\nbegin\n"

        # Map inputs to internal signals
        for module in circuit.modules:
            if isinstance(module, modules.Input):
                architecture_str += f"    {module.name}_sig <= {module.name};\n"

        # Instantiate modules and map ports
        for module in circuit.modules:
            if isinstance(module, modules.Input) or isinstance(module, modules.Output):
                continue  # Inputs and outputs are handled separately
            else:
                instance_name = module.name + "_inst"
                module_type_name = type(module).__name__
                module_type_name = VHDLWriter.entity_name_changer(module_type_name)
                architecture_str += f"\n    -- {module_type_name} instance\n"
                architecture_str += f"    {instance_name} : entity work.{module_type_name}\n"

                # Map generics
                generics = VHDLWriter.get_generics_map(module)
                if generics:
                    architecture_str += "        generic map (\n"
                    architecture_str += ",\n".join(f"            {k} => {v}" for k, v in generics.items())
                    architecture_str += "\n        )\n"

                architecture_str += "        port map (\n"
                if isinstance(module, modules.Register):
                    architecture_str += "            clk => clk,\n"

                # Map inputs
                port_mappings = []
                for idx, input_name in enumerate(module.inputs):
                    if input_name == "clk":
                        port_mappings.append(f"            {input_name} => clk")
                    else:
                        connected_module = module.input_modules[idx]
                        src_signal = VHDLWriter.get_signal_name(connected_module)
                        port_mappings.append(f"            {input_name} => {src_signal}")

                # Map outputs
                port_mappings.append(f"            {module.output} => {module.name}_sig")

                architecture_str += ",\n".join(port_mappings)
                architecture_str += "\n        );\n"

        # Connect outputs to top-level ports
        for module in circuit.modules:
            if isinstance(module, modules.Output):
                src_module = module.input_modules[0]
                src_signal = VHDLWriter.get_signal_name(src_module)
                architecture_str += f"    {module.name} <= {src_signal};\n"

        architecture_str += "\nend Structural;\n\n"

        f.write(architecture_str)

    @staticmethod
    def get_generics_map(module: modules.Module):
        generics = {}
        if isinstance(module, (modules.Add, modules.Sub, modules.Mult)):
            in0_width = module.input_modules[0].output_word_size()
            in1_width = module.input_modules[1].output_word_size()
            out_width = module.output_word_size()
            generics['IN0_WIDTH'] = in0_width
            generics['IN1_WIDTH'] = in1_width
            generics['OUT_WIDTH'] = out_width
        elif isinstance(module, (modules.ShiftLeft, modules.ShiftRight)):
            in_width = module.input_modules[0].output_word_size()
            out_width = module.output_word_size()
            shift_amount = module.shift_length
            generics['IN_WIDTH'] = in_width
            generics['OUT_WIDTH'] = out_width
            generics['SHIFT_AMOUNT'] = shift_amount
        elif isinstance(module, modules.Register):
            data_width = module.output_word_size()
            generics['DATA_WIDTH'] = data_width
        elif isinstance(module, modules.Constant):
            data_width = module.output_word_size()
            value = module.value
            generics['DATA_WIDTH'] = data_width
            generics['VALUE'] = value
        else:
            data_width = module.output_word_size()
            generics['DATA_WIDTH'] = data_width
        return generics

    @staticmethod
    def get_signal_name(module):
        if isinstance(module, modules.Input):
            return f"{module.name}_sig"
        elif isinstance(module, modules.Output):
            return module.name
        else:
            return f"{module.name}_sig"
