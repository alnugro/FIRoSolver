try:
    import modules
except:
    import pydsp_local.modules




class Connection:
    def __init__(self, src_module, dst_module, dst_port) -> None:
        self.src_module = src_module
        self.dst_module = dst_module
        self.dst_port = dst_port


class Circuit:
    def __init__(self) -> None:
        self.modules = []
        self.connections = []
        self.input_data = {}
    
    def get_invalid_modules(self):
        invalid_modules = []
        for m in self.modules:
            valid, reason = m.is_valid()
            if valid:
                continue
            invalid_modules.append((m, reason))
        return invalid_modules

    def is_valid(self):
        invalid_modules = self.get_invalid_modules()
        return len(invalid_modules) == 0

    def add(self, module):
        if any(m.name == module.name for m in self.modules):
            raise Exception(f"Module names must be unique, cannot have two modules with name '{module.name}'!")
        self.modules.append(module)
        return self.modules[-1]
    
    def connect(self, src_module, dst_module, dst_port):
        if (any(c.dst_module.name == dst_module.name and c.dst_port == dst_port for c in self.connections)):
            raise Exception(f"Cannot connect two signals to module '{dst_module.name}' port '{dst_port}'!")
        self.connections.append(Connection(src_module=src_module, dst_module=dst_module, dst_port=dst_port))
        dst_module.input_modules[dst_port] = src_module
    
    def print_info(self):
        for module in self.modules:
            print(f"module '{module.name}' output range: {module.output_range()}, output word size: {module.output_word_size()}")
        for connection in self.connections:
            print(f"connection '{connection.src_module.name}' -> '{connection.dst_module.name}' port {connection.dst_port}")
    
    def validate(self):
        if not self.is_valid():
            invalid_modules = self.get_invalid_modules()
            for (m, reason) in invalid_modules:
                print(f"-> module '{m.name}' failed verification (reason: '{reason}')")
            raise Exception(f"Constructed invalid circuit :(")
