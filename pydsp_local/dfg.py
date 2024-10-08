from enum import Enum
try:
    import modules
except:
    import pydsp_local.modules as modules


class OperationType(Enum):
    INPUT=0
    OUTPUT=1
    CONSTANT=2
    ADD=3
    SUB=4
    MULT=5
    COMPARATOR=6
    MUX=7
    SHIFT_LEFT=8
    SHIFT_RIGHT=9
    TRUNCATE_MSB=10


OPERATION_MAP = {
    OperationType.INPUT: modules.Input,
    OperationType.OUTPUT: modules.Output,
    OperationType.CONSTANT: modules.Constant,
    OperationType.ADD: modules.Add,
    OperationType.SUB: modules.Sub,
    OperationType.MULT: modules.Mult,
    OperationType.COMPARATOR: modules.Comparator,
    OperationType.MUX: modules.Mux,
    OperationType.SHIFT_LEFT: modules.ShiftLeft,
    OperationType.SHIFT_RIGHT: modules.ShiftRight,
    OperationType.TRUNCATE_MSB: modules.TruncateMSBs
}


class Vertex:
    def __init__(self, name, operation_type, operation_parameters) -> None:
        self.name = name
        self.operation_type = operation_type
        self.operation_parameters = operation_parameters


class Edge:
    def __init__(self, src_vertex, dst_vertex, dst_port, distance) -> None:
        self.src_vertex = src_vertex
        self.dst_vertex = dst_vertex
        self.dst_port = dst_port
        self.distance = distance


class Graph:
    def __init__(self, name="Graph") -> None:
        self.name = name
        self.vertices = []
        self.edges = []
    
    def get_incoming_edges(self, v):
        incoming_edges = []
        for e in self.edges:
            if e.dst_vertex != v:
                continue
            incoming_edges.append(e)
        return incoming_edges

    def number_of_inputs(self, v):
        return len(self.get_incoming_edges(v))
    
    def add_vertex(self, name, operation_type=None, operation_parameters=None):
        if any(v.name == name for v in self.vertices):
            raise Exception(f"Cannot create multiple vertices with name '{name}' -> names must be unique")
        self.vertices.append(Vertex(name, operation_type, operation_parameters))
        return self.vertices[-1]
    
    def add_edge(self, src_vertex, dst_vertex, dst_port, distance):
        if any([e.dst_vertex == dst_vertex and e.dst_port == dst_port for e in self.edges]):
            raise Exception(f"Cannot connect multiple edges to the same dst port '{dst_port}' of vertex '{dst_vertex}'")
        self.edges.append(Edge(src_vertex, dst_vertex, dst_port, distance))
        return self.edges[-1]

    def print_info(self):
        print(f"{self.name}:")
        for v in self.vertices:
            param_str = "" if v.operation_parameters is None else f" with parameter(s): {v.operation_parameters}"
            operation_type_str = "" if v.operation_type is None else f" ({v.operation_type})"
            print(f"-> vertex '{v.name}'{operation_type_str}{param_str}")
        for e in self.edges:
            print(f"-> edge '{e.src_vertex.name}' --> '{e.dst_vertex.name}' (port {e.dst_port}) with distance={e.distance}")