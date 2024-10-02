from dfg import *
from file_writers import GraphvizWriter
from circuit import Circuit
from modules import *


def example_1():
    g = Graph("Example graph")
    i0 = g.add_vertex(name="i0", operation_type=OperationType.INPUT)
    i1 = g.add_vertex(name="i1", operation_type=OperationType.INPUT)
    a0 = g.add_vertex(name="a0", operation_type=OperationType.ADD)
    o0 = g.add_vertex(name="o0", operation_type=OperationType.OUTPUT)
    g.add_edge(src_vertex=i0, dst_vertex=a0, dst_port=0, distance=0)
    g.add_edge(src_vertex=i1, dst_vertex=a0, dst_port=1, distance=2)
    g.add_edge(src_vertex=a0, dst_vertex=o0, dst_port=0, distance=0)
    g.print_info()
    GraphvizWriter.write("example.dot", g)


def example_2():
    # build graph
    g = Graph("Example graph")
    i0 = g.add_vertex(name="i0", operation_type=OperationType.INPUT)
    i1 = g.add_vertex(name="i1", operation_type=OperationType.INPUT)
    a0 = g.add_vertex(name="a0", operation_type=OperationType.ADD)
    o0 = g.add_vertex(name="o0", operation_type=OperationType.OUTPUT)
    g.add_edge(src_vertex=i0, dst_vertex=a0, dst_port=0, distance=0)
    g.add_edge(src_vertex=i1, dst_vertex=a0, dst_port=1, distance=2)
    g.add_edge(src_vertex=a0, dst_vertex=o0, dst_port=0, distance=0)
    g.print_info()
    # implement the corresponding circuit 
    dat = DataType.SIGNED
    word_size = 8
    i_min = -(2**(word_size-1))
    i_max = -i_min-1
    c = Circuit()
    instantiated = {v: False for v in g.vertices}
    module_dict = {}
    # instantiate vertices until we are done
    # -> the following way only works for graphs without cycles
    while not all(instantiated[v] for v in g.vertices):
        for v_dst in g.vertices:
            if instantiated[v_dst]:
                continue
            src_edges = g.get_incoming_edges(v_dst)
            if not all(instantiated[e.src_vertex] for e in src_edges):
                continue
            instantiated[v_dst] = True
            module_type = OPERATION_MAP[v_dst.operation_type]
            print(f"module_type = {module_type}")
            if module_type is Input:
                input_ranges = [[i_min, i_max]]
            else:
                src_edges.sort(key=lambda x: x.dst_port)
                input_ranges = []
                for e in src_edges:
                    src_module = module_dict[e.src_vertex]
                    input_ranges.append(src_module.output_range())
            dst_module = module_type(input_ranges=input_ranges, data_type=dat, name=v_dst.name)
            module_dict[v_dst] = c.add(dst_module)
            for e in src_edges:
                src_module = module_dict[e.src_vertex]
                c.connect(src_module=src_module, dst_module=dst_module, dst_port=e.dst_port)

    c.print_info()
    # export graphviz files for both
    GraphvizWriter.write("example_g.dot", g)
    GraphvizWriter.write("example_c.dot", c)


def main():
    example_2()


if __name__ == '__main__':
    main()