from dfg import *
from file_writers import *


def main():
    dfg = Graph("Example graph")
    v0 = dfg.add_vertex(name="v0")
    v1 = dfg.add_vertex(name="v1")
    v2 = dfg.add_vertex(name="v2")
    v3 = dfg.add_vertex(name="v3")
    dfg.add_edge(src_vertex=v0, dst_vertex=v1, dst_port=0, distance=0)
    dfg.add_edge(src_vertex=v1, dst_vertex=v2, dst_port=0, distance=2)
    dfg.add_edge(src_vertex=v2, dst_vertex=v0, dst_port=0, distance=0)
    dfg.add_edge(src_vertex=v2, dst_vertex=v3, dst_port=0, distance=0)
    dfg.add_edge(src_vertex=v3, dst_vertex=v1, dst_port=1, distance=1)
    dfg.add_edge(src_vertex=v0, dst_vertex=v3, dst_port=1, distance=0)
    dfg.add_edge(src_vertex=v0, dst_vertex=v2, dst_port=1, distance=1)


if __name__ == '__main__':
    main()
