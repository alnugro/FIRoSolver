from graphviz import Source

# Load the .dot file and render it
dot_path = 'example.dot'
graph = Source.from_file(dot_path)
graph.view()  # This will open the rendered graph in the default viewer
