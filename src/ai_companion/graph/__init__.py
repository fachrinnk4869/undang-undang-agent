from ai_companion.graph.graph import create_workflow_graph

graph_builder = create_workflow_graph()

if __name__ == "__main__":
    # This is just to ensure the graph is built correctly
    print("Graph created successfully with nodes and edges:")
    for node in graph_builder.nodes:
        print(f"Node: {node}")
    for edge in graph_builder.edges:
        print(f"Edge: {edge}")
    # You can also compile the graph if needed
    graph = graph_builder.compile()
    print("Graph compiled successfully.")
