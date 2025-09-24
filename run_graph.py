"""
Script to run the chatbot graph.

This script imports and runs the graph from the chatbot module.
"""

# Import the graph module
from chatbot.graph import GraphConfig

if __name__ == "__main__":
    # Create an instance of GraphConfig
    graph_config = GraphConfig()
    
    # Create the graph
    graph = graph_config.create_graph()
    
    # Print a success message
    print("Graph created successfully!")
    
    # Save the graph visualization as a PNG file
    from langchain_core.runnables.graph import MermaidDrawMethod
    import os
    
    # Generate the graph image
    graph_image = graph.get_graph().draw_mermaid_png(
        draw_method=MermaidDrawMethod.API,
    )
    
    # Save the image to a file
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'graph_visualization.png')
    with open(output_path, 'wb') as f:
        f.write(graph_image)
    
    print(f"Graph visualization saved to: {output_path}")
    print("You can open this file to view the graph.")
    
    # Try to open the image automatically
    try:
        import subprocess
        subprocess.run(['start', output_path], shell=True, check=True)
        print("Opening the graph visualization...")
    except Exception as e:
        print(f"Could not automatically open the image: {e}")
        print("Please open the file manually.")
    
