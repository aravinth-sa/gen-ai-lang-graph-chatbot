"""
Graph visualization utility for the chatbot conversation flow.
This module provides functions to visualize the conversation flow graph structure.
"""

import json
import os
from typing import Dict, Any

# Import visualization libraries if available
try:
    import networkx as nx
    import matplotlib.pyplot as plt
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False
    print("Note: Install networkx and matplotlib for graph visualization")

def print_graph_structure(workflow):
    """Print a text representation of the graph structure to the console."""
    print("\n=== Conversation Flow Graph Structure ===\n")
    print("Nodes:")
    for node_name in workflow.nodes:
        print(f"  - {node_name}")
    
    print("\nEdges:")
    for source, target_dict in workflow._adjacency_map.items():
        if source == "__start__":
            print(f"  Entry Point → {workflow._entrypoint}")
        else:
            for target, condition in target_dict.items():
                if target == "__end__":
                    print(f"  {source} → END")
                else:
                    if condition is None:
                        print(f"  {source} → {target}")
                    else:
                        print(f"  {source} → {target} [conditional]")
    
    print("\n=== Conversation Flow Visualization ===\n")
    print("question_classifier → on_topic_router")
    print("  ├── [Yes] → category_detector → category_router")
    print("  │     ├── [Category detected] → retrieve → retrieval_grader → proceed_router")
    print("  │     │     ├── [Documents found] → generate_answer → END")
    print("  │     │     ├── [No documents, retry < max] → refine_question → retrieve")
    print("  │     │     └── [No documents, retry ≥ max] → cannot_answer → END")
    print("  │     └── [No category detected] → category_slot_filler → END")
    print("  └── [No] → off_topic_response → END")
    print("\n==================================\n")

def export_graph_structure(workflow):
    """Export the graph structure to a JSON file."""
    graph_structure = {
        "nodes": list(workflow.nodes),
        "edges": [],
        "entry_point": workflow._entrypoint
    }
    
    for source, target_dict in workflow._adjacency_map.items():
        if source != "__start__":
            for target, condition in target_dict.items():
                if target != "__end__":
                    edge = {"source": source, "target": target}
                    if condition is not None:
                        edge["conditional"] = True
                    graph_structure["edges"].append(edge)
                else:
                    graph_structure["edges"].append({"source": source, "target": "END"})
    
    # Save to file
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, 'graph_structure.json')
        with open(output_path, 'w') as f:
            json.dump(graph_structure, f, indent=2)
        print(f"Graph structure saved to {output_path}")
        return graph_structure
    except Exception as e:
        print(f"Error saving graph structure: {e}")
        return None

def visualize_graph(graph_structure=None, workflow=None):
    """Generate a visual representation of the graph using NetworkX and Matplotlib."""
    if not HAS_VISUALIZATION:
        print("NetworkX and/or Matplotlib not available. Cannot generate visualization.")
        return
    
    # If graph_structure not provided, generate it from workflow
    if graph_structure is None and workflow is not None:
        graph_structure = export_graph_structure(workflow)
    
    if not graph_structure:
        print("No graph structure available for visualization.")
        return
    
    try:
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add nodes
        for node in graph_structure["nodes"]:
            G.add_node(node)
        G.add_node("END")
        
        # Add edges
        for edge in graph_structure["edges"]:
            G.add_edge(edge["source"], edge["target"])
        
        # Set up the plot with a larger figure size
        plt.figure(figsize=(12, 10))
        
        # Create a layout for the nodes
        pos = nx.spring_layout(G, seed=42, k=0.8)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=2000, node_color="lightblue", alpha=0.8)
        
        # Highlight special nodes
        special_nodes = {"question_classifier": "lightgreen", "END": "lightcoral", "category_slot_filler": "lightyellow"}
        for node, color in special_nodes.items():
            if node in G.nodes():
                nx.draw_networkx_nodes(G, pos, nodelist=[node], node_size=2000, node_color=color, alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, width=1.5, arrowsize=20)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
        
        # Save the figure
        plt.title("Conversation Flow Graph", size=15)
        plt.axis("off")
        plt.tight_layout()
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        graph_image_path = os.path.join(script_dir, 'graph_visualization.png')
        plt.savefig(graph_image_path)
        print(f"Graph visualization saved to {graph_image_path}")
        plt.close()
        
    except Exception as e:
        print(f"Error generating graph visualization: {e}")

def visualize_conversation_flow(workflow=None):
    """Main function to visualize the conversation flow graph."""
    if workflow is None:
        print("No workflow provided. Cannot visualize graph.")
        return
    
    # Print text representation
    print_graph_structure(workflow)
    
    # Export graph structure to JSON
    graph_structure = export_graph_structure(workflow)
    
    # Generate visual graph
    visualize_graph(graph_structure)
