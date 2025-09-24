"""
Script to visualize the conversation flow graph.

This script generates both a text representation and a visual diagram of the conversation flow graph.
It helps developers understand the structure and flow of the chatbot conversation.

Usage:
    python visualize_graph.py [--text-only]

Options:
    --text-only    Generate only the text representation (no visual diagram)
"""

import sys
import os
from chatbot.graph import GraphConfig
from chatbot.graph_visualizer import print_graph_structure, export_graph_structure, visualize_graph

def main():
    # Parse command line arguments
    text_only = "--text-only" in sys.argv
    
    print("\n📊 Generating conversation flow graph visualization...\n")
    
    # Create the graph
    graph_config = GraphConfig()
    workflow = graph_config.create_graph({})
    
    # Print text representation
    print_graph_structure(workflow)
    
    # Export graph structure to JSON
    graph_structure = export_graph_structure(workflow)
    
    # Generate visual graph if not text-only mode
    if not text_only:
        try:
            visualize_graph(graph_structure)
            print("✅ Visual graph generated successfully!")
        except Exception as e:
            print(f"❌ Error generating visual graph: {e}")
            print("   Make sure you have networkx and matplotlib installed:")
            print("   pip install networkx matplotlib")
    
    print("\n📋 Graph visualization complete!\n")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    chatbot_dir = os.path.join(script_dir, 'chatbot')
    
    print("📁 Check the following files:")
    print(f"   - {os.path.join(chatbot_dir, 'graph_structure.json')}")
    if not text_only:
        print(f"   - {os.path.join(chatbot_dir, 'graph_visualization.png')}\n")

if __name__ == "__main__":
    main()
