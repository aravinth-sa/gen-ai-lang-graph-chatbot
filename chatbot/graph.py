"""
Conversation flow graph definition for the chatbot.

This module defines the conversation flow graph structure using LangGraph.
It connects various nodes (states) and defines the routing logic between them.

To visualize this graph, use the graph_visualizer.py module.
"""

from typing import Any, Dict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from chatbot.states import AgentState, AgentInput, question_rewriter, question_classifier, off_topic_response, retrieve, retrieval_grader, generate_answer, refine_question, cannot_answer, on_topic_router, proceed_router, category_detector, category_detector_router, category_router, category_slot_filler, slot_filler_router

class GraphConfig:
    """Configuration class for creating the conversation flow graph.
    
    This class is responsible for defining and creating the conversation flow graph
    structure for the chatbot. It connects various nodes (states) and defines the
    routing logic between them.
    """
    
    def create_graph(self, config: Dict[str, Any] = None) -> StateGraph:
        """Create and return the conversation flow graph.
        
        Args:
            config: Optional configuration dictionary for the graph.
            
        Returns:
            A compiled StateGraph instance representing the conversation flow.
        """
        # Set default config if none provided
        if config is None:
            config = {}
            
        # Get recursion limit from config or use default
        recursion_limit = config.get("recursion_limit", 10)
            
        # Configure the memory saver with more robust persistence settings
        checkpointer = MemorySaver()

        # Workflow - use AgentInput as the input type and AgentState as the state type
        workflow = StateGraph(AgentState, AgentInput)
        #workflow.add_node("question_rewriter", question_rewriter)
        workflow.add_node("question_classifier", question_classifier)
        workflow.add_node("off_topic_response", off_topic_response)
        # workflow.add_node("category_detector", category_detector)
        # workflow.add_node("category_slot_filler", category_slot_filler)
        # workflow.add_node("slot_filler_router", slot_filler_router)
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("retrieval_grader", retrieval_grader)
        workflow.add_node("generate_answer", generate_answer)
        workflow.add_node("refine_question", refine_question)
        workflow.add_node("cannot_answer", cannot_answer)

        #workflow.add_edge("question_rewriter", "question_classifier")
        workflow.add_conditional_edges(
            "question_classifier",
            on_topic_router,
            {
                "retrieve": "retrieve",
                "off_topic_response": "off_topic_response",
            },
        )
        
        # Add edge from category_detector to category_router via category_detector_router
        # workflow.add_conditional_edges(
        #     "category_detector",
        #     category_router,
        #     {
        #         "retrieve": "retrieve",
        #         "fill_slots": "category_slot_filler",
        #     },
        # )
        workflow.add_edge("retrieve", "retrieval_grader")
        workflow.add_conditional_edges(
            "retrieval_grader",
            proceed_router,
            {
                "generate_answer": "generate_answer",
                "refine_question": "refine_question",
                "cannot_answer": "cannot_answer",
            },
        )
        workflow.add_edge("refine_question", "retrieve")
        # workflow.add_conditional_edges(
        #     "category_slot_filler",
        #     slot_filler_router,
        #     {
        #         "retrieve": "retrieve",
        #         END: END
        #     }
        # )
        workflow.add_edge("generate_answer", END)
        workflow.add_edge("cannot_answer", END)
        workflow.add_edge("off_topic_response", END)
        workflow.set_entry_point("question_classifier")
        graph = workflow.compile(checkpointer=checkpointer) 
        return graph

if __name__ == "__main__":
    from IPython.display import Image, display
    from langchain_core.runnables.graph import MermaidDrawMethod

    # Create an instance of GraphConfig
    graph_config = GraphConfig()
    
    # Create the graph
    graph = graph_config.create_graph()
    
    # Display the graph
    display(
            Image(
                graph.get_graph().draw_mermaid_png(
                    draw_method=MermaidDrawMethod.API,
                )
            )
    )