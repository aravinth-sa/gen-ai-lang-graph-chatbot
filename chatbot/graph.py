
from typing import Any, Dict, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from chatbot.states import AgentState, question_rewriter, question_classifier, off_topic_response, retrieve, retrieval_grader,generate_answer, refine_question, cannot_answer, on_topic_router, proceed_router 

class GraphConfig(TypedDict):

    def create_graph(config: Dict[str, Any]) -> StateGraph:
        checkpointer = MemorySaver()

        # Workflow
        workflow = StateGraph(AgentState)
        workflow.add_node("question_rewriter", question_rewriter)
        workflow.add_node("question_classifier", question_classifier)
        workflow.add_node("off_topic_response", off_topic_response)
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("retrieval_grader", retrieval_grader)
        workflow.add_node("generate_answer", generate_answer)
        workflow.add_node("refine_question", refine_question)
        workflow.add_node("cannot_answer", cannot_answer)

        workflow.add_edge("question_rewriter", "question_classifier")
        workflow.add_conditional_edges(
            "question_classifier",
            on_topic_router,
            {
                "retrieve": "retrieve",
                "off_topic_response": "off_topic_response",
            },
        )
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
        workflow.add_edge("generate_answer", END)
        workflow.add_edge("cannot_answer", END)
        workflow.add_edge("off_topic_response", END)
        workflow.set_entry_point("question_rewriter")
        graph = workflow.compile(checkpointer=checkpointer) 
        return graph
