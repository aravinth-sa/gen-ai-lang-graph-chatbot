"""
Additional nodes not currently used in the conversation flow graph.

This module contains node functions that are not actively used in graph.py.
These nodes may be used for alternative flows, testing, or future features.
"""

import json
import os
from typing import List, Dict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END

from .states import (
    AgentState, CategoryDetection, get_timestamp, get_default_llm,
    get_grader_llm, get_creative_llm
)


# ==================== QUESTION REWRITER NODE (NOT USED) ====================

def question_rewriter(state: AgentState):
    print(f"[{get_timestamp()}] Entering question_rewriter with following state: {state}")

    # Initialize or use provided conversation history
    if "conversation_history" in state.get("question", {}):
        # If conversation history is provided in the input, use it
        print(f"[{get_timestamp()}] Using provided conversation history in question_rewriter")
        state["conversation_history"] = state["question"].get("conversation_history", [])
    elif "conversation_history" not in state or state["conversation_history"] is None:
        print(f"[{get_timestamp()}] Initializing empty conversation history in question_rewriter")
        state["conversation_history"] = []
    else:
        print(f"[{get_timestamp()}] Using existing conversation history with {len(state['conversation_history'])} messages in question_rewriter")

    # Initialize state variables if they don't exist
    if "documents" not in state:
        state["documents"] = []
    if "relavant_documents" not in state:
        state["relavant_documents"] = []
    if "on_topic" not in state:
        state["on_topic"] = ""
    if "rephrased_question" not in state:
        state["rephrased_question"] = ""
    if "proceed_to_generate" not in state:
        state["proceed_to_generate"] = False
    if "rephrase_count" not in state:
        state["rephrase_count"] = 0
        
    # Add the current question to conversation history if it's not already there
    current_question = state["question"].content
    if not any(msg.get("content") == current_question and msg.get("role") == "user" 
              for msg in state["conversation_history"][-3:]):
        state["conversation_history"].append({
            "role": "user",
            "content": current_question
        })

    if "messages" not in state or state["messages"] is None:
        state["messages"] = []

    # Add current question to messages if not already present
    if state["question"] not in state["messages"]:
        state["messages"].append(state["question"])

    # Prepare conversation context for rephrasing
    current_question = state["question"].content
    
    # Add system message and conversation history
    messages = [
        SystemMessage(
            content="You are a helpful assistant that rephrases the user's question to be a standalone question optimized for retrieval. "
                   "Consider the conversation history when rephrasing to maintain context."
        )
    ]
    
    # Add conversation history if available
    for msg in state["conversation_history"]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    
    # Add current question
    messages.append(HumanMessage(content=current_question))
    
    # Generate rephrased question using conversation context
    rephrase_prompt = ChatPromptTemplate.from_messages(messages)
    llm = get_default_llm()
    prompt = rephrase_prompt.format()
    response = llm.invoke(prompt)
    better_question = response.content.strip()
    print(f"question_rewriter: Rephrased question: {better_question}")
    state["rephrased_question"] = better_question
    
    return state


# ==================== OFF-TOPIC RESPONSE NODE (NOT USED) ====================

def off_topic_response(state: AgentState):
    print(f"[{get_timestamp()}] Entering off_topic_response")
    print("Entering off_topic_response")
    if "messages" not in state or state["messages"] is None:
        state["messages"] = []
    
    # Initialize a chat model with higher temperature for more creative responses
    chat = get_creative_llm()
    
    # Generate a creative response
    response = chat.invoke([
        SystemMessage(content="You are a creative assistant that generates fun, engaging responses when asked off-topic questions."),
        HumanMessage(content="I was asked this question but it's off-topic. Please generate a creative, fun response that gently guides the conversation back to the topic of Building Materials & Hardware. The original question was: " + state["question"].content)
    ])
    
    # Add the AI's response to the messages
    state["messages"].append(AIMessage(content=response.content))
    return state


# ==================== CATEGORY DETECTION NODES (NOT USED) ====================

# Load the slots-category.json file
def load_category_slots():
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(script_dir, 'dataset', 'slots', 'slots-category.json')
    with open(file_path, 'r') as f:
        return json.load(f)


def category_detector(state: AgentState):
    """Detect the product category from the user query"""
    print(f"[{get_timestamp()}] Entering category_detector")
    # Initialize category-related fields if they don't exist
    if "detected_category" not in state:
        state["detected_category"] = ""
    if "category_slot" not in state:
        state["category_slot"] = ""
    
    print(f"DEBUG: detected_category='{state['detected_category']}', category_slot='{state['category_slot']}'")
    # Load the category slots
    category_data = load_category_slots()
    categories = list(category_data["categories"].keys())
    categories_str = ", ".join(categories)
    
    # Prepare the system message with category information
    system_message = SystemMessage(
        content=f"""You are a product category detector for a building materials supplier.
        
        Your task is to identify which product category the user is asking about from the following list:
        {categories_str}
        
        Analyze the user's question and determine the most likely product category they are referring to.
        If multiple categories could apply, choose the most specific one.
        If no category clearly applies, respond with 'unknown'.
        
        Also provide a confidence level ('high', 'medium', or 'low') for your detection.
        - 'high': The category is explicitly mentioned or strongly implied
        - 'medium': The category is reasonably implied but not explicit
        - 'low': The category is a best guess but uncertain
        
        Return only the category name and confidence level."""
    )
    
    # Get the current question
    current_question = state["rephrased_question"]
    
    # Add conversation history context if available
    messages = [system_message]
    if "conversation_history" in state and state["conversation_history"]:
        # Add the last few exchanges for context
        context_str = "\nConversation history:\n"
        for i, msg in enumerate(state["conversation_history"][-3:]):
            role = "User" if msg["role"] == "user" else "Assistant"
            context_str += f"{role}: {msg['content']}\n"
        messages.append(SystemMessage(content=context_str))
    
    # Add the current question
    messages.append(HumanMessage(content=current_question))
    
    # Create the prompt
    category_prompt = ChatPromptTemplate.from_messages(messages)
    llm = get_default_llm()
    structured_llm = llm.with_structured_output(CategoryDetection)
    detector_llm = category_prompt | structured_llm
    
    # Detect the category
    try:
        result = detector_llm.invoke({})
        detected_category = result.category.strip().lower()
        confidence = result.confidence.strip().lower()
        
        # Validate the detected category
        if detected_category in categories:
            state["detected_category"] = detected_category
            print(f"category_detector: Detected category '{detected_category}' with {confidence} confidence")
        elif detected_category != "unknown":
            # Try to find the closest match
            print(f"category_detector: Category '{detected_category}' not found in available categories, setting to unknown")
            state["detected_category"] = "unknown"
        else:
            state["detected_category"] = "unknown"
            print("category_detector: No category detected")
            
    except Exception as e:
        print(f"Error in category detection: {str(e)}")
        state["detected_category"] = "unknown"
    
    return state


def category_detector_router(state: AgentState):
    """Route from category detector to either category_router or category_slot_filler"""
    print(f"[{get_timestamp()}] Entering category_detector_router")
    
    # Always route to category_router node after detection
    return "category_router"


def category_router(state: AgentState):
    """Route based on whether a category was detected"""
    print(f"[{get_timestamp()}] Entering category_router")
    
    detected_category = state.get("detected_category", "").strip().lower()
    
    if detected_category and detected_category != "unknown":
        print(f"category_router: Category '{detected_category}' detected, asking user to fill the slot")
        return "fill_slots"
    else:
        print("category_router: No category detected, proceeding to retrieve")
        return "retrieve"


def category_slot_filler(state: AgentState):
    """Ask the user to specify a product category"""
    print(f"[{get_timestamp()}] Entering category_slot_filler")
    
    if "messages" not in state or state["messages"] is None:
        state["messages"] = []
    
    # Load the category slots
    category_data = load_category_slots()
    categories = list(category_data["categories"].keys())
    
    # Group categories for better readability
    grouped_categories = []
    for i in range(0, len(categories), 5):  # Group by 5
        group = categories[i:i+5]
        grouped_categories.append(", ".join(group))
    
    categories_display = "\n".join([f"• {group}" for group in grouped_categories])
    
    # Create a message asking the user to specify a category
    response_content = f"""To help you better, I need to know which product category you're interested in. 
    
Please specify one of the following categories:

{categories_display}

Once you select a category, I can provide specific information about products, specifications, and options available."""
    
    # Add the message to state
    state["messages"].append(AIMessage(content=response_content))
    
    # Set a flag to indicate we're waiting for category input
    state["category_slot"] = "pending"
    
    return state


def slot_filler_router(state: AgentState):
    """Route based on whether a category slot has been filled"""
    print(f"[{get_timestamp()}] Entering slot_filler_router")
    
    # Get the user's input
    user_input = state["question"].content.lower()
    
    # Load categories
    category_data = load_category_slots()
    categories = list(category_data["categories"].keys())
    
    # Check if input matches any category
    for category in categories:
        if category.lower() in user_input:
            state["detected_category"] = category
            state["category_slot"] = "filled"
            print(f"slot_filler_router: Category slot filled with '{category}'")
            
            # Add confirmation message
            confirmation = f"Great! I'll provide information about {category}. Let me search for relevant details..."
            state["messages"].append(AIMessage(content=confirmation))
            
            return "retrieve"
    
    # If no category match is found, ask again but keep the slot pending
    print("slot_filler_router: No category match found, asking again")
    
    # Keep the category_slot as 'pending'
    state["category_slot"] = "pending"
    
    # Add a message asking the user to specify a category again
    response_content = "I'm sorry, I couldn't identify a specific product category from your response. Could you please specify one of the categories listed above?"
    state["messages"].append(AIMessage(content=response_content))
    
    # Return END to finish this conversation turn, but the next question will start from slot_filler_router
    return END


# ==================== BACKWARD COMPATIBILITY ALIASES ====================

# Keep old function names for backward compatibility
question_classifier = None  # This was an alias for intent_classifier, which is now in nodes.py
on_topic_router = None  # This was an alias for intent_router, which is now in nodes.py
