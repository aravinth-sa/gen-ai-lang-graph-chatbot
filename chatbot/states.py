"""
State definitions and helper functions for the chatbot.

This module defines the state structures (TypedDict classes) and Pydantic models
used throughout the chatbot system. It also contains LLM configuration functions
and utility helpers.

All node functions have been moved to:
- nodes.py: Active nodes used in the graph
- additional_nodes.py: Unused nodes for future use
"""

from typing import TypedDict, List, Dict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from config import Config
from datetime import datetime


# ==================== CENTRALIZED LLM CONFIGURATION ====================
# All LLM instances are created here for easy model switching
#
# HOW TO CHANGE MODELS:
# 1. To change the default model for all operations, modify the model parameter in get_default_llm()
# 2. To change the grading model, modify the model parameter in get_grader_llm()
# 3. To change the creative response model, modify the model parameter in get_creative_llm()
# 4. To add a new LLM configuration, create a new function following the same pattern
#
# AVAILABLE MODELS (OpenAI):
# - gpt-4o-mini: Fast, cost-effective, good for most tasks
# - gpt-4o: More capable, higher cost
# - gpt-3.5-turbo: Fastest, cheapest, good for simple tasks
# - gpt-4: Most capable, highest cost
#
# AVAILABLE MODELS (Google Gemini):
# - gemini-2.5-flash: Latest Flash model, fast and capable (currently used)
# - gemini-2.0-flash-exp: Experimental Flash model
# - gemini-1.5-pro: Pro model with enhanced capabilities
# Note: Using gemini-2.5-flash for optimal performance
#
# To switch between providers:
# 1. Update the functions below to use get_gemini_llm() instead of get_llm()
# 2. Or modify get_llm() to return ChatGoogleGenerativeAI instead of ChatOpenAI
# ============================================================================

# ==================== OpenAI LLM Functions ====================
def get_llm(model: str = "gpt-3.5-turbo", temperature: float = 0.0):
    """Factory function to create OpenAI LLM instances with consistent configuration.
    
    Args:
        model: The OpenAI model name to use (default: "gpt-4o-mini")
        temperature: The temperature setting (default: 0.0 for deterministic output)
    
    Returns:
        ChatOpenAI instance configured with the specified parameters
    """
    return ChatOpenAI(model=model, temperature=temperature)

def get_default_llm():
    """Get the default LLM for most tasks (Now using Gemini 2.5 Flash)"""
    return get_gemini_llm()

def get_grader_llm():
    """Get LLM for document grading (Now using Gemini 2.5 Flash)"""
    return get_gemini_llm()

def get_creative_llm():
    """Get LLM for creative responses (Now using Gemini 2.5 Flash)"""
    return get_gemini_llm()

# ==================== Google Gemini LLM Functions ====================
def get_gemini_llm(model: str = "gemini-2.5-flash", temperature: float = 0.0):
    """Factory function to create Google Gemini LLM instances with consistent configuration.
    
    Args:
        model: The Gemini model name to use (default: "gemini-2.5-flash")
        temperature: The temperature setting (default: 0.0 for deterministic output)
    
    Returns:
        ChatGoogleGenerativeAI instance configured with the specified parameters
    """
    return ChatGoogleGenerativeAI(
        model=model, 
        temperature=temperature,
        google_api_key=Config.GOOGLE_API_KEY,
        convert_system_message_to_human=True
    )

def get_gemini_default_llm():
    """Get the default Gemini LLM for most tasks"""
    return get_gemini_llm(model="gemini-2.5-flash", temperature=0.0)

def get_gemini_grader_llm():
    """Get Gemini LLM for document grading (fast and cost-effective)"""
    return get_gemini_llm(model="gemini-2.5-flash", temperature=0.0)

def get_gemini_creative_llm():
    """Get Gemini LLM for creative responses (higher temperature)"""
    return get_gemini_llm(model="gemini-2.5-flash", temperature=0.7)

def get_gemini_pro_llm():
    """Get Gemini Pro LLM for more complex tasks"""
    return get_gemini_llm(model="gemini-2.5-flash", temperature=0.0)

# ==================== END LLM CONFIGURATION ====================


# ==================== UTILITY FUNCTIONS ====================

def get_timestamp():
    """Get current timestamp as formatted string"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ==================== STATE DEFINITIONS ====================

class AgentState(TypedDict):
    """Main state structure for the chatbot agent"""
    messages: List[BaseMessage]
    documents: List[Document]
    relavant_documents: List[Document]
    on_topic: str
    rephrased_question: str
    proceed_to_generate: bool
    rephrase_count: int
    question: HumanMessage
    conversation_history: List[Dict[str, str]]  # To store conversation context
    context_summary: str  # To store summarized context from conversation as plain text
    category_slot: str  # To store the product category slot
    detected_category: str  # To store the detected product category
    intent_type: str  # To store the intent type: 'project', 'product', 'stock', etc.
    project_stages: List[Dict[str, str]]  # To store project stages with descriptions
    stage_products: Dict[str, List[Document]]  # To store products for each stage
    is_greeting: bool  # To store whether the message is a greeting
    product_identifier: str  # To store product SKU or name
    branch_id: str  # To store branch ID for stock queries
    stock_data: Dict[str, any]  # To store stock/inventory information from API

class AgentInput(TypedDict):
    """Input state structure for the agent"""
    question: HumanMessage
    conversation_history: List[Dict[str, str]] = []  # Optional conversation history input


# ==================== PYDANTIC MODELS ====================

class IntentClassification(BaseModel):
    """Model for intent classification results"""
    on_topic: str = Field(
        description="Is the question related to building materials? 'Yes' or 'No'"
    )
    intent_type: str = Field(
        description="Type of intent: 'project' for project-based queries, 'product' for product/general queries, 'product-metadata' for specific product information queries with SKU or product name, 'stock' for stock/inventory queries, 'off_topic' for unrelated queries"
    )
    product_identifier: str = Field(
        description="If intent_type is 'product-metadata' or 'stock', this field contains the identified product SKU or name. Empty string otherwise.",
        default=""
    )
    branch_id: str = Field(
        description="If intent_type is 'stock', this field contains the branch ID or branch name mentioned in the query. Empty string otherwise.",
        default=""
    )


class ConversationRelevance(BaseModel):
    """Model for conversation relevance assessment"""
    is_related: str = Field(
        description="Is the current question related to the conversation history? 'Yes' or 'No'"
    )
    summary: str = Field(
        description="Summary of the conversation context if related, otherwise empty string"
    )


class ContextSummary(BaseModel):
    """Model for context summary extracted from conversation history"""
    project_type: str = Field(
        description="Type of project or product category being discussed (e.g., 'decking', 'fencing', 'bathroom renovation'). Empty if not applicable."
    )
    goal: str = Field(
        description="User's main goal or objective (e.g., 'build a new backyard deck', 'renovate kitchen'). Empty if not clear."
    )
    stage: str = Field(
        description="Current stage of the project or conversation (e.g., 'planning', 'material selection', 'installation'). Empty if not applicable."
    )
    preferences: List[str] = Field(
        description="List of preferences mentioned by the user (e.g., ['color', 'type', 'brand', 'price type','weather-resistant', 'durable', 'easy-to-install']). Empty list if none.",
        default=[]
    )
    last_suggested_products: List[str] = Field(
        description="List of products that were recently suggested or discussed (e.g., ['treated pine posts', 'decking screws']). Empty list if none."
    )


class GradedDocuments(BaseModel):
    """Model for document grading results"""
    scores: List[str] = Field(
        description="List of 'Yes' or 'No' values indicating if each document is relevant to the question"
    )


class CategoryDetection(BaseModel):
    """Model for category detection results"""
    category: str = Field(
        description="The detected product category from the user query"
    )
    confidence: str = Field(
        description="Confidence level: 'high', 'medium', or 'low'"
    )


class ProjectStage(BaseModel):
    """Model for a single project stage"""
    stage_name: str = Field(description="Clear, concise name for the stage")
    description: str = Field(description="Brief description of what needs to be done in this stage")


class ProjectStages(BaseModel):
    """Model for project stages"""
    stages: List[ProjectStage] = Field(
        description="List of project stages with stage_name and description for each"
    )
