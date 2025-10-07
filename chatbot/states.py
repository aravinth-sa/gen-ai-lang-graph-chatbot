from typing import TypedDict, List, Dict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.schema import Document
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from .chain import rag_chain
from .retriever import retriever
from config import Config

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
    return get_llm()

def get_grader_llm():
    """Get LLM for document grading (Now using Gemini 2.5 Flash)"""
    return get_llm()

def get_creative_llm():
    """Get LLM for creative responses (Now using Gemini 2.5 Flash)"""
    return get_llm("gpt-4")

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

class AgentState(TypedDict):
    messages: List[BaseMessage]
    documents: List[Document]
    relavant_documents: List[Document]
    on_topic: str
    rephrased_question: str
    proceed_to_generate: bool
    rephrase_count: int
    question: HumanMessage
    conversation_history: List[Dict[str, str]]  # To store conversation context
    category_slot: str  # To store the product category slot
    detected_category: str  # To store the detected product category
    intent_type: str  # To store the intent type: 'project' or 'product'
    project_stages: List[Dict[str, str]]  # To store project stages with descriptions
    stage_products: Dict[str, List[Document]]  # To store products for each stage

# Define the input state structure
class AgentInput(TypedDict):
    question: HumanMessage
    conversation_history: List[Dict[str, str]] = []  # Optional conversation history input

class GradeQuestion(BaseModel):
    score: str = Field(
        description="Question is about the specified topics? If yes -> 'Yes' if not -> 'No'"
    )

class IntentClassification(BaseModel):
    on_topic: str = Field(
        description="Is the question related to building materials? 'Yes' or 'No'"
    )
    intent_type: str = Field(
        description="Type of intent: 'project' for project-based queries, 'product' for product/general queries, 'off_topic' for unrelated queries"
    )

class ConversationRelevance(BaseModel):
    is_related: str = Field(
        description="Is the current question related to the conversation history? 'Yes' or 'No'"
    )
    summary: str = Field(
        description="Summary of the conversation context if related, otherwise empty string"
    )


from datetime import datetime

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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

def intent_classifier(state: AgentState):
    """Classify user intent as project-based or product-based query"""
    print(f"[{get_timestamp()}] Entering intent_classifier##################################")

    # Initialize or use provided conversation history
    if "conversation_history" in state.get("question", {}):
        print(f"[{get_timestamp()}] Using provided conversation history")
        state["conversation_history"] = state["question"].get("conversation_history", [])
    elif "conversation_history" not in state or state["conversation_history"] is None:
        print(f"[{get_timestamp()}] Initializing empty conversation history")
        state["conversation_history"] = []
    else:
        print(f"[{get_timestamp()}] Using existing conversation history with {len(state['conversation_history'])} messages")

    # Initialize state fields
    if "intent_type" not in state:
        state["intent_type"] = ""
    if "project_stages" not in state:
        state["project_stages"] = []
    if "stage_products" not in state:
        state["stage_products"] = {}
        
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
    state["rephrased_question"] = current_question

    system_message = SystemMessage(
        content="""You are an intent classifier for a building material supplier website in New Zealand.
        
        Your task is to:
        1. Determine if the question is related to building materials/construction
        2. Classify the intent type as either 'project' or 'product'
        
        INTENT CLASSIFICATION:
        
        'project' - Questions about:
        - Building or constructing something new (e.g., "I want to build a deck", "building a fence", "constructing a pergola")
        - Renovation or home improvement projects (e.g., "renovating my kitchen", "updating my bathroom")
        - DIY projects that involve multiple steps/stages
        - Questions asking for step-by-step guidance or project planning
        - Questions like "how do I build...", "what are the steps to...", "I'm planning to..."
        
        'product' - Questions about:
        - Specific products or materials (e.g., "what tiles do you have?", "show me decking boards")
        - Product specifications, features, or comparisons
        - Pricing, availability, or ordering information
        - Installation instructions for a specific product
        - General product inquiries
        
        'off_topic' - Questions completely unrelated to building materials or construction
        
        TOPIC CLASSIFICATION:
        - Answer 'Yes' if related to building materials/construction
        - Answer 'No' if completely unrelated (politics, sports, entertainment, etc.)
        
        Return both the topic relevance ('Yes' or 'No') and the intent type ('project', 'product', or 'off_topic')."""
    )

    # Prepare conversation context for classification
    messages = [system_message]
    
    # Add conversation history if available
    if "conversation_history" in state and state["conversation_history"]:
        messages.append(SystemMessage(content="CONVERSATION HISTORY (for context only):"))
        
        for i, msg in enumerate(state["conversation_history"][-4:], 1):
            role = "USER" if msg["role"] == "user" else "ASSISTANT"
            messages.append(HumanMessage(
                content=f"[{i}] {role}: {msg['content']}"
            ))
            print(f"[{get_timestamp()}] Adding message to conversation history: {msg['content']}")
    
    # Add current question with clear separation
    messages.extend([
        SystemMessage(content="CURRENT QUESTION TO CLASSIFY:"),
        HumanMessage(content=state['rephrased_question']),
        SystemMessage(content="Classify the topic relevance and intent type.")
    ])
    
    # Print all messages for debugging
    print("\n### MESSAGES FOR INTENT CLASSIFICATION ###")
    for i, msg in enumerate(messages):
        print(f"Message {i+1} - Type: {type(msg).__name__}, Content: {msg.content}")
    print("### END OF MESSAGES ###\n")
    
    # Create the prompt with conversation context
    intent_prompt = ChatPromptTemplate.from_messages(messages)
    llm = get_default_llm()
    structured_llm = llm.with_structured_output(IntentClassification)
    classifier_llm = intent_prompt | structured_llm
    result = classifier_llm.invoke({})
    
    state["on_topic"] = result.on_topic.strip()
    state["intent_type"] = result.intent_type.strip().lower()
    
    print(f"intent_classifier: on_topic = {state['on_topic']}, intent_type = {state['intent_type']} ##################################")
    return state

# Keep the old function name for backward compatibility
question_classifier = intent_classifier

def intent_router(state: AgentState):
    """Route based on intent type: project or product"""
    print(f"[{get_timestamp()}] Entering intent_router")
    
    on_topic = state.get("on_topic", "").strip().lower()
    intent_type = state.get("intent_type", "").strip().lower()
    
    if on_topic != "yes":
        print("Routing to off_topic_response")
        return "off_topic_response"
    
    if intent_type == "project":
        print("Routing to project_stage_generator")
        return "project_stage_generator"
    else:  # product or default
        print("Routing to retrieve")
        return "retrieve"

# Keep the old function name for backward compatibility
on_topic_router = intent_router


def retrieve(state: AgentState):
    print(f"[{get_timestamp()}] Entering retrieve")
    print("Entering retrieve")
    
    # Build enhanced query by combining context and rephrased question
    query_to_use = state["rephrased_question"]
    
    # If conversation history exists, check if the question is related to the conversation
    if state.get("conversation_history") and len(state["conversation_history"]) > 0:
        # Combined prompt: check relevance AND generate summary in one LLM call
        combined_prompt = ChatPromptTemplate.from_template("""
            Analyze if the current user question is related to or dependent on the previous conversation.
            
            Conversation History:
            {conversation}
            
            Current Question:
            {question}
            
            Task 1 - Determine Relevance:
            Answer 'Yes' if the question:
            - References something from the conversation (e.g., "it", "that", "the one you mentioned")
            - Is a follow-up question about the same topic
            - Requires context from the conversation to be understood
            
            Answer 'No' if the question:
            - Is a completely new topic
            - Can be understood independently without conversation context
            - Does not reference anything from the previous conversation
            
            Task 2 - Generate Summary:
            If the question IS related (Yes), provide a concise summary of the conversation context that is relevant to the current question.
            Focus only on facts and user intent.
            If the question is NOT related (No), leave the summary empty.
            
            Return:
            - is_related: 'Yes' or 'No'
            - summary: conversation summary if related, otherwise empty string
            """)
        
        llm = get_default_llm()
        structured_llm = llm.with_structured_output(ConversationRelevance)
        relevance_chain = combined_prompt | structured_llm
        
        result = relevance_chain.invoke({
            "conversation": state["conversation_history"],
            "question": state["rephrased_question"]
        })
        
        is_related = result.is_related.strip().lower()
        print(f"retrieve: Question relevance to conversation: {is_related}")
        
        # Only use context-enhanced query if the question is related to the conversation
        if is_related == "yes" and result.summary.strip():
            # Combine context summary with rephrased question for better retrieval
            query_to_use = f"{result.summary.strip()}\n\nCurrent question: {state['rephrased_question']}"
            print(f"retrieve: Using context-enhanced query: {query_to_use}")
        else:
            print(f"retrieve: Question not related to conversation, using rephrased question only: {query_to_use}")
    else:
        print(f"retrieve: No conversation history, using rephrased question only: {query_to_use}")
    
    # Retrieve documents using the enhanced query
    documents = retriever.get_retriever().invoke(query_to_use)
    print(f"retrieve: Retrieved {len(documents)} documents")
    state["documents"] = documents
    return state


class GradedDocuments(BaseModel):
    scores: List[str] = Field(
        description="List of 'Yes' or 'No' values indicating if each document is relevant to the question"
    )

def retrieval_grader(state: AgentState):
    print(f"[{get_timestamp()}] Entering retrieval_grader")
    print("Entering retrieval_grader")
    
    if not state["documents"]:
        state["proceed_to_generate"] = False
        return state
        
    system_message = SystemMessage(
        content="""You are a strict grader assessing the relevance of multiple retrieved documents to a user question.
For each document, carefully evaluate if it contains specific, factual information that directly answers or is highly relevant to the user's question.

A document should ONLY be marked as 'Yes' if:
1. It contains specific information that directly answers the user's question
2. The information is factual and not just tangentially related
3. The document provides more than just general background information
4. The content is specific to the query and not too broad or generic

Mark as 'No' if:
- The document is only loosely related to the topic
- It contains only general information without specific details
- The connection to the user's question is too vague or indirect
- The document is about a similar but different topic

Return a list of 'Yes' or 'No' values, one for each document in the same order as provided.
Example: ["No", "Yes", "No"]

Be strict in your assessment. When in doubt, prefer 'No'."""
    )

    # Prepare the documents content for batch processing
    documents_content = [doc.page_content for doc in state["documents"]]
    documents_str = "\n\n---DOCUMENT {}---\n{}\n"
    documents_formatted = "\n".join(
        documents_str.format(i+1, content) 
        for i, content in enumerate(documents_content)
    )

    human_message = HumanMessage(
        content=f"""User question: {state['rephrased_question']}

Instructions:
1. Carefully read each document below
2. For each document, determine if it contains specific, factual information that directly answers the user's question
3. Be strict in your assessment - only mark as 'Yes' if the document is highly relevant and contains specific information that answers the question
4. If the document is only tangentially related or contains only general information, mark as 'No'

Documents to evaluate:
{documents_formatted}

Return a JSON array of 'Yes' or 'No' values in the exact same order as the documents. Example: ["No", "Yes", "No"]

Be strict and precise in your evaluation. When in doubt, prefer 'No'."""
    )

    llm = get_grader_llm()
    structured_llm = llm.with_structured_output(GradedDocuments)
    
    grade_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
    grader_llm = grade_prompt | structured_llm
    
    try:
        result = grader_llm.invoke({})
        scores = [score.strip().lower() for score in result.scores]
        
        # Filter documents based on scores
        relevant_docs = [
            doc for doc, score in zip(state["documents"], scores) 
            if score == "yes"
        ]
        
        # Log results
        for i, (doc, score) in enumerate(zip(state["documents"], scores)):
            print(f"Document {i+1}: {score.upper()} - {doc.page_content[:50]}...")
            
        state["relavant_documents"] = relevant_docs
        state["proceed_to_generate"] = len(relevant_docs) > 0
        print(f"retrieval_grader: {len(relevant_docs)} relevant documents found")
        
    except Exception as e:
        print(f"Error in batch document grading: {str(e)}")
        # Fallback to original behavior if batch processing fails
        state["proceed_to_generate"] = False
    
    return state

def proceed_router(state: AgentState):
    print(f"[{get_timestamp()}] Entering proceed_router")
    print("Entering proceed_router")
    rephrase_count = state.get("rephrase_count", 0)
    MAX_RETRIES = 3  # Set your desired maximum retry limit
    print(f"proceed to generate = {state['proceed_to_generate']}, rephrase count = {rephrase_count}")
    if state.get("proceed_to_generate", False) and rephrase_count <= MAX_RETRIES:
        print("Routing to generate_answer")
        return "generate_answer"
    elif rephrase_count >= MAX_RETRIES:
        print(f"Maximum rephrase attempts ({MAX_RETRIES}) reached. Routing to off_topic_response.")
        return "cannot_answer"
    else:
        print(f"Routing to refine_question (attempt {rephrase_count + 1}/{MAX_RETRIES})")
        return "refine_question"
    
def refine_question(state: AgentState):
    print(f"[{get_timestamp()}] Entering refine_question")
    print("Entering refine_question")
    rephrase_count = state.get("rephrase_count", 0)
    if rephrase_count >= 3:
        print("Maximum rephrase attempts reached")
        return state
    question_to_refine = state["rephrased_question"]
    
    # Prepare system message with conversation context
    system_message = SystemMessage(
        content="""You are a helpful assistant that slightly refines the user's question to improve retrieval results.
                Consider the conversation history when refining to maintain context.
                Provide a slightly adjusted version of the question."""
    )
    
    # Build messages list with conversation history
    messages = [system_message]
    
    # Add conversation history if available
    if "conversation_history" in state and state["conversation_history"]:
        # Add the last few exchanges for context
        messages.append(SystemMessage(content="CONVERSATION HISTORY (for context):"))
        for msg in state["conversation_history"][-3:]:  # Last 3 exchanges
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
    
    # Add the question to refine
    human_message = HumanMessage(
        content=f"Original question: {question_to_refine}\n\nProvide a slightly refined question that considers the conversation context."
    )
    messages.append(human_message)
    
    refine_prompt = ChatPromptTemplate.from_messages(messages)
    llm = get_default_llm()
    prompt = refine_prompt.format()
    response = llm.invoke(prompt)
    refined_question = response.content.strip()
    print(f"refine_question: Refined question: {refined_question}")
    state["rephrased_question"] = refined_question
    state["rephrase_count"] = rephrase_count + 1
    return state

def generate_answer(state: AgentState):
    print(f"[{get_timestamp()}] Entering generate_answer")
    print("Entering generate_answer")
    if "messages" not in state or state["messages"] is None:
        raise ValueError("State must include 'messages' before generating an answer.")

    # Initialize conversation history if it doesn't exist
    if "conversation_history" not in state or state["conversation_history"] is None:
        state["conversation_history"] = []

    # Get the current question and documents
    documents = state["relavant_documents"]
    rephrased_question = state["rephrased_question"]
    
    # Prepare conversation context for RAG
    context_messages = []
    
    # Add conversation history as context (last 4 exchanges)
    if state["conversation_history"]:
        for msg in state["conversation_history"][-4:]:  # Keep last 4 exchanges for context
            if msg["role"] == "user":
                context_messages.append(HumanMessage(content=msg["content"]))
            else:
                context_messages.append(AIMessage(content=msg["content"]))
    
    # Add current question
    context_messages.append(HumanMessage(content=rephrased_question))
    
    # Generate response with RAG chain
    response = rag_chain.invoke(
        {
            "history": context_messages, 
            "context": documents, 
            "question": rephrased_question
        }
    )

    # Get the base response
    generation = response.content.strip()
    
    # Add SKU hyperlinks if any SKU is found
    from .tools import add_sku_hyperlink, format_product_suggestions
    generation = add_sku_hyperlink(generation)
    
    # Add product cards if documents contain product information
    product_suggestions = format_product_suggestions(documents)
    if product_suggestions:
        generation += product_suggestions
    
    # Add sources if available
    if documents:
        # Extract unique source URLs from documents
        sources = set()
        for doc in documents:
            try:
                # Try to get URL from different possible metadata fields
                if hasattr(doc, 'metadata'):
                    metadata = doc.metadata
                    # Check for URL in different possible metadata fields
                    url = metadata.get('url') or metadata.get('source')
                    if url:
                        sources.add(url)
            except Exception as e:
                print(f"Error processing document metadata: {e}")

        # if sources:
        #     generation += "\nSources:\n"
        #     generation += "\n".join(f"- {source}" for source in sorted(sources))
    
    # Update conversation history with the current exchange
    # state["conversation_history"].append({
    #     "role": "user",
    #     "content": rephrased_question
    # })
    # state["conversation_history"].append({
    #     "role": "assistant",
    #     "content": generation
    # })
    
    # Keep conversation history to a reasonable size (last 10 exchanges)
    if len(state["conversation_history"]) > 10:
        state["conversation_history"] = state["conversation_history"][-10:]
    
    # Add the response to messages
    state["messages"].append(AIMessage(content=generation))
    print(f"generate_answer: Generated response: {generation}")
    print(f"[{get_timestamp()}] ENding  question_classifier")
    return state

def cannot_answer(state: AgentState):
    print(f"[{get_timestamp()}] Entering cannot_answer")
    print("Entering cannot_answer")
    if "messages" not in state or state["messages"] is None:
        state["messages"] = []
    state["messages"].append(
        AIMessage(
            content="I'm sorry, but I cannot find the information you're looking for."
        )
    )
    return state


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


import json
import os

# Load the slots-category.json file
def load_category_slots():
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(script_dir, 'dataset', 'slots', 'slots-category.json')
    with open(file_path, 'r') as f:
        return json.load(f)


class CategoryDetection(BaseModel):
    category: str = Field(
        description="The detected product category from the user query"
    )
    confidence: str = Field(
        description="Confidence level: 'high', 'medium', or 'low'"
    )


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


# ==================== PROJECT-BASED RECOMMENDATION NODES ====================

class ProjectStage(BaseModel):
    """Model for a single project stage"""
    stage_name: str = Field(description="Clear, concise name for the stage")
    description: str = Field(description="Brief description of what needs to be done in this stage")

class ProjectStages(BaseModel):
    """Model for project stages"""
    stages: List[ProjectStage] = Field(
        description="List of project stages with stage_name and description for each"
    )

def project_stage_generator(state: AgentState):
    """Generate project stages based on the user's project description"""
    print(f"[{get_timestamp()}] Entering project_stage_generator")
    
    if "project_stages" not in state:
        state["project_stages"] = []
    
    # Get the project description from the question
    project_description = state["rephrased_question"]
    
    # Create system message for stage generation
    system_message = SystemMessage(
        content="""You are a construction and DIY project expert for a building materials supplier in New Zealand.
        
        Your task is to break down a user's project into logical stages/steps.
        
        For each stage, provide:
        1. stage_name: A clear, concise name for the stage (e.g., "Foundation Preparation", "Framing", "Decking Installation")
        2. description: A brief description of what needs to be done in this stage
        
        Guidelines:
        - Keep stages logical and sequential
        - Typically 3-7 stages for most projects
        - Focus on construction/building stages, not planning or design
        - Be specific to the type of project mentioned
        
        Examples:
        - For "building a deck": Foundation Preparation, Frame Construction, Decking Installation, Railing Installation, Finishing
        - For "renovating a bathroom": Demolition, Plumbing Rough-in, Electrical Work, Tiling, Fixture Installation, Finishing
        - For "building a fence": Post Installation, Rail Installation, Panel/Picket Installation, Gate Installation, Finishing
        
        Return a list of stages with stage_name and description for each."""
    )
    
    # Add conversation history for context
    messages = [system_message]
    if "conversation_history" in state and state["conversation_history"]:
        context_str = "\nConversation history:\n"
        for msg in state["conversation_history"][-3:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            context_str += f"{role}: {msg['content']}\n"
        messages.append(SystemMessage(content=context_str))
    
    # Add the project description
    messages.append(HumanMessage(content=f"Project: {project_description}\n\nGenerate the project stages."))
    
    # Create the prompt
    stage_prompt = ChatPromptTemplate.from_messages(messages)
    llm = get_creative_llm()
    structured_llm = llm.with_structured_output(ProjectStages)
    generator_llm = stage_prompt | structured_llm
    
    try:
        result = generator_llm.invoke({})
        # Convert Pydantic models to dictionaries
        state["project_stages"] = [
            {"stage_name": stage.stage_name, "description": stage.description}
            for stage in result.stages
        ]
        print(f"project_stage_generator: Generated {len(result.stages)} stages")
        for i, stage in enumerate(result.stages, 1):
            print(f"  Stage {i}: {stage.stage_name}")
    except Exception as e:
        print(f"Error generating project stages: {str(e)}")
        state["project_stages"] = []
    
    return state


def project_stage_product_retrieval(state: AgentState):
    """Retrieve products for each project stage"""
    print(f"[{get_timestamp()}] Entering project_stage_product_retrieval")
    
    if "stage_products" not in state:
        state["stage_products"] = {}
    
    project_stages = state.get("project_stages", [])
    
    if not project_stages:
        print("No project stages found, skipping product retrieval")
        return state
    
    # For each stage, retrieve relevant products
    for stage in project_stages:
        stage_name = stage.get("stage_name", "")
        stage_description = stage.get("description", "")
        
        # Create a search query combining stage name and description
        search_query = f"{stage_name}: {stage_description}"
        
        print(f"Retrieving products for stage: {stage_name}")
        
        # Use the retriever to get products for this stage
        try:
            documents = retriever.get_retriever(search_kwargs={"k": 3}).invoke(search_query)
            state["stage_products"][stage_name] = documents
            print(f"  Retrieved {len(documents)} products for {stage_name}")
        except Exception as e:
            print(f"Error retrieving products for stage {stage_name}: {str(e)}")
            state["stage_products"][stage_name] = []
    
    return state


def generate_project_response(state: AgentState):
    """Generate a comprehensive project response with stages and products"""
    print(f"[{get_timestamp()}] Entering generate_project_response")
    
    if "messages" not in state or state["messages"] is None:
        state["messages"] = []
    
    project_stages = state.get("project_stages", [])
    stage_products = state.get("stage_products", {})
    
    if not project_stages:
        # Fallback if no stages were generated
        state["messages"].append(
            AIMessage(content="I apologize, but I couldn't generate a project plan for your request. Could you provide more details about your project?")
        )
        return state
    
    # Build the response
    response_parts = []
    
    # Introduction
    response_parts.append(f"Here's a step-by-step guide for your project:\n")
    
    # For each stage, add stage info and product recommendations
    for i, stage in enumerate(project_stages, 1):
        stage_name = stage.get("stage_name", f"Stage {i}")
        stage_description = stage.get("description", "")
        
        response_parts.append(f"\n## {i}. {stage_name}")
        response_parts.append(f"{stage_description}\n")
        
        # Add product recommendations for this stage
        products = stage_products.get(stage_name, [])
        if products:
            # Use the product card formatter
            from .tools import format_product_suggestions
            product_cards = format_product_suggestions(products)
            if product_cards:
                response_parts.append("**Recommended Products:**")
                response_parts.append(product_cards)
            else:
                # Check if we have valid products in fallback
                valid_products = []
                for doc in products[:3]:  # Limit to top 3 products per stage
                    if hasattr(doc, 'metadata'):
                        product_code = doc.metadata.get('product_id', 'N/A')
                        # Only include products with valid SKU codes
                        if product_code and product_code != 'N/A':
                            product_title = doc.metadata.get('product_title', 'Product')
                            valid_products.append(f"- {product_title} (SKU: {product_code})")
                
                if valid_products:
                    response_parts.append("**Recommended Products:**")
                    response_parts.extend(valid_products)
                else:
                    response_parts.append("*Consult with our team for specific product recommendations for this stage.*")
        else:
            response_parts.append("*Consult with our team for specific product recommendations for this stage.*")
    
    # Add closing message
    response_parts.append("\n\n💡 **Tip:** Visit your local PlaceMakers store or contact our team for personalized advice and to ensure you have all the materials you need for your project!")
    
    # Combine all parts
    generation = "\n".join(response_parts)
    
    # Add SKU hyperlinks
    from .tools import add_sku_hyperlink
    generation = add_sku_hyperlink(generation)
    
    # Add the response to messages
    state["messages"].append(AIMessage(content=generation))
    print(f"generate_project_response: Generated project response with {len(project_stages)} stages")
    
    return state