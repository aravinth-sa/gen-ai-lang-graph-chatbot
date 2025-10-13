"""
Active nodes used in the conversation flow graph.

This module contains all the node functions that are actively used in graph.py.
These nodes handle the main conversation flow including intent classification,
retrieval, grading, and response generation.
"""

from typing import List, Dict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.schema import Document
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime

from .states import (
    AgentState, IntentClassification, ContextSummary, GradedDocuments,
    ProjectStage, ProjectStages, get_timestamp, get_default_llm,
    get_grader_llm, get_creative_llm
)
from .chain import rag_chain
from .retriever import retriever


# ==================== GREETING HANDLER NODE ====================

def greeting_handler(state: AgentState):
    """Handle greeting messages from users with a standard welcome response"""
    print(f"[{get_timestamp()}] Entering greeting_handler")
    
    # Get the current question
    current_question = state["question"].content.lower().strip()
    
    # List of greeting patterns
    greetings = [
        "hi", "hello", "hey", "kia ora", "good morning", "good afternoon", 
        "good evening", "greetings", "howdy", "sup", "yo", "hiya"
    ]
    
    # Check if the message is a greeting (exact match or starts with greeting)
    is_greeting = False
    for greeting in greetings:
        if current_question == greeting or current_question.startswith(greeting + " ") or current_question.startswith(greeting + "!"):
            is_greeting = True
            break
    
    # Store greeting detection result in state
    state["is_greeting"] = is_greeting
    
    if is_greeting:
        print(f"[{get_timestamp()}] Detected greeting, generating welcome response")
        
        # Initialize messages if not present
        if "messages" not in state or state["messages"] is None:
            state["messages"] = []
        
        # Add user question to messages if not already there
        if state["question"] not in state["messages"]:
            state["messages"].append(state["question"])
        
        # Generate welcome response
        welcome_message = """Kia ora! Welcome to BuildMate, your 24/7 PlaceMakers expert. I'm here to help with your building projects, product searches, or order needs. What can I do for you today?<br><br>Here are some things I can help you with:<br>• <strong>Find Products</strong> - Browse our range of building materials and products<br>• <strong>Get Project Advice</strong> - Get step-by-step guidance for your building projects<br>• <strong>Learn About Deals</strong> - Discover current offers and promotions"""
        
        state["messages"].append(AIMessage(content=welcome_message))
        print(f"[{get_timestamp()}] Generated greeting response")
    
    return state


def greeting_router(state: AgentState):
    """Route based on whether message is a greeting"""
    print(f"[{get_timestamp()}] Entering greeting_router")
    
    if state.get("is_greeting", False):
        print("Routing to END (greeting handled)")
        return "end_greeting"
    else:
        print("Routing to intent_classifier (not a greeting)")
        return "continue_flow"


# ==================== INTENT CLASSIFICATION NODE ====================

def intent_classifier(state: AgentState):
    """Classify user intent as project-based or product-based query"""
    print(f"[{get_timestamp()}] Entering intent_classifier ##################################")

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
    llm = get_grader_llm()
    structured_llm = llm.with_structured_output(IntentClassification)
    classifier_llm = intent_prompt | structured_llm
    result = classifier_llm.invoke({})
    
    state["on_topic"] = result.on_topic.strip()
    state["intent_type"] = result.intent_type.strip().lower()
    
    print(f"intent_classifier: on_topic = {state['on_topic']}, intent_type = {state['intent_type']} ##################################")
    
    # ==================== CONTEXT SUMMARIZER ====================
    # Generate context summary from conversation history
    if "context_summary" not in state:
        state["context_summary"] = {}
    
    if state.get("conversation_history") and len(state["conversation_history"]) > 0:
        print(f"[{get_timestamp()}] Generating context summary from conversation history")
        
        # Create system message for context summarization
        summary_system_message = SystemMessage(
            content="""You are a context summarizer for a building materials chatbot.
            
            Analyze the conversation history and extract key information into a structured summary.
            
            Extract the following information:
            1. project_type: The type of project or product category being discussed (e.g., 'decking', 'fencing', 'bathroom renovation'). 
               Leave empty if not applicable or unclear.
            
            2. goal: The user's main goal or objective (e.g., 'build a new backyard deck', 'renovate kitchen'). 
               Leave empty if not clear from the conversation.
            
            3. stage: The current stage of the project or conversation (e.g., 'planning', 'material selection', 'installation'). 
               Leave empty if not applicable.
            
            4. last_suggested_products: List of products that were recently suggested or discussed (e.g., ['treated pine posts', 'decking screws']). 
               Return empty list if no products were mentioned.
            
            Be concise and extract only factual information from the conversation.
            If information is not present or unclear, use empty strings or empty lists as appropriate."""
        )
        
        # Build conversation history string
        conversation_str = "CONVERSATION HISTORY:\n"
        for msg in state["conversation_history"][-6:]:  # Last 6 messages for context
            role = "User" if msg["role"] == "user" else "Assistant"
            conversation_str += f"{role}: {msg['content']}\n"
        
        # Create messages for summarization
        summary_messages = [
            summary_system_message,
            HumanMessage(content=conversation_str),
            SystemMessage(content="Extract the context summary as structured data.")
        ]
        
        # Create the prompt
        summary_prompt = ChatPromptTemplate.from_messages(summary_messages)
        llm = get_default_llm()
        structured_llm = llm.with_structured_output(ContextSummary)
        summarizer_llm = summary_prompt | structured_llm
        
        try:
            summary_result = summarizer_llm.invoke({})
            state["context_summary"] = {
                "project_type": summary_result.project_type.strip(),
                "goal": summary_result.goal.strip(),
                "stage": summary_result.stage.strip(),
                "last_suggested_products": summary_result.last_suggested_products
            }
            print(f"intent_classifier: Generated context summary: {state['context_summary']}")
        except Exception as e:
            print(f"Error generating context summary: {str(e)}")
            state["context_summary"] = {
                "project_type": "",
                "goal": "",
                "stage": "",
                "last_suggested_products": []
            }
    else:
        # No conversation history, initialize empty summary
        state["context_summary"] = {
            "project_type": "",
            "goal": "",
            "stage": "",
            "last_suggested_products": []
        }
        print(f"[{get_timestamp()}] No conversation history, initialized empty context summary")
    
    # ==================== END CONTEXT SUMMARIZER ====================
    print(f"[{get_timestamp()}] Finished Intent Classifier")
    return state


# ==================== INTENT ROUTER ====================

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


# ==================== RETRIEVAL NODE ====================

def retrieve(state: AgentState):
    print(f"[{get_timestamp()}] Entering retrieve")
    print("Entering retrieve")
    
    # Build enhanced query using context summary
    query_to_use = state["rephrased_question"]
    
    # Use context_summary instead of conversation_history for retrieval
    context_summary = state.get("context_summary", {})
    
    if context_summary and any(context_summary.values()):
        # Build context string from summary
        context_parts = []
        
        if context_summary.get("project_type"):
            context_parts.append(f"Project type: {context_summary['project_type']}")
        
        if context_summary.get("goal"):
            context_parts.append(f"Goal: {context_summary['goal']}")
        
        if context_summary.get("stage"):
            context_parts.append(f"Current stage: {context_summary['stage']}")
        
        if context_summary.get("last_suggested_products"):
            products_str = ", ".join(context_summary['last_suggested_products'])
            context_parts.append(f"Previously discussed products: {products_str}")
        
        if context_parts:
            context_str = " | ".join(context_parts)
            query_to_use = f"Context: {context_str}\n\nQuestion: {state['rephrased_question']}"
            print(f"retrieve: Using context-enhanced query with summary: {query_to_use}")
        else:
            print(f"retrieve: Context summary exists but is empty, using rephrased question only")
    else:
        print(f"retrieve: No context summary, using rephrased question only: {query_to_use}")
    
    # Retrieve documents using the enhanced query
    documents = retriever.get_retriever().invoke(query_to_use)
    print(f"retrieve: Retrieved {len(documents)} documents")
    print(f"[{get_timestamp()}] Finished retrival ##################################")
    state["documents"] = documents
    return state


# ==================== RETRIEVAL GRADER NODE ====================

def retrieval_grader(state: AgentState):
    print(f"[{get_timestamp()}] Entering retrieval_grader ##################################")
    
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
        print(f"[{get_timestamp()}] Finished retrieval_grader ##################################")
        
    except Exception as e:
        print(f"Error in batch document grading: {str(e)}")
        # Fallback to original behavior if batch processing fails
        state["proceed_to_generate"] = False
    
    return state


# ==================== PROCEED ROUTER ====================

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


# ==================== REFINE QUESTION NODE ====================

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


# ==================== GENERATE ANSWER NODE ====================

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
    
    # Prepare context using context_summary instead of conversation_history
    context_summary = state.get("context_summary", {})
    
    # Build context string from summary
    context_str = ""
    if context_summary:
        if context_summary.get("project_type"):
            context_str += f"Project: {context_summary['project_type']}. "
        if context_summary.get("goal"):
            context_str += f"Goal: {context_summary['goal']}. "
        if context_summary.get("stage"):
            context_str += f"Stage: {context_summary['stage']}. "
        if context_summary.get("last_suggested_products"):
            products = ", ".join(context_summary['last_suggested_products'])
            context_str += f"Previous products: {products}."
    
    # Generate response with RAG chain using context_summary
    response = rag_chain.invoke(
        {
            "history": context_str,  # Using context summary instead of message history
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
    #print(f"generate_answer: Generated response: {generation}")
    print(f"[{get_timestamp()}] ENding  generate_answer ##################################")
    return state


# ==================== CANNOT ANSWER NODE ====================

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


# ==================== PROJECT-BASED RECOMMENDATION NODES ====================

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
            documents = retriever.get_retriever(search_kwargs={"k": 5,"score_threshold": 0.7}).invoke(search_query)
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
