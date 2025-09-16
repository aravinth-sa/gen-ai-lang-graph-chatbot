from typing import TypedDict, List, Dict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.schema import Document
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from chatbot.chain import rag_chain
from chatbot.retriever import retriever

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


class GradeQuestion(BaseModel):
    score: str = Field(
        description="Question is about the specified topics? If yes -> 'Yes' if not -> 'No'"
    )


from datetime import datetime

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def question_rewriter(state: AgentState):
    print(f"[{get_timestamp()}] Entering question_rewriter with following state: {state}")

    # Initialize conversation history if it doesn't exist
    if "conversation_history" not in state or state["conversation_history"] is None:
        state["conversation_history"] = []

    # Only reset these state variables if we're starting a new conversation
    if len(state["conversation_history"]) == 0:
        state["documents"] = []
        state["relavant_documents"] = []
        state["on_topic"] = ""
        state["rephrased_question"] = ""
        state["proceed_to_generate"] = False
        state["rephrase_count"] = 0

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
    llm = ChatOpenAI(model="gpt-4o-mini")
    prompt = rephrase_prompt.format()
    response = llm.invoke(prompt)
    better_question = response.content.strip()
    print(f"question_rewriter: Rephrased question: {better_question}")
    state["rephrased_question"] = better_question
    
    return state

def question_classifier(state: AgentState):
    print(f"[{get_timestamp()}] Entering question_classifier")

    # Initialize conversation history if it doesn't exist
    if "conversation_history" not in state or state["conversation_history"] is None:
        state["conversation_history"] = []

    # Only reset these state variables if we're starting a new conversation
    if len(state["conversation_history"]) == 0:
        state["documents"] = []
        state["on_topic"] = ""
        state["rephrased_question"] = ""
        state["proceed_to_generate"] = False
        state["rephrase_count"] = 0

    if "messages" not in state or state["messages"] is None:
        state["messages"] = []

    # Add current question to messages if not already present
    if state["question"] not in state["messages"]:
        state["messages"].append(state["question"])

    # Prepare conversation context for rephrasing
    current_question = state["question"].content
    state["rephrased_question"] = current_question

    system_message = SystemMessage(
        content="""You are a domain-specific classifier for a building material supplier website.
        
        Your task is to determine if a user's question is related to building materials, construction, or related services. 
        
        ALWAYS ANSWER 'Yes' IF THE QUESTION IS RELATED TO:
        - Any building or construction materials (e.g., wood, tiles, pipes, fixtures)
        - Kitchen or bathroom products and installations
        - Home improvement, renovation, or DIY projects
        - Product specifications, features, or comparisons
        - Pricing, availability, or ordering information
        - Installation, maintenance, or usage instructions
        - Delivery, shipping, or project planning
        - Any follow-up questions about previously discussed products or topics
        - General questions about the company's products or services
        - Questions containing product names, categories, or types
        
        ONLY ANSWER 'No' IF:
        - The question is completely unrelated to building materials or construction
        - The question is about topics like politics, sports, entertainment, etc.
        - The question is clearly spam or nonsensical
        
        When in doubt, default to 'Yes' to ensure we don't miss relevant questions.
        
        Respond with exactly 'Yes' or 'No' only, no explanations."""
    )

    # Prepare conversation context for classification
    messages = [system_message]
    
    # Add conversation history if available
    if "conversation_history" in state and state["conversation_history"]:
        # Add the last few exchanges for context with clear separation
        messages.append(SystemMessage(content="CONVERSATION HISTORY (for context only):"))
        
        for i, msg in enumerate(state["conversation_history"][-4:], 1):  # Last 4 exchanges
            role = "USER" if msg["role"] == "user" else "ASSISTANT"
            messages.append(HumanMessage(
                content=f"[{i}] {role}: {msg['content']}"
            ))
    
    # Add current question with clear separation
    messages.extend([
        SystemMessage(content="CURRENT QUESTION TO CLASSIFY:"),
        HumanMessage(content=state['rephrased_question']),
        SystemMessage(content="Based on the conversation history and current question, is this related to building materials? Answer 'Yes' or 'No' only.")
    ])
    
    # Create the prompt with conversation context
    grade_prompt = ChatPromptTemplate.from_messages(messages)
    llm = ChatOpenAI(model="gpt-4o-mini")
    structured_llm = llm.with_structured_output(GradeQuestion)
    grader_llm = grade_prompt | structured_llm
    result = grader_llm.invoke({})
    state["on_topic"] = result.score.strip()
    print(f"question_classifier: on_topic = {state['on_topic']}")
    return state

def on_topic_router(state: AgentState):
    print(f"[{get_timestamp()}] Entering on_topic_router")
    print("Entering on_topic_router")
    on_topic = state.get("on_topic", "").strip().lower()
    if on_topic == "yes":
        print("Routing to retrieve")
        return "retrieve"
    else:
        print("Routing to off_topic_response")
        return "off_topic_response"


def retrieve(state: AgentState):
    print(f"[{get_timestamp()}] Entering retrieve")
    print("Entering retrieve")
    documents = retriever.get_retriever().invoke(state["rephrased_question"])
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

    llm = ChatOpenAI(model="gpt-3.5-turbo")
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
    system_message = SystemMessage(
        content="""You are a helpful assistant that slightly refines the user's question to improve retrieval results.
Provide a slightly adjusted version of the question."""
    )
    human_message = HumanMessage(
        content=f"Original question: {question_to_refine}\n\nProvide a slightly refined question."
    )
    refine_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
    llm = ChatOpenAI(model="gpt-4o-mini")
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
    from chatbot.tools import add_sku_hyperlink
    generation = add_sku_hyperlink(generation)
    
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

        if sources:
            generation += "\nSources:\n"
            generation += "\n".join(f"- {source}" for source in sorted(sources))
    
    # Update conversation history with the current exchange
    state["conversation_history"].append({
        "role": "user",
        "content": rephrased_question
    })
    state["conversation_history"].append({
        "role": "assistant",
        "content": generation
    })
    
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
    chat = ChatOpenAI(temperature=0.7)  # Increased temperature for more creative responses
    
    # Generate a creative response
    response = chat.invoke([
        SystemMessage(content="You are a creative assistant that generates fun, engaging responses when asked off-topic questions."),
        HumanMessage(content="I was asked this question but it's off-topic. Please generate a creative, fun response that gently guides the conversation back to the topic of Building Materials & Hardware. The original question was: " + state["question"].content)
    ])
    
    # Add the AI's response to the messages
    state["messages"].append(AIMessage(content=response.content))
    return state