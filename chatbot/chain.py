from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from config import Config

# Import centralized LLM configuration
# Note: We can't directly import from .states due to circular dependency
# So we define a local function that creates the same LLM instance

# ==================== OpenAI RAG LLM ====================
def get_rag_llm():
    """Get OpenAI LLM for RAG chain generation.
    
    To change the model used for answer generation:
    1. Modify the model parameter below
    2. Or switch to get_gemini_rag_llm() for Google Gemini
    """
    return ChatOpenAI(api_key=Config.OPENAI_API_KEY, model="gpt-3.5-turbo", temperature=0.0)

# ==================== Google Gemini RAG LLM ====================
def get_gemini_rag_llm():
    """Get Google Gemini LLM for RAG chain generation.
    
    To use Gemini for answer generation:
    1. Replace 'llm = get_rag_llm()' with 'llm = get_gemini_rag_llm()' below
    2. Ensure GOOGLE_API_KEY is set in your .env file
    """
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.0,
        google_api_key=Config.GOOGLE_API_KEY,
        convert_system_message_to_human=True
    )

# Set up LLM for RAG chain (now using Gemini 2.5 Flash)
llm = get_rag_llm()  # Using Gemini 2.5 Flash

template = """You are an PlaceMakers Expert in New Zealand, Answer the question based on the following documents and the Chathistory. Especially take the latest question into consideration:

Chathistory: {history}

documents: {context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

rag_chain = prompt | llm