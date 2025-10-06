from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from config import Config

# Set up Gemini (gpt-4o equivalent from Google)
llm = ChatOpenAI(api_key=Config.OPENAI_API_KEY,model="gpt-4o-mini")

template = """You are an PlaceMakers Expert in New Zealand, Answer the question based on the following documents and the Chathistory. Especially take the latest question into consideration:

Chathistory: {history}

documents: {context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

rag_chain = prompt | llm