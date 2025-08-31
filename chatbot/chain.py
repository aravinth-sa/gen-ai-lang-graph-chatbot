from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from config import Config

# Set up Gemini (gpt-4o equivalent from Google)
llm = ChatOpenAI(api_key=Config.OPENAI_API_KEY,model="gpt-4o")

template = """Answer the question based on the following context and the Chathistory. Especially take the latest question into consideration:

Chathistory: {history}

Context: {context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

rag_chain = prompt | llm