import os
import sys
from typing import Any, Dict, List
from pathlib import Path

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PineconeRetriever:
    """Class to handle Pinecone vector retrieval with OpenAI embeddings"""
    
    def __init__(self):
        """Initialize the retriever with Pinecone and OpenAI clients"""
        try:
            # Validate config
            Config.validate_config()
            self.config = Config()
            
            # Initialize OpenAI client for embeddings
            self.openai_client = OpenAI(api_key=self.config.OPENAI_API_KEY)
            
            # Initialize Pinecone
            self.pc = Pinecone(api_key=self.config.PINECONE_API_KEY)
            
            # Get index
            self.index = self.pc.Index(self.config.PINECONE_INDEX_NAME)
            
            # Initialize OpenAI embeddings for LangChain with the new package
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                dimensions=1536,
                openai_api_key=self.config.OPENAI_API_KEY
            )
            
            logger.info("PineconeRetriever initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize PineconeRetriever: {e}")
            raise
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a text using OpenAI API"""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            raise
    
    def get_retriever(self, search_kwargs: Dict[str, Any] = None):
        """Get a retriever instance with the specified search parameters"""
        try:
            # Default search kwargs
            if search_kwargs is None:
                search_kwargs = {"k": 3}  # Default to top 5 results
            
            # Ensure we're using the correct embedding model that matches the index
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                dimensions=1536,  # Match the index dimension
                openai_api_key=self.config.OPENAI_API_KEY
            )
                
            # Create a retriever using LangChain's PineconeVectorStore
            vectorstore = PineconeVectorStore(
                index=self.index,
                embedding=self.embeddings,
                text_key="text",  # This is the key that will be used to extract text from metadata
                namespace=""  # Make sure we're using the default namespace
            )
            
            # Create a custom retriever that formats the results correctly
            retriever = vectorstore.as_retriever(
                search_kwargs=search_kwargs
            )
            
            return retriever
            
        except Exception as e:
            logger.error(f"Error creating retriever: {e}")
            raise

# Create a singleton instance
retriever = PineconeRetriever()
