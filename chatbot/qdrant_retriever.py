import os
import sys
from typing import Any, Dict, List
from pathlib import Path

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from openai import OpenAI
import logging

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue
    from langchain_qdrant import QdrantVectorStore
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    raise ImportError("Please install qdrant-client, langchain-community, and langchain-openai.")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QdrantRetriever:
    """Class to handle Qdrant vector retrieval with OpenAI embeddings"""
    def __init__(self):
        try:
            Config.validate_config()
            self.config = Config()
            # Initialize OpenAI client for embeddings
            self.openai_client = OpenAI(api_key=self.config.OPENAI_API_KEY)
            # Initialize Qdrant client (in-memory)
            self.qdrant = QdrantClient(":memory:")
            self.collection_name = os.getenv('QDRANT_COLLECTION_NAME', 'placemakers-content')
            self._get_or_create_collection()
            # Embedding model for LangChain
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                dimensions=1536,
                openai_api_key=self.config.OPENAI_API_KEY
            )
            logger.info("QdrantRetriever initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize QdrantRetriever: {e}")
            raise

    def _get_or_create_collection(self):
        # Check if collection exists, else create
        collections = self.qdrant.get_collections().collections
        if self.collection_name not in [c.name for c in collections]:
            from qdrant_client.http.models import VectorParams, Distance
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.config.VECTOR_DIMENSION,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created new Qdrant collection: {self.collection_name}")
        else:
            logger.info(f"Using existing Qdrant collection: {self.collection_name}")

    def get_embedding(self, text: str) -> List[float]:
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
        try:
            if search_kwargs is None:
                search_kwargs = {"k": 3}
            # Use LangChain's Qdrant vectorstore wrapper
            vectorstore = QdrantVectorStore(
                client=self.qdrant,
                collection_name=self.collection_name,
                embedding=self.embeddings,
                content_payload_key="text"
            )
            retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
            return retriever
        except Exception as e:
            logger.error(f"Error creating Qdrant retriever: {e}")
            raise

# Create a singleton instance
qdrant_retriever = QdrantRetriever()
