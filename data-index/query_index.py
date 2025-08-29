import sys
import os
from typing import List, Dict, Any
from pathlib import Path

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from pinecone import Pinecone
from openai import OpenAI
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PineconeQuery:
    """Class to query the Pinecone vector database"""
    
    def __init__(self):
        """Initialize the query interface"""
        try:
            Config.validate_config()
            self.config = Config()
            
            # Initialize OpenAI client for embeddings
            self.openai_client = OpenAI(api_key=self.config.OPENAI_API_KEY)
            
            # Initialize Pinecone
            self.pc = Pinecone(api_key=self.config.PINECONE_API_KEY, environment=self.config.PINECONE_ENVIRONMENT)
            
            # Get index
            self.index = self.pc.Index(self.config.PINECONE_INDEX_NAME)
            
            logger.info("PineconeQuery initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize PineconeQuery: {e}")
            raise
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a text using OpenAI API"""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-large",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            raise
    
    def search(self, query: str, top_k: int = 5, filter_dict: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search the index with a query"""
        try:
            # Get embedding for query
            query_embedding = self.get_embedding(query)
            
            # Search the index
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            return results.matches
            
        except Exception as e:
            logger.error(f"Error searching index: {e}")
            raise
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the index"""
        try:
            stats = self.index.describe_index_stats()
            return stats
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {}
    
    def list_all_vectors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List all vectors in the index (for debugging)"""
        try:
            # Get stats to see total count
            stats = self.get_index_stats()
            total_vectors = stats.get('total_vector_count', 0)
            
            logger.info(f"Total vectors in index: {total_vectors}")
            
            # Fetch vectors (note: this is limited by Pinecone's API)
            results = self.index.query(
                vector=[0] * self.config.VECTOR_DIMENSION,  # Dummy vector
                top_k=min(limit, total_vectors),
                include_metadata=True
            )
            
            return results.matches
            
        except Exception as e:
            logger.error(f"Error listing vectors: {e}")
            raise

def main():
    """Main function to demonstrate querying"""
    try:
        # Initialize query interface
        query_interface = PineconeQuery()
        
        # Get index stats
        stats = query_interface.get_index_stats()
        logger.info(f"Index statistics: {stats}")
        
        # Example queries
        example_queries = [
            "decking materials and installation",
            "fencing options for home",
            "timber products",
            "outdoor construction",
            "building materials"
        ]
        
        for query in example_queries:
            logger.info(f"\n--- Searching for: '{query}' ---")
            results = query_interface.search(query, top_k=3)
            
            for i, match in enumerate(results, 1):
                metadata = match.metadata
                logger.info(f"{i}. Score: {match.score:.4f}")
                logger.info(f"   Page: {metadata.get('page_name', 'N/A')}")
                logger.info(f"   Heading: {metadata.get('heading', 'N/A')}")
                logger.info(f"   URL: {metadata.get('url', 'N/A')}")
                logger.info(f"   Products: {metadata.get('num_products', 0)}")
                logger.info("")
        
        # List some vectors for debugging
        logger.info("--- Sample vectors in index ---")
        vectors = query_interface.list_all_vectors(limit=5)
        for i, vector in enumerate(vectors, 1):
            metadata = vector.metadata
            logger.info(f"{i}. ID: {vector.id}")
            logger.info(f"   Page: {metadata.get('page_name', 'N/A')}")
            logger.info(f"   Score: {vector.score:.4f}")
            logger.info("")
        
    except Exception as e:
        logger.error(f"Query process failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
