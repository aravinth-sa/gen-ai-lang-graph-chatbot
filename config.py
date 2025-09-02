import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for API keys and settings"""
    
    # Pinecone Configuration
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT', 'gcp-starter')
    PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'placemakers-content')
    
    # OpenAI Configuration (for embeddings)
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    
    # Vector Database Settings
    VECTOR_DIMENSION = 1536  # OpenAI text-embedding-3-large dimension
    METRIC = 'cosine'
    
    # Indexing Settings
    BATCH_SIZE = 100
    MAX_TEXT_LENGTH = 8000  # Maximum text length for embedding
    
    @classmethod
    def validate_config(cls):
        """Validate that all required API keys are present"""
        missing_keys = []
        
        if not cls.PINECONE_API_KEY:
            missing_keys.append('PINECONE_API_KEY')
        
        if not cls.OPENAI_API_KEY:
            missing_keys.append('OPENAI_API_KEY')
        
        if missing_keys:
            raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}")
        
        return True
