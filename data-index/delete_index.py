#!/usr/bin/env python3
"""
Script to delete the existing Pinecone index
"""

import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from pinecone import Pinecone

def delete_index():
    """Delete the existing index"""
    try:
        # Initialize Pinecone
        pc = Pinecone(api_key=Config.PINECONE_API_KEY, environment=Config.PINECONE_ENVIRONMENT)
        
        index_name = Config.PINECONE_INDEX_NAME
        
        # Check if index exists
        if index_name in pc.list_indexes():
            print(f"Deleting index: {index_name}")
            pc.delete_index(index_name)
            print(f"Index {index_name} deleted successfully")
        else:
            print(f"Index {index_name} does not exist")
            
    except Exception as e:
        print(f"Error deleting index: {e}")

if __name__ == "__main__":
    delete_index()
