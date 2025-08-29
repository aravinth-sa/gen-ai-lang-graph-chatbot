#!/usr/bin/env python3
"""
Test script to verify Pinecone and OpenAI setup
"""

import sys
import os
from pathlib import Path

from pinecone import Pinecone

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_config():
    """Test configuration setup"""
    print("Testing configuration...")
    
    try:
        from config import Config
        
        # Test config validation
        Config.validate_config()
        print("✅ Configuration validation passed")
        
        # Test config values
        print(f"Pinecone Environment: {Config.PINECONE_ENVIRONMENT}")
        print(f"Pinecone Index Name: {Config.PINECONE_INDEX_NAME}")
        print(f"Vector Dimension: {Config.VECTOR_DIMENSION}")
        print(f"Batch Size: {Config.BATCH_SIZE}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_openai():
    """Test OpenAI API connection"""
    print("\nTesting OpenAI API...")
    
    try:
        from config import Config
        from openai import OpenAI
        
        client = OpenAI(api_key=Config.OPENAI_API_KEY)
        
        # Test with a simple embedding
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input="Hello, world!"
        )
        
        embedding = response.data[0].embedding
        print(f"✅ OpenAI API test passed - Embedding dimension: {len(embedding)}")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI API test failed: {e}")
        return False

def test_pinecone():
    """Test Pinecone API connection"""
    print("\nTesting Pinecone API...")
    
    try:
        from config import Config
        
        
        # Initialize Pinecone
        pc = Pinecone(api_key=Config.PINECONE_API_KEY, environment=Config.PINECONE_ENVIRONMENT)
        # List indexes
        indexes = pc.list_indexes()
        print(f"✅ Pinecone API test passed - Available indexes: {indexes}")
        
        # Check if our index exists
        if Config.PINECONE_INDEX_NAME in indexes:
            print(f"✅ Index '{Config.PINECONE_INDEX_NAME}' exists")
            
            # Get index stats
            index = pc.Index(Config.PINECONE_INDEX_NAME)
            stats = index.describe_index_stats()
            print(f"Index stats: {stats}")
        else:
            print(f"ℹ️  Index '{Config.PINECONE_INDEX_NAME}' does not exist yet (will be created during indexing)")
        
        return True
        
    except Exception as e:
        print(f"❌ Pinecone API test failed: {e}")
        return False

def test_json_data():
    """Test JSON data file"""
    print("\nTesting JSON data file...")
    
    try:
        json_file_path = Path(__file__).parent.parent / "dataset" / "output" / "output-raw-page.json"
        
        if not json_file_path.exists():
            print(f"❌ JSON file not found: {json_file_path}")
            return False
        
        import json
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ JSON data test passed - Loaded {len(data)} documents")
        
        # Show sample document structure
        if data:
            sample_doc = data[0]
            print(f"Sample document keys: {list(sample_doc.keys())}")
            print(f"Sample page_id: {sample_doc.get('page_id', 'N/A')}")
            print(f"Sample page_name: {sample_doc.get('page_name', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ JSON data test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🔍 Running setup tests...\n")
    
    tests = [
        test_config,
        test_openai,
        test_pinecone,
        test_json_data
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    print("\n" + "="*50)
    print("📊 Test Results Summary:")
    print("="*50)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 All {total} tests passed! You're ready to index data.")
        print("\nNext steps:")
        print("1. Run: python index_data_to_db.py")
        print("2. Run: python query_index.py")
    else:
        print(f"⚠️  {passed}/{total} tests passed")
        print("\nPlease fix the failed tests before proceeding.")
        print("Check the error messages above and ensure:")
        print("- API keys are correctly set in .env file")
        print("- Internet connection is available")
        print("- JSON data file exists")

if __name__ == "__main__":
    main()
