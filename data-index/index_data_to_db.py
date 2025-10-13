import json
import logging
import sys
import os
import csv
from typing import List, Dict, Any
from pathlib import Path

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from openai import OpenAI
import time
from pinecone import Pinecone, ServerlessSpec

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('indexing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PineconeIndexer:
    """Class to handle indexing data into Pinecone vector database"""
    
    def __init__(self):
        """Initialize the indexer with configuration validation"""
        try:
            Config.validate_config()
            self.config = Config()
            
            # Initialize OpenAI client for embeddings
            self.openai_client = OpenAI(api_key=self.config.OPENAI_API_KEY)
            
            # Initialize Pinecone
            self.pc = Pinecone(api_key=self.config.PINECONE_API_KEY, environment=self.config.PINECONE_ENVIRONMENT)
            
            # Get or create index
            self.index = self._get_or_create_index()
            
            logger.info("PineconeIndexer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize PineconeIndexer: {e}")
            raise
    
    def _get_or_create_index(self):
        """Get existing index or create a new one"""
        index_name = self.config.PINECONE_INDEX_NAME
        
        # Check if index exists
        if index_name in self.pc.list_indexes():
            logger.info(f"Using existing index: {index_name}")
            return self.pc.Index(index_name)
        
        # Create new index
        logger.info(f"Creating new index: {index_name}")
        try:
            self.pc.create_index(
                name=index_name,
                dimension=self.config.VECTOR_DIMENSION,
                metric=self.config.METRIC,
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            
            # Wait for index to be ready
            while not self.pc.describe_index(index_name).status['ready']:
                time.sleep(1)
        except Exception as e:
            if "ALREADY_EXISTS" in str(e):
                logger.info(f"Index {index_name} already exists, using existing index")
            else:
                raise e
        
        return self.pc.Index(index_name)
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a text using OpenAI API"""
        try:
            # Truncate text if too long
            if len(text) > self.config.MAX_TEXT_LENGTH:
                text = text[:self.config.MAX_TEXT_LENGTH]
            
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            raise
    
    def prepare_document_for_indexing(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare a document for indexing by creating a comprehensive text representation
        
        Handles both content data and product data formats
        """
        
        # Combine all text content
        text_parts = []
        
        # Check if this is a product document
        if doc.get('code') and doc.get('title'):
            # This is a product document
            
            # Add product code and title
            text_parts.append(f"Product Code: {doc['code']}")
            text_parts.append(f"Product Title: {doc['title']}")
            
            # Add brand and subBrand
            if doc.get('brand'):
                text_parts.append(f"Brand: {doc['brand']}")
            if doc.get('subBrand'):
                text_parts.append(f"Sub Brand: {doc['subBrand']}")
            
            # Add product description
            if doc.get('description'):
                text_parts.append(f"Description: {doc['description']}")
            
            # Add product features (feature1 to feature10)
            features = []
            for i in range(1, 11):
                feature_key = f'feature{i}'
                if doc.get(feature_key):
                    features.append(doc[feature_key])
            if features:
                text_parts.append(f"Features: {' | '.join(features)}")
            
            # Add product keywords (handle both list and pipe-separated string)
            keywords = doc.get('keywords', [])
            if isinstance(keywords, str) and keywords:
                keywords = [k.strip() for k in keywords.split('|') if k.strip()]
            if keywords and len(keywords) > 0:
                text_parts.append(f"Keywords: {' | '.join(keywords)}")
            
            # Add product categories (handle both list and pipe-separated string)
            categories = doc.get('category', [])
            if isinstance(categories, str) and categories:
                categories = [c.strip() for c in categories.split('|') if c.strip()]
            if categories and len(categories) > 0:
                text_parts.append(f"Categories: {' | '.join(categories)}")
            
            # Add subClassName
            if doc.get('subClassName'):
                text_parts.append(f"Sub Class: {doc['subClassName']}")
            
            # Add specifications
            spec_parts = []
            if doc.get('placemakersClassification/1.0/Specification.width'):
                spec_parts.append(f"Width: {doc['placemakersClassification/1.0/Specification.width']}")
            if doc.get('placemakersClassification/1.0/Specification.height'):
                spec_parts.append(f"Height: {doc['placemakersClassification/1.0/Specification.height']}")
            if doc.get('placemakersClassification/1.0/Specification.depth'):
                spec_parts.append(f"Depth: {doc['placemakersClassification/1.0/Specification.depth']}")
            if doc.get('placemakersClassification/1.0/Specification.volume'):
                spec_parts.append(f"Volume: {doc['placemakersClassification/1.0/Specification.volume']}")
            if doc.get('placemakersClassification/1.0/Specification.warrantyduration'):
                spec_parts.append(f"Warranty: {doc['placemakersClassification/1.0/Specification.warrantyduration']}")
            if doc.get('placemakersClassification/1.0/Specification.wels'):
                spec_parts.append(f"WELS: {doc['placemakersClassification/1.0/Specification.wels']}")
            if spec_parts:
                text_parts.append(f"Specifications: {' | '.join(spec_parts)}")
            
            # Add discount price
            if doc.get('discountPrice'):
                text_parts.append(f"Price: ${doc['discountPrice']}")
            
            # Add product URL
            if doc.get('url'):
                text_parts.append(f"URL: {doc['url']}")
            
            # Combine all text
            full_text = ' '.join(text_parts)
            
            # Create metadata - convert pipe-separated strings to lists
            keywords_list = keywords if isinstance(keywords, list) else []
            categories_list = categories if isinstance(categories, list) else []
            
            metadata = {
                'product_id': doc.get('code', ''),
                'product_title': doc.get('title', ''),
                'product_url': doc.get('url', ''),
                'product_brand': doc.get('brand', ''),
                'product_subBrand': doc.get('subBrand', ''),
                'product_categories': categories_list,
                'product_keywords': keywords_list,
                'product_subClassName': doc.get('subClassName', ''),
                'product_price': str(doc.get('discountPrice', '')),
                'product_thumb_image': doc.get('thumb_image', ''),
                'spec_width': doc.get('placemakersClassification/1.0/Specification.width', ''),
                'spec_height': doc.get('placemakersClassification/1.0/Specification.height', ''),
                'spec_depth': doc.get('placemakersClassification/1.0/Specification.depth', ''),
                'spec_volume': doc.get('placemakersClassification/1.0/Specification.volume', ''),
                'spec_warranty': doc.get('placemakersClassification/1.0/Specification.warrantyduration', ''),
                'spec_wels': doc.get('placemakersClassification/1.0/Specification.wels', ''),
                'text_length': len(full_text),
                'doc_type': 'product'
            }
            
            # Add features to metadata
            for i in range(1, 11):
                feature_key = f'feature{i}'
                if doc.get(feature_key):
                    metadata[feature_key] = doc[feature_key]
            
            return {
                'id': f"product_{doc.get('code', hash(full_text))}",
                'page_content': full_text,
                'metadata': metadata
            }
        
        else:
            # This is a content document
            
            # Add page name and heading
            if doc.get('page_name'):
                text_parts.append(f"Page: {doc['page_name']}")
            if doc.get('heading'):
                text_parts.append(f"Heading: {doc['heading']}")
            
            # Add sub-headings
            if doc.get('sub_headings'):
                text_parts.append(f"Sub-headings: {' | '.join(doc['sub_headings'])}")
            
            # Add paragraphs
            if doc.get('paragraphs'):
                text_parts.append(f"Content: {' '.join(doc['paragraphs'])}")
            
            # Add products information
            if doc.get('products'):
                product_info = []
                for product in doc['products']:
                    product_text = f"{product.get('name', '')}"
                    if product.get('brand'):
                        product_text += f" (Brand: {product['brand']})"
                    if product.get('sku'):
                        product_text += f" (SKU: {product['sku']})"
                    product_info.append(product_text)
                text_parts.append(f"Products: {' | '.join(product_info)}")
            
            # Add tags
            if doc.get('tags'):
                text_parts.append(f"Tags: {' | '.join(doc['tags'])}")
            
            # Add URL
            if doc.get('url'):
                text_parts.append(f"URL: {doc['url']}")
            
            # Combine all text
            full_text = ' '.join(text_parts)
            
            # Create metadata
            metadata = {
                'page_id': doc.get('page_id', ''),
                'page_name': doc.get('page_name', ''),
                'heading': doc.get('heading', ''),
                'url': doc.get('url', ''),
                'num_products': len(doc.get('products', [])),
                'num_paragraphs': len(doc.get('paragraphs', [])),
                'num_sub_headings': len(doc.get('sub_headings', [])),
                'text_length': len(full_text),
                'doc_type': 'content'
            }
            
            return {
                'id': doc.get('page_id', f"doc_{hash(full_text)}"),
                'page_content': full_text,
                'metadata': metadata
            }
    
    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Index a list of documents into Pinecone"""
        logger.info(f"Starting to index {len(documents)} documents")
        
        # Prepare documents
        prepared_docs = []
        for doc in documents:
            try:
                prepared_doc = self.prepare_document_for_indexing(doc)
                prepared_docs.append(prepared_doc)
            except Exception as e:
                logger.error(f"Error preparing document {doc.get('page_id', 'unknown')}: {e}")
                continue
        
        logger.info(f"Prepared {len(prepared_docs)} documents for indexing")
        
        # Process in batches
        for i in range(0, len(prepared_docs), self.config.BATCH_SIZE):
            batch = prepared_docs[i:i + self.config.BATCH_SIZE]
            self._index_batch(batch, i // self.config.BATCH_SIZE + 1)
        
        logger.info("Indexing completed successfully")
    
    def _index_batch(self, batch: List[Dict[str, Any]], batch_num: int) -> None:
        """Index a batch of documents"""
        logger.info(f"Processing batch {batch_num} with {len(batch)} documents")
        
        vectors = []
        for doc in batch:
            try:
                # Get embedding
                embedding = self.get_embedding(doc['page_content'])
                
                # Create vector record with text in metadata for retrieval
                metadata = dict(doc['metadata'])
                metadata['text'] = doc['page_content']  # Add text content to metadata
                
                vector_record = {
                    'id': doc['id'],
                    'values': embedding,
                    'metadata': metadata
                }
                vectors.append(vector_record)
                
            except Exception as e:
                logger.error(f"Error processing document {doc['id']}: {e}")
                continue
        
        if vectors:
            try:
                # Upsert vectors to Pinecone
                self.index.upsert(vectors=vectors)
                logger.info(f"Successfully indexed batch {batch_num} with {len(vectors)} vectors")
            except Exception as e:
                logger.error(f"Error upserting batch {batch_num}: {e}")
                raise
        
        # Rate limiting - small delay between batches
        time.sleep(0.1)
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the index"""
        try:
            stats = self.index.describe_index_stats()
            return stats
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {}

def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """Load JSON data from file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} documents from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading JSON data from {file_path}: {e}")
        raise

def load_csv_data(file_path: str) -> List[Dict[str, Any]]:
    """Load product data from CSV file"""
    try:
        products = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                products.append(row)
        logger.info(f"Loaded {len(products)} products from {file_path}")
        return products
    except Exception as e:
        logger.error(f"Error loading CSV data from {file_path}: {e}")
        raise

def main():
    """Main function to run the indexing process"""
    try:
        # Initialize indexer
        indexer = PineconeIndexer()
        all_documents = []
        
        # Define the path to the content JSON file
        content_file_path = Path(__file__).parent.parent / "dataset" / "output" / "output-raw-page_v2.json"
        
        # Load content data if exists
        if content_file_path.exists():
            content_documents = load_json_data(str(content_file_path))
            if content_documents:
                logger.info(f"Loaded {len(content_documents)} content documents")
                all_documents.extend(content_documents)
            else:
                logger.warning("No content documents found in output-raw-page.json")
        else:
            logger.warning(f"Content JSON file not found: {content_file_path}")
        
        # Define the path to the product CSV file
        product_file_path = Path(__file__).parent.parent / "dataset" / "output" / "product_v2.csv"
        
        # Load product data if exists
        if product_file_path.exists():
            product_documents = load_csv_data(str(product_file_path))
            if product_documents:
                logger.info(f"Loaded {len(product_documents)} product documents")
                all_documents.extend(product_documents)
            else:
                logger.warning("No product documents found in product_v2.csv")
        else:
            logger.warning(f"Product CSV file not found: {product_file_path}")
        
        # Index all documents
        logger.info(f"Indexing a total of {len(all_documents)} documents")
        indexer.index_documents(all_documents)
        
        # Get and display stats
        stats = indexer.get_index_stats()
        logger.info(f"Index statistics: {stats}")
        
        logger.info("Indexing process completed successfully!")
        
    except Exception as e:
        logger.error(f"Indexing process failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
