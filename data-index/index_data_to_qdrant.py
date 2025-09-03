import json
import logging
import sys
import os
from typing import List, Dict, Any
from pathlib import Path

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from openai import OpenAI
import time

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import PointStruct, Distance, VectorParams
except ImportError:
    raise ImportError("Please install qdrant-client: pip install qdrant-client")

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

class QdrantIndexer:
    """Class to handle indexing data into Qdrant vector database"""
    def __init__(self):
        try:
            Config.validate_config()
            self.config = Config()
            # Initialize OpenAI client for embeddings
            self.openai_client = OpenAI(api_key=self.config.OPENAI_API_KEY)
            # Initialize Qdrant client (in-memory)
            self.qdrant = QdrantClient(":memory:")
            # Get or create collection
            self.collection_name = os.getenv('QDRANT_COLLECTION_NAME', 'placemakers-content')
            self._get_or_create_collection()
            logger.info("QdrantIndexer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize QdrantIndexer: {e}")
            raise

    def _get_or_create_collection(self):
        # Check if collection exists, else create
        collections = self.qdrant.get_collections().collections
        if self.collection_name not in [c.name for c in collections]:
            logger.info(f"Creating new Qdrant collection: {self.collection_name}")
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.config.VECTOR_DIMENSION,
                    distance=Distance.COSINE
                )
            )
        else:
            logger.info(f"Using existing Qdrant collection: {self.collection_name}")

    def get_embedding(self, text: str) -> List[float]:
        try:
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
        text_parts = []
        if doc.get('page_name'):
            text_parts.append(f"Page: {doc['page_name']}")
        if doc.get('heading'):
            text_parts.append(f"Heading: {doc['heading']}")
        if doc.get('sub_headings'):
            text_parts.append(f"Sub-headings: {' | '.join(doc['sub_headings'])}")
        if doc.get('paragraphs'):
            text_parts.append(f"Content: {' '.join(doc['paragraphs'])}")
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
        if doc.get('tags'):
            text_parts.append(f"Tags: {' | '.join(doc['tags'])}")
        if doc.get('url'):
            text_parts.append(f"URL: {doc['url']}")
        full_text = ' '.join(text_parts)
        metadata = {
            'page_id': doc.get('page_id', ''),
            'page_name': doc.get('page_name', ''),
            'heading': doc.get('heading', ''),
            'url': doc.get('url', ''),
            'num_products': len(doc.get('products', [])),
            'num_paragraphs': len(doc.get('paragraphs', [])),
            'num_sub_headings': len(doc.get('sub_headings', [])),
            'text_length': len(full_text)
        }
        return {
            'id': str(uuid.uuid4()),
            'page_content': full_text,
            'metadata': metadata
        }

    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        logger.info(f"Starting to index {len(documents)} documents to Qdrant")
        prepared_docs = []
        for doc in documents:
            try:
                prepared_doc = self.prepare_document_for_indexing(doc)
                prepared_docs.append(prepared_doc)
            except Exception as e:
                logger.error(f"Error preparing document {doc.get('page_id', 'unknown')}: {e}")
                continue
        logger.info(f"Prepared {len(prepared_docs)} documents for indexing")
        for i in range(0, len(prepared_docs), self.config.BATCH_SIZE):
            batch = prepared_docs[i:i + self.config.BATCH_SIZE]
            self._index_batch(batch, i // self.config.BATCH_SIZE + 1)
        logger.info("Indexing to Qdrant completed successfully")

    def _index_batch(self, batch: List[Dict[str, Any]], batch_num: int) -> None:
        logger.info(f"Processing batch {batch_num} with {len(batch)} documents")
        points = []
        for doc in batch:
            try:
                embedding = self.get_embedding(doc['page_content'])
                payload = dict(doc['metadata'])
                payload['text'] = doc['page_content']
                points.append(PointStruct(
                    id=doc['id'],
                    vector=embedding,
                    payload=payload
                ))
            except Exception as e:
                logger.error(f"Error processing document {doc['id']}: {e}")
                continue
        if points:
            try:
                self.qdrant.upsert(collection_name=self.collection_name, points=points)
                logger.info(f"Successfully indexed batch {batch_num} with {len(points)} points")
            except Exception as e:
                logger.error(f"Error upserting batch {batch_num}: {e}")
                raise
        time.sleep(0.1)

    def get_collection_stats(self) -> Dict[str, Any]:
        try:
            stats = self.qdrant.get_collection(self.collection_name)
            return stats.dict()
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}

def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} documents from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading JSON data from {file_path}: {e}")
        raise

def main():
    try:
        json_file_path = Path(__file__).parent.parent / "dataset" / "output" / "output-raw-page.json"
        if not json_file_path.exists():
            logger.error(f"JSON file not found: {json_file_path}")
            return
        documents = load_json_data(str(json_file_path))
        if not documents:
            logger.error("No documents found in JSON file")
            return
        indexer = QdrantIndexer()
        indexer.index_documents(documents)
        stats = indexer.get_collection_stats()
        logger.info(f"Collection statistics: {stats}")
        logger.info("Qdrant indexing process completed successfully!")
    except Exception as e:
        logger.error(f"Qdrant indexing process failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
