# Pinecone Vector Database Indexing

This directory contains scripts to index JSON data into a Pinecone vector database for semantic search capabilities.

## Setup Instructions

### 1. Install Dependencies

First, install the required Python packages:

```bash
pip install -r requirements.txt
```

### 2. Set Up API Keys

Create a `.env` file in the root directory with your API keys:

```bash
# Copy the example file
cp env_example.txt .env
```

Then edit the `.env` file and add your actual API keys:

```env
# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=gcp-starter
PINECONE_INDEX_NAME=placemakers-content

# OpenAI Configuration (for embeddings)
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Get API Keys

#### Pinecone API Key
1. Go to [Pinecone Console](https://app.pinecone.io/)
2. Sign up or log in
3. Create a new project or use existing one
4. Copy your API key from the API Keys section
5. Note your environment (e.g., `gcp-starter`, `us-west1-gcp`, etc.)

#### OpenAI API Key
1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Sign up or log in
3. Go to API Keys section
4. Create a new API key
5. Copy the key (it starts with `sk-`)

## Usage

### Index Data

To index the JSON data into Pinecone:

```bash
cd data-index
python index_data_to_db.py
```

This will:
- Load the JSON data from `../dataset/output/output-raw-page.json`
- Create embeddings using OpenAI's text-embedding-3-large model
- Index the data into Pinecone with metadata
- Process data in batches for efficiency
- Log the progress to both console and `indexing.log` file

### Query Index

To test the indexed data and run example queries:

```bash
cd data-index
python query_index.py
```

This will:
- Show index statistics
- Run example queries
- Display search results with relevance scores
- Show sample vectors in the index

## Configuration

The configuration is managed in `../config.py`. Key settings include:

- `VECTOR_DIMENSION`: 1024 (for OpenAI text-embedding-3-large)
- `BATCH_SIZE`: 100 (documents processed per batch)
- `MAX_TEXT_LENGTH`: 8000 (maximum text length for embedding)

## Data Structure

The JSON data is expected to have the following structure:

```json
{
  "page_id": "unique_id",
  "page_name": "Page Title",
  "heading": "Main Heading",
  "sub_headings": ["Sub heading 1", "Sub heading 2"],
  "paragraphs": ["Paragraph 1", "Paragraph 2"],
  "products": [
    {
      "sku": "123456",
      "name": "Product Name",
      "brand": "Brand Name"
    }
  ],
  "tags": ["tag1", "tag2"],
  "url": "https://example.com/page"
}
```

## Indexed Metadata

Each vector in Pinecone includes metadata for filtering and retrieval:

- `page_id`: Unique identifier
- `page_name`: Page title
- `heading`: Main heading
- `url`: Page URL
- `num_products`: Number of products
- `num_paragraphs`: Number of paragraphs
- `num_sub_headings`: Number of sub-headings
- `text_length`: Total text length

## Error Handling

The scripts include comprehensive error handling:

- API key validation
- Network error handling
- Rate limiting
- Batch processing with retry logic
- Detailed logging

## Logging

All operations are logged to:
- Console output
- `indexing.log` file (for indexing operations)

## Troubleshooting

### Common Issues

1. **Missing API Keys**: Ensure both Pinecone and OpenAI API keys are set in `.env`
2. **Index Not Found**: The script will create a new index if it doesn't exist
3. **Rate Limiting**: The script includes delays between batches to respect API limits
4. **Large Data**: For very large datasets, consider increasing `BATCH_SIZE` in config

### Check Index Status

To check if your index exists and get statistics:

```python
from config import Config
import pinecone

pinecone.init(api_key=Config.PINECONE_API_KEY, environment=Config.PINECONE_ENVIRONMENT)
indexes = pinecone.list_indexes()
print(f"Available indexes: {indexes}")
```

## Cost Considerations

- **OpenAI**: Charges per token for embeddings (~$0.0001 per 1K tokens)
- **Pinecone**: Free tier includes 1 index with 100K vectors, then pay-per-use

Monitor your usage in both platforms' dashboards.
