# Analyst - Semantic Search with Jina Embeddings

This project provides semantic search functionality for Hugging Face datasets using the Jina embeddings v2 base EN model. The system allows you to search through a curated dataset of Hugging Face datasets using natural language queries.

## Features

- **Semantic Search**: Find relevant datasets using natural language queries
- **Category Filtering**: Search within specific task categories
- **Similar Dataset Discovery**: Find datasets similar to a given dataset
- **Fast Vector Search**: Uses FAISS for efficient similarity search
- **Rich Metadata**: Access to dataset information including downloads, likes, authors, and categories

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have the dataset file `dataset_semantic.parquet` in the project directory.

## Usage

### Quick Start

Run the demo to see the semantic search in action:
```bash
python demo_search.py
```

### Interactive Search

Run the interactive search interface:
```bash
python main.py search
```

Or use the standalone search interface:
```bash
python semantic_search.py
```

### Programmatic Usage

```python
from semantic_search import SemanticSearchEngine

# Initialize the search engine
engine = SemanticSearchEngine()
engine.initialize()

# Search for datasets
results = engine.search("machine learning datasets", top_k=5)

# Search within a specific category
results = engine.search_by_category("medical data", "professional_medicine", top_k=3)

# Find similar datasets
similar = engine.get_similar_datasets("argilla/databricks-dolly-15k-curated-en", top_k=5)

# Get dataset information
info = engine.get_dataset_info()
```

## Dataset Structure

The `dataset_semantic.parquet` file contains the following columns:

- `datasetId`: Unique identifier for the dataset
- `author`: Dataset author/organization
- `last_modified`: Last modification timestamp
- `downloads`: Number of downloads
- `likes`: Number of likes
- `tags`: Dataset tags
- `task_categories`: List of task categories
- `createdAt`: Creation timestamp
- `card`: Dataset card information
- `embedding`: Pre-computed Jina embeddings (768-dimensional vectors)

## Search Capabilities

### Basic Search
Search for datasets using natural language:
```python
results = engine.search("computer vision datasets for object detection")
```

### Category Filtering
Search within specific categories:
```python
results = engine.search_by_category("medical data", "professional_medicine")
```

### Similar Dataset Discovery
Find datasets similar to a known dataset:
```python
similar = engine.get_similar_datasets("microsoft/DialoGPT-medium")
```

### Advanced Filtering
Filter results by minimum similarity score:
```python
results = engine.search("NLP", min_score=0.7)
```

## Performance

- **Initialization**: ~2-5 seconds (loads model and builds FAISS index)
- **Search Speed**: ~10-50ms per query (depending on query complexity)
- **Memory Usage**: ~500MB-1GB (depending on dataset size)

## Available Categories

The dataset includes datasets from various categories such as:
- Machine Learning
- Natural Language Processing
- Computer Vision
- Medical/Healthcare
- Legal
- Finance
- Social Sciences
- And many more...

## Technical Details

- **Embedding Model**: `jinaai/jina-embeddings-v2-base-en`
- **Vector Search**: FAISS with cosine similarity
- **Embedding Dimension**: 768
- **Index Type**: FlatIP (Inner Product) for cosine similarity

## File Structure

```
Analyst/
├── main.py                 # Main entry point with search demo
├── semantic_search.py      # Core semantic search engine
├── demo_search.py         # Demo script
├── requirements.txt       # Python dependencies
├── dataset_semantic.parquet # Dataset with embeddings
├── stage_1.py            # Original stage 1 code
├── stage_2.py            # Original stage 2 code
└── README.md             # This file
```

## Examples

### Example 1: Finding ML Datasets
```python
results = engine.search("machine learning datasets for beginners")
for result in results:
    print(f"{result['datasetId']} - {result['author']}")
    print(f"Downloads: {result['downloads']:,}, Score: {result['score']:.3f}")
```

### Example 2: Medical Dataset Search
```python
results = engine.search_by_category("medical imaging", "professional_medicine")
```

### Example 3: Finding Similar Datasets
```python
similar = engine.get_similar_datasets("huggingface/CodeBERTa")
```

## Troubleshooting

### Common Issues

1. **Model Loading Error**: Ensure you have internet connection for first-time model download
2. **Memory Issues**: The system requires sufficient RAM for the embeddings and FAISS index
3. **Dataset Not Found**: Ensure `dataset_semantic.parquet` is in the project directory

### Performance Tips

1. The FAISS index is cached after first build for faster subsequent loads
2. Use specific queries for better results
3. Filter by categories when possible to narrow down results

## License

This project uses the Jina embeddings model and follows the respective licensing terms.
# analyst
