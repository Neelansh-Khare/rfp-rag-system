# GCP Vertex AI Multimodal RAG System

Production-ready Retrieval-Augmented Generation (RAG) system for processing and querying multimodal RFP and proposal documents using Google Cloud Platform's Vertex AI.

## Features

- **Multimodal Processing**: Extracts text, tables, and images from PDFs
- **Vertex AI Vector Search**: Scalable semantic search with filtering
- **Gemini Vision**: Image analysis using GCP-native Gemini (no external APIs)
- **Smart Filtering**: Filter by project, document type, and content type
- **Answer Generation**: RAG-based answers using Gemini 1.5 Flash

## Quick Setup

### 1. Prerequisites

- Google Cloud Project with Vertex AI enabled
- Authenticated `gcloud` CLI: `gcloud auth application-default login`
- Python 3.9+

### 2. Install Dependencies

```bash
pip install -r requirements_gcp_multimodal.txt
```

### 3. Configure Environment

Create `.env` file:

```bash
BUCKET_NAME="your-bucket-name"
INDEX_ID="projects/YOUR_PROJECT_NUM/locations/REGION/indexes/INDEX_NUM"
ENDPOINT_ID="projects/YOUR_PROJECT_NUM/locations/REGION/indexEndpoints/ENDPOINT_NUM"
```

### 4. Add Your Documents

Organize PDFs in `input/` directory:

```
input/
├── rfps/
│   ├── project1_rfp.pdf
│   └── project2_rfp.pdf
└── proposals/
    ├── project1_proposal.pdf
    └── project2_proposal.pdf
```

### 5. Ingest Documents

```bash
# Dry run (preview what will be processed)
python ingest_classified_pairs.py

# Actually process and index
python ingest_classified_pairs.py --process
```

Wait 5-15 minutes for Vertex AI index to update after ingestion.

### 6. Query Your Data

```bash
# Run working examples
python working_examples.py
```

## Usage

### Basic Querying

```python
from gcp_rag_multimodal_vertex import VertexAIMultimodalRAG
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize
rag = VertexAIMultimodalRAG(
    bucket_name=os.getenv("BUCKET_NAME"),
    index_id=os.getenv("INDEX_ID"),
    endpoint_id=os.getenv("ENDPOINT_ID"),
    enable_vision_ai=True  # Uses GCP Gemini Vision
)

# Search across all documents
results = rag.query("public engagement requirements", top_k=5)

# Filter by document type
rfp_results = rag.query("budget", filters={"document_type": "rfp"})

# Filter by project
project_results = rag.query("timeline", filters={"project_id": "project7"})

# Filter by content type
table_results = rag.query("cost breakdown", filters={"content_type": "table"})

# Combine filters
results = rag.query(
    "deliverables",
    filters={
        "project_id": "project3",
        "document_type": "rfp"
    }
)
```

### Generate Answers with Gemini

```python
# Generate answer using RAG context
answer = rag.generate_answer(
    "What are the common engagement approaches?",
    top_k=10
)

print(answer["answer"])
print(f"Sources: {len(answer['sources'])}")
```

## Managing Projects

### List All Projects

```bash
python remove_project.py list
```

### Remove a Project

```bash
# Preview what will be removed
python remove_project.py project1

# Actually remove and archive
python remove_project.py project1 --execute

# Clear GCS data and re-ingest
python remove_project.py project1 --clear-gcs
python ingest_classified_pairs.py --process
```

**Note:** Vertex AI Vector Search doesn't support individual vector deletion. To remove a project:
1. Archive the project files: `python remove_project.py project1 --execute`
2. Clear all embeddings: `gsutil -m rm -r gs://BUCKET_NAME/embeddings/`
3. Re-ingest remaining projects: `python ingest_classified_pairs.py --process`
4. Wait 5-15 minutes for index update

## Architecture

| Component | Technology |
|-----------|------------|
| Vector Database | Vertex AI Vector Search (TreeAH index) |
| Embeddings | text-embedding-004 (768 dimensions) |
| Storage | Google Cloud Storage |
| Vision AI | Gemini 1.5 Flash (multimodal) |
| Answer Generation | Gemini 1.5 Flash |
| Region | northamerica-northeast1 |

## Project Structure

```
ragProcessing/
├── gcp_rag_multimodal_vertex.py    # Core RAG implementation
├── ingest_classified_pairs.py       # Batch ingestion script
├── remove_project.py                # Project removal tool
├── working_examples.py              # Quick query examples
├── example_queries.py               # Comprehensive examples
├── requirements_gcp_multimodal.txt  # Dependencies
├── .env                             # Configuration
└── input/                           # Your documents
    ├── rfps/                        # RFP PDFs
    ├── proposals/                   # Proposal PDFs
    └── archived/                    # Removed projects
```

## Key Capabilities

### Multimodal Extraction

**Text:** Semantic chunking with overlap, preserves document structure

**Tables:** Multiple extraction methods (Camelot, Tabula, PyMuPDF), converted to markdown

**Images:** OCR with Tesseract, Gemini Vision analysis for charts/diagrams

### Advanced Filtering

- `project_id`: Filter by specific project
- `document_type`: `"rfp"` or `"proposal"`
- `content_type`: `"text"`, `"table"`, or `"image"`

Filters can be combined for precise queries.

## Troubleshooting

**No results returned?**
- Wait 5-15 minutes after ingestion for index update
- Verify `.env` configuration
- Test without filters: `rag.query("test", top_k=1)`

**Slow queries?**
- Reduce `top_k` parameter
- Use specific filters to narrow search
- Metadata retrieval searches GCS files (can be slow)

**Import errors?**
- Install dependencies: `pip install -r requirements_gcp_multimodal.txt`
- Verify Python 3.9+ is installed

**Authentication errors?**
- Run: `gcloud auth application-default login`
- Ensure GCP project has Vertex AI enabled

## Performance

- **Query Speed**: 1-2 seconds for vector search
- **Index Updates**: 5-15 minutes after new data upload
- **Embedding Generation**: ~1 second per chunk
- **Current Index Size**: 5,717+ vectors from 13 project pairs

## Examples

See [working_examples.py](working_examples.py) for ready-to-run examples:

```bash
python working_examples.py
```

Examples include:
1. Search across all projects
2. Filter by RFPs only
3. Filter by specific project
4. Search tables only

## Additional Documentation

- **NEXT_STEPS.md**: Comprehensive usage guide
- **CLAUDE.md**: AI assistant instructions for large codebase analysis

## License

Proprietary - NeuraCities

## Support

- GCP Console: [Vertex AI Indexes](https://console.cloud.google.com/vertex-ai/matching-engine/indexes?project=neuracities-pilot)
- Cloud Storage: [Bucket Browser](https://console.cloud.google.com/storage/browser)
