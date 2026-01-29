# CLAUDE.md - Project Guide for AI Assistants

## Project Overview

**Legal RAG System for Hong Kong Property Law**

A Retrieval-Augmented Generation system specialized for Hong Kong property law, focusing on the rights and obligations of Incorporated Owners (IO) and individual property owners. Built with PyTorch and designed for high accuracy in legal document retrieval and response generation.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Build the index (first time)
python scripts/build_index.py --data-dir data/property_owner_ordinances

# Start the web UI
uvicorn src.api.main:app --reload --port 8000

# Upload documents via CLI
python scripts/upload_ordinance.py --help
```

## Project Structure

```
legal_advisor_pytorch/
├── data/
│   ├── hkel_legal_import/           # Full HKEL dataset (3,152 ordinances)
│   ├── property_owner_ordinances/   # Narrowed dataset (47 ordinances)
│   │   ├── cap_344_en_c/            # Building Management Ordinance
│   │   ├── cap_123_en_c/            # Buildings Ordinance
│   │   ├── cap_572_en_c/            # Fire Safety (Buildings) Ordinance
│   │   └── ...
│   └── index/                       # Generated index files
│       ├── chunks/                  # JSON chunk files
│       ├── vector_index/            # Embeddings
│       └── knowledge_graph/         # Cross-reference graph
├── src/
│   ├── preprocessing/
│   │   ├── hkel_parser.py          # XML parser for HKEL format
│   │   ├── chunker.py              # Hierarchical legal chunking
│   │   └── upload_handler.py       # Document upload processing
│   ├── retrieval/
│   │   ├── hybrid_retriever.py     # Dense + sparse retrieval
│   │   └── knowledge_graph.py      # Cross-reference resolution
│   └── api/
│       └── main.py                 # FastAPI web server
├── scripts/
│   ├── build_index.py              # Build full index
│   └── upload_ordinance.py         # CLI upload tool
├── docs/
│   └── rag_architecture_plan.md    # Detailed architecture documentation
└── requirements.txt
```

## Key Concepts

### Dataset
- **HKEL Format**: Hong Kong e-Legislation XML schema
- **47 Ordinances** covering: Building Management (Cap. 344), Buildings Ordinance (Cap. 123), Fire Safety (Cap. 572, 95), Urban Renewal (Cap. 563), Town Planning (Cap. 131)
- **Bilingual**: English with Chinese term annotations

### Chunking Strategy
- **Section-based**: Never breaks legal semantic units
- **Hierarchical**: Ordinance > Part > Section > Subsection > Paragraph
- **Metadata-rich**: Every chunk includes full citation path, cross-references, definitions

### Retrieval
- **Hybrid**: Dense (BGE-M3 embeddings) + Sparse (BM25)
- **Knowledge Graph**: Cross-reference resolution via NetworkX
- **Context Expansion**: Automatically includes parent sections and definitions

## Common Tasks

### Adding a New Ordinance
```bash
# Via CLI
python scripts/upload_ordinance.py --file path/to/cap_999.xml

# Via Web UI
# Go to http://localhost:8000 and drag-drop the file
```

### Rebuilding the Index
```bash
python scripts/build_index.py --data-dir data/property_owner_ordinances
# Or via CLI:
python scripts/upload_ordinance.py --rebuild-index
```

### Querying (API)
```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the duties of management committee?", "top_k": 5}'
```

## Code Patterns

### Parsing XML Ordinances
```python
from src.preprocessing import HKELParser, parse_ordinance_directory

# Single file
parser = HKELParser()
ordinance = parser.parse_file(Path("cap_344.xml"))

# Directory
ordinances = parse_ordinance_directory(Path("data/property_owner_ordinances"))
```

### Creating Chunks
```python
from src.preprocessing import LegalChunker

chunker = LegalChunker(max_chunk_size=2000)
chunks = chunker.chunk_ordinance(ordinance)
```

### Retrieval
```python
from src.retrieval import HybridRetriever

retriever = HybridRetriever(model_name="BAAI/bge-base-en-v1.5")
retriever.index_chunks(chunks)
results = retriever.retrieve("incorporated owners duties", top_k=10)
```

## Important Files

| File | Purpose |
|------|---------|
| `src/preprocessing/hkel_parser.py` | Core XML parsing logic |
| `src/preprocessing/chunker.py` | Chunking with legal hierarchy preservation |
| `src/retrieval/hybrid_retriever.py` | Main retrieval implementation |
| `src/api/main.py` | Web UI and REST API |
| `docs/rag_architecture_plan.md` | Full system architecture documentation |

## Dependencies

- **PyTorch**: Core ML framework
- **sentence-transformers**: Embedding models (BGE-M3)
- **rank-bm25**: Sparse retrieval
- **networkx**: Knowledge graph
- **FastAPI**: Web API

## Notes for Development

1. **Accuracy is Priority**: This is a legal system - always preserve citations and cross-references
2. **Hierarchical Structure**: Legal documents have strict hierarchy - never flatten it
3. **Cross-References**: Laws reference each other frequently - the knowledge graph handles this
4. **Bilingual Terms**: Many legal terms have both English and Chinese - preserve both
5. **Temporal Validity**: Laws change - track `startPeriod` and `status` attributes

## Environment Variables

Create `.env` file for configuration:
```
EMBEDDING_MODEL=BAAI/bge-m3
DATA_DIR=data/property_owner_ordinances
INDEX_DIR=data/index
```

## Testing

```bash
pytest tests/
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/api/upload/file` | POST | Upload ordinance file |
| `/api/upload/text` | POST | Upload text document |
| `/api/uploads` | GET | List uploads |
| `/api/uploads/{id}` | DELETE | Delete upload |
| `/api/query` | POST | Query the RAG system |
| `/api/index/status` | GET | Index statistics |
| `/api/index/rebuild` | POST | Rebuild index |

## Incident Log & Fixes (2026-01-29)

### Issue 1: XML Parsing Error - `prefix 'xml' not found in prefix map`
**Symptom:** Most HKEL XML files failed to parse with namespace errors.

**Root Cause:** The HKEL XML files use `xml:lang` attributes (e.g., `xml:lang="en"`, `xml:lang="zh-Hant-HK"`), but the parser's namespace dictionary didn't include the reserved `xml` namespace.

**Fix:** Added `xml` and `xsi` namespaces to `src/preprocessing/hkel_parser.py`:
```python
NAMESPACES = {
    'hklm': 'http://www.xml.gov.hk/schemas/hklm/1.0',
    'dc': 'http://purl.org/dc/elements/1.1/',
    'dcterms': 'http://purl.org/dc/terms/',
    'xhtml': 'http://www.w3.org/1999/xhtml',
    'xml': 'http://www.w3.org/XML/1998/namespace',  # Added
    'xsi': 'http://www.w3.org/2001/XMLSchema-instance'  # Added
}
```

### Issue 2: Empty Tensor Normalization Error
**Symptom:** `IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)` when indexing 0 chunks.

**Root Cause:** `_generate_embeddings()` in `hybrid_retriever.py` called `F.normalize(embeddings, p=2, dim=1)` on an empty tensor.

**Fix:** Added empty input check in `src/retrieval/hybrid_retriever.py`:
```python
def _generate_embeddings(self, texts: List[str]) -> torch.Tensor:
    if not texts:
        return torch.empty(0, self.model.get_sentence_embedding_dimension())
    # ... rest of function
```

### Issue 3: API Re-indexing on Every Query
**Symptom:** First query took forever and eventually timed out; server was re-generating embeddings for all 8,814 chunks.

**Root Cause:** `_load_retriever()` in `src/api/main.py` always rebuilt the index from chunks instead of loading the pre-built embeddings from `data/index/vector_index/`.

**Fix:** Updated `_load_retriever()` to load pre-built index first:
```python
def _load_retriever(force_rebuild: bool = False):
    vector_index_path = INDEX_DIR / "vector_index"

    # Try to load pre-built index first
    if not force_rebuild and vector_index_path.exists():
        retriever = HybridRetriever(model_name="BAAI/bge-base-en-v1.5")
        retriever.load_index(vector_index_path)
        return retriever, knowledge_graph
    # ... fallback to rebuild
```

### Issue 4: Empty Data Directory
**Symptom:** `property_owner_ordinances/` contained only MANIFEST.md, no actual XML files.

**Root Cause:** The narrowed dataset needed to be populated from `hkel_legal_import/`.

**Fix:** Copied 47 relevant ordinances from `hkel_legal_import/` to `property_owner_ordinances/`:
```bash
# Example: copy Building Management Ordinance
cp -r data/hkel_legal_import/cap_344_en_c data/property_owner_ordinances/
```

### Issue 5: Web UI Missing Query Interface (2026-01-29)
**Symptom:** Users could only upload documents via the web UI; querying required using curl or the API directly.

**Fix:** Added a complete query interface to the web UI (`src/api/main.py`):
- Search box with Enter key support
- Filter by number of results (5/10/20)
- Filter by ordinance (Cap. 344, 123, 572, etc.)
- Example query buttons for quick testing
- Results display with scores, hierarchy paths, and cross-references

### Current Status (Post-Fix)
- **49 ordinances** successfully parsed
- **4,470 chunks** indexed (property_owner_ordinances dataset)
- **8,814 chunks** total (including all uploaded documents)
- **Knowledge graph:** 4,442 nodes, 3,254 edges
- **Embedding model:** BAAI/bge-base-en-v1.5
- **Server:** Running on http://localhost:8000
- **Web UI:** Full query and upload interface
