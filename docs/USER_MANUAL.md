# Legal RAG System - User Manual

## Overview

The Legal RAG (Retrieval-Augmented Generation) System is a specialized search tool for Hong Kong property law. It helps you find relevant legal provisions from ordinances including:

- **Building Management Ordinance (Cap. 344)** - IO duties, meetings, management committees
- **Buildings Ordinance (Cap. 123)** - Construction, safety, building works
- **Fire Safety Ordinances (Cap. 572, 95)** - Fire services, safety requirements
- **Town Planning Ordinance (Cap. 131)** - Zoning, development permissions
- **Landlord and Tenant Ordinance (Cap. 7)** - Tenancy rights and obligations

---

## Getting Started

### Starting the Server

```bash
cd ~/Desktop/legal_advisor_pytorch
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Accessing the System

- **Web Interface:** http://localhost:8000
- **API Documentation:** http://localhost:8000/docs

---

## Using the Web Interface

### 1. Searching for Legal Information

The web interface at http://localhost:8000 provides a full-featured search interface:

**Search Box:**
1. Enter your question in the search box (e.g., "What are the duties of the management committee?")
2. Press **Enter** or click **Search**
3. Results appear below with relevance scores

**Search Options:**
- **Results** - Choose 5, 10, or 20 results
- **Filter by Ordinance** - Limit search to a specific ordinance:
  - Cap. 344 - Building Management
  - Cap. 123 - Buildings Ordinance
  - Cap. 572 - Fire Safety (Buildings)
  - Cap. 95 - Fire Services
  - Cap. 131 - Town Planning
  - Cap. 7 - Landlord and Tenant
  - Cap. 618 - Lifts and Escalators
  - Cap. 563 - Urban Renewal Authority

**Example Queries:**
Click any example query button to instantly search:
- "Management committee duties"
- "Form IO corporation"
- "Fire safety"
- "AGM voting"

**Understanding Results:**
Each result shows:
- **Ordinance & Section** - e.g., "Cap. 344 s.40C"
- **Score** - Relevance percentage (higher = better match)
- **Hierarchy Path** - Document structure location
- **Text** - Excerpt from the provision
- **Cross-references** - Links to related sections (clickable)

### 2. Uploading Documents

**File Upload (XML/TXT):**
1. Click the "File Upload" tab
2. Drag and drop your file or click to browse
3. Optionally enter a custom ID
4. Click "Upload File"

**Text Input:**
1. Click the "Text Input" tab
2. Enter the document title
3. Select document type (Regulatory Statement, Guideline, Notice, etc.)
4. Paste the content
5. Click "Upload Text"

---

## Using the API

### Query the Knowledge Base

Send a POST request to search for relevant legal provisions:

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the duties of the management committee?",
    "top_k": 5
  }'
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | required | Your search question |
| `top_k` | integer | 10 | Number of results to return |
| `filter_ordinance` | string | null | Filter by ordinance (e.g., "Cap. 344") |
| `expand_context` | boolean | true | Include related sections and definitions |

**Example Response:**
```json
{
  "query": "What are the duties of the management committee?",
  "results": [
    {
      "chunk_id": "cap344_s40C",
      "score": 0.652,
      "ordinance": "Cap. 344",
      "section": "40C.",
      "hierarchy_path": "Cap. 344 > Miscellaneous > Appointment of management committee",
      "text": "40C. Appointment of management committee...",
      "cross_references": [...]
    }
  ],
  "total": 5
}
```

### Check Index Status

```bash
curl http://localhost:8000/api/index/status
```

**Response:**
```json
{
  "index_directory": "/path/to/data/index",
  "chunk_files": 48,
  "total_chunks": 8814,
  "retriever_loaded": true,
  "knowledge_graph_loaded": true
}
```

### List Uploaded Documents

```bash
curl http://localhost:8000/api/uploads
```

### Upload a File

```bash
curl -X POST http://localhost:8000/api/upload/file \
  -F "file=@/path/to/ordinance.xml"
```

### Upload Text

```bash
curl -X POST http://localhost:8000/api/upload/text \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Building Safety Guidelines 2024",
    "text": "1. All buildings must comply with...",
    "doc_type": "guideline"
  }'
```

### Delete a Document

```bash
curl -X DELETE http://localhost:8000/api/uploads/{ordinance_id}
```

### Rebuild Index

```bash
curl -X POST http://localhost:8000/api/index/rebuild
```

---

## Sample Queries

Here are some example queries to try:

| Query | Relevant Ordinances |
|-------|---------------------|
| "What are the duties of the management committee?" | Cap. 344 |
| "How to form an incorporated owners corporation?" | Cap. 344 |
| "Fire safety requirements for buildings" | Cap. 572, Cap. 95 |
| "Building inspection and repair obligations" | Cap. 123 |
| "Tenant rights for termination of lease" | Cap. 7 |
| "Town planning application process" | Cap. 131 |
| "Lift and escalator safety regulations" | Cap. 618 |
| "Annual general meeting requirements" | Cap. 344 |
| "Common parts maintenance responsibility" | Cap. 344 |
| "Penalty for fire safety violations" | Cap. 95 |

---

## Understanding Results

### Score
- Ranges from 0 to 1
- Higher score = more relevant
- Scores above 0.6 are typically good matches

### Hierarchy Path
Shows the document structure:
```
Cap. 344 > Part III > Meetings > Annual General Meeting
```

### Cross References
Links to related provisions in other sections or ordinances.

---

## Tips for Better Results

1. **Be Specific** - Include key legal terms like "incorporated owners", "management committee", "DMC"

2. **Use Ordinance Names** - Mention "Building Management Ordinance" or "Cap. 344" for targeted results

3. **Ask Questions** - Natural language queries work well:
   - "What is the quorum for an AGM?"
   - "How many votes are needed to pass a resolution?"

4. **Filter by Ordinance** - Use `filter_ordinance` parameter to search within a specific ordinance:
   ```json
   {"query": "meeting requirements", "filter_ordinance": "Cap. 344"}
   ```

5. **Expand Context** - Keep `expand_context: true` to include parent sections and relevant definitions

---

## Troubleshooting

### Server Won't Start
```bash
# Check if port 8000 is in use
lsof -i :8000

# Kill existing process if needed
kill -9 <PID>
```

### No Results Returned
- Check if the index is built: `curl http://localhost:8000/api/index/status`
- Rebuild if needed: `python scripts/build_index.py`

### Slow First Query
The first query loads the embedding model and index into memory. Subsequent queries will be faster.

### Connection Refused
Ensure the server is running:
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

---

## Data Coverage

### Included Ordinances (47 files)

| Category | Ordinances |
|----------|------------|
| Building Management | Cap. 344, 344A, 344B |
| Buildings | Cap. 123, 123A-P |
| Fire Safety | Cap. 572, 95, 95A-H |
| Urban Renewal | Cap. 563 |
| Town Planning | Cap. 131, 131A-C |
| Landlord & Tenant | Cap. 7, 7A |
| Lifts & Escalators | Cap. 618, 618A-B |
| Energy Efficiency | Cap. 610, 610A-B |

### Statistics
- **Total Chunks:** ~8,800
- **Definitions:** ~830
- **Cross-References:** ~3,200

---

## Support

For technical issues or feature requests, please contact the system administrator or refer to the developer documentation in `CLAUDE.md`.

---

*Last Updated: January 2026*
