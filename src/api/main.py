"""
Legal RAG API Server

FastAPI-based web server for:
- Uploading ordinances and regulatory statements
- Querying the legal knowledge base
- Managing the document index
"""

import sys
from pathlib import Path
from typing import List, Optional
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json

from preprocessing.upload_handler import OrdinanceUploadHandler, DynamicIndexUpdater


# Configuration
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "property_owner_ordinances"
INDEX_DIR = Path(__file__).parent.parent.parent / "data" / "index"

# Initialize components
upload_handler = OrdinanceUploadHandler(data_dir=DATA_DIR, index_dir=INDEX_DIR)
index_updater = DynamicIndexUpdater(index_dir=INDEX_DIR)

# Optional: Initialize retriever and knowledge graph (lazy loading)
retriever = None
knowledge_graph = None

# FastAPI app
app = FastAPI(
    title="Legal RAG System",
    description="Upload and query Hong Kong property law ordinances",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class TextUploadRequest(BaseModel):
    text: str
    title: str
    doc_type: str = "regulatory_statement"
    metadata: Optional[dict] = None


class QueryRequest(BaseModel):
    query: str
    top_k: int = 10
    filter_ordinance: Optional[str] = None
    expand_context: bool = True


class UploadResponse(BaseModel):
    success: bool
    ordinance_id: str
    ordinance_name: str
    chunks_created: int
    cross_references_found: int
    definitions_extracted: int
    warnings: List[str] = []
    error_message: Optional[str] = None


# API Endpoints

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main upload UI"""
    return get_upload_page_html()


@app.post("/api/upload/file", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    custom_id: Optional[str] = Form(None)
):
    """
    Upload an ordinance file (XML or TXT).

    - **file**: The ordinance file (XML in HKEL format or plain text)
    - **custom_id**: Optional custom identifier for the document
    """
    content = await file.read()

    result = upload_handler.upload_file(
        file_content=content,
        filename=file.filename,
        content_type=file.content_type,
        custom_id=custom_id
    )

    if not result.success:
        raise HTTPException(status_code=400, detail=result.error_message)

    # Update index if retriever is loaded
    if retriever is not None:
        _update_index_with_upload(result.ordinance_id)

    return UploadResponse(**result.to_dict())


@app.post("/api/upload/text", response_model=UploadResponse)
async def upload_text(request: TextUploadRequest):
    """
    Upload a plain text regulatory statement.

    - **text**: The regulatory statement text
    - **title**: Document title
    - **doc_type**: Type (regulatory_statement, guideline, notice, etc.)
    - **metadata**: Optional additional metadata
    """
    result = upload_handler.upload_text(
        text=request.text,
        title=request.title,
        doc_type=request.doc_type,
        metadata=request.metadata
    )

    if not result.success:
        raise HTTPException(status_code=400, detail=result.error_message)

    # Update index if retriever is loaded
    if retriever is not None:
        _update_index_with_upload(result.ordinance_id)

    return UploadResponse(**result.to_dict())


@app.get("/api/uploads")
async def list_uploads():
    """List all uploaded documents"""
    uploads = upload_handler.list_uploads()
    return {"uploads": uploads, "total": len(uploads)}


@app.delete("/api/uploads/{ordinance_id}")
async def delete_upload(ordinance_id: str):
    """Delete an uploaded document"""
    success = upload_handler.delete_upload(ordinance_id)

    if not success:
        raise HTTPException(status_code=404, detail="Upload not found")

    # Rebuild index if retriever is loaded
    if retriever is not None:
        _rebuild_index()

    return {"success": True, "message": f"Deleted {ordinance_id}"}


@app.post("/api/query")
async def query_documents(request: QueryRequest):
    """
    Query the legal knowledge base.

    - **query**: Search query
    - **top_k**: Number of results to return
    - **filter_ordinance**: Optional filter by ordinance (e.g., "Cap. 344")
    - **expand_context**: Whether to include parent sections and definitions
    """
    global retriever, knowledge_graph

    # Lazy load retriever
    if retriever is None:
        retriever, knowledge_graph = _load_retriever()

    if retriever is None:
        raise HTTPException(
            status_code=503,
            detail="Index not built. Please build the index first."
        )

    results = retriever.retrieve(
        query=request.query,
        top_k=request.top_k,
        filter_ordinance=request.filter_ordinance
    )

    if request.expand_context:
        results = retriever.expand_context(results)

    return {
        "query": request.query,
        "results": [
            {
                "chunk_id": r.chunk_id,
                "score": r.score,
                "ordinance": r.ordinance,
                "section": r.section,
                "hierarchy_path": r.hierarchy_path,
                "text": r.text,
                "cross_references": r.cross_references
            }
            for r in results
        ],
        "total": len(results)
    }


@app.get("/api/index/status")
async def index_status():
    """Get the status of the index"""
    chunks_dir = INDEX_DIR / "chunks"

    chunk_files = list(chunks_dir.glob("*_chunks.json")) if chunks_dir.exists() else []
    total_chunks = 0

    for chunk_file in chunk_files:
        with open(chunk_file) as f:
            chunks = json.load(f)
            total_chunks += len(chunks)

    return {
        "index_directory": str(INDEX_DIR),
        "chunk_files": len(chunk_files),
        "total_chunks": total_chunks,
        "retriever_loaded": retriever is not None,
        "knowledge_graph_loaded": knowledge_graph is not None
    }


@app.post("/api/index/rebuild")
async def rebuild_index():
    """Rebuild the entire index"""
    global retriever, knowledge_graph

    try:
        retriever, knowledge_graph = _load_retriever(force_rebuild=True)
        return {"success": True, "message": "Index rebuilt successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions

def _load_retriever(force_rebuild: bool = False):
    """Load or create the retriever and knowledge graph"""
    from retrieval import HybridRetriever, LegalKnowledgeGraph

    vector_index_path = INDEX_DIR / "vector_index"
    kg_path = INDEX_DIR / "knowledge_graph"

    # Try to load pre-built index first
    if not force_rebuild and vector_index_path.exists() and (vector_index_path / "embeddings.pt").exists():
        retriever = HybridRetriever(model_name="BAAI/bge-base-en-v1.5")
        retriever.load_index(vector_index_path)

        knowledge_graph = LegalKnowledgeGraph()
        if kg_path.exists():
            knowledge_graph.load(kg_path)

        return retriever, knowledge_graph

    # Fall back to rebuilding from chunks
    chunks = index_updater.get_all_chunks()

    if not chunks:
        return None, None

    retriever = HybridRetriever(model_name="BAAI/bge-base-en-v1.5")
    knowledge_graph = LegalKnowledgeGraph()

    retriever.index_chunks(chunks, save_path=vector_index_path)
    knowledge_graph.build_from_chunks(chunks)
    knowledge_graph.save(kg_path)

    return retriever, knowledge_graph


def _update_index_with_upload(ordinance_id: str):
    """Update index with newly uploaded document"""
    global retriever, knowledge_graph

    chunks_file = INDEX_DIR / "chunks" / f"{ordinance_id}_chunks.json"
    if chunks_file.exists():
        with open(chunks_file) as f:
            new_chunks = json.load(f)

        index_updater.update_index(retriever, knowledge_graph, new_chunks)


def _rebuild_index():
    """Rebuild the full index"""
    global retriever, knowledge_graph

    if retriever is not None:
        index_updater.rebuild_full_index(retriever, knowledge_graph)


def get_upload_page_html() -> str:
    """Return the HTML for the upload page"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Legal RAG - Upload Ordinances</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #e0e0e0;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        header {
            text-align: center;
            padding: 40px 0;
        }
        h1 {
            font-size: 2.5rem;
            color: #4fc3f7;
            margin-bottom: 10px;
        }
        .subtitle {
            color: #90a4ae;
            font-size: 1.1rem;
        }
        .card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 30px;
            margin-bottom: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .card h2 {
            color: #4fc3f7;
            margin-bottom: 20px;
            font-size: 1.3rem;
        }
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        .tab {
            padding: 12px 24px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            color: #90a4ae;
            cursor: pointer;
            transition: all 0.3s;
        }
        .tab:hover, .tab.active {
            background: #4fc3f7;
            color: #1a1a2e;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .upload-area {
            border: 2px dashed rgba(79, 195, 247, 0.3);
            border-radius: 12px;
            padding: 60px 30px;
            text-align: center;
            transition: all 0.3s;
            cursor: pointer;
        }
        .upload-area:hover, .upload-area.dragover {
            border-color: #4fc3f7;
            background: rgba(79, 195, 247, 0.05);
        }
        .upload-icon {
            font-size: 48px;
            margin-bottom: 20px;
        }
        .upload-area p {
            color: #90a4ae;
            margin-bottom: 10px;
        }
        .upload-area .supported {
            font-size: 0.9rem;
            color: #607d8b;
        }
        input[type="file"] {
            display: none;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            color: #90a4ae;
            margin-bottom: 8px;
            font-size: 0.9rem;
        }
        input[type="text"], textarea, select {
            width: 100%;
            padding: 12px 16px;
            background: rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            color: #e0e0e0;
            font-size: 1rem;
        }
        input[type="text"]:focus, textarea:focus, select:focus {
            outline: none;
            border-color: #4fc3f7;
        }
        textarea {
            min-height: 200px;
            resize: vertical;
            font-family: monospace;
        }
        .btn {
            padding: 12px 30px;
            background: #4fc3f7;
            color: #1a1a2e;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }
        .btn:hover {
            background: #29b6f6;
            transform: translateY(-2px);
        }
        .btn:disabled {
            background: #607d8b;
            cursor: not-allowed;
            transform: none;
        }
        .result {
            margin-top: 20px;
            padding: 20px;
            border-radius: 8px;
            display: none;
        }
        .result.success {
            background: rgba(76, 175, 80, 0.1);
            border: 1px solid rgba(76, 175, 80, 0.3);
            display: block;
        }
        .result.error {
            background: rgba(244, 67, 54, 0.1);
            border: 1px solid rgba(244, 67, 54, 0.3);
            display: block;
        }
        .result h3 {
            margin-bottom: 10px;
        }
        .result.success h3 {
            color: #81c784;
        }
        .result.error h3 {
            color: #e57373;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .stat {
            background: rgba(0, 0, 0, 0.2);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        .stat-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #4fc3f7;
        }
        .stat-label {
            font-size: 0.8rem;
            color: #90a4ae;
            margin-top: 5px;
        }
        .uploads-list {
            margin-top: 20px;
        }
        .upload-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 8px;
            margin-bottom: 10px;
        }
        .upload-item-info h4 {
            color: #4fc3f7;
            margin-bottom: 5px;
        }
        .upload-item-info p {
            font-size: 0.85rem;
            color: #90a4ae;
        }
        .delete-btn {
            background: rgba(244, 67, 54, 0.2);
            color: #e57373;
            border: 1px solid rgba(244, 67, 54, 0.3);
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
        }
        .delete-btn:hover {
            background: rgba(244, 67, 54, 0.3);
        }
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        .loading.active {
            display: block;
        }
        .spinner {
            border: 3px solid rgba(79, 195, 247, 0.1);
            border-top: 3px solid #4fc3f7;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .warnings {
            margin-top: 15px;
            padding: 10px;
            background: rgba(255, 193, 7, 0.1);
            border: 1px solid rgba(255, 193, 7, 0.3);
            border-radius: 6px;
        }
        .warnings h4 {
            color: #ffd54f;
            margin-bottom: 10px;
        }
        .warnings ul {
            margin-left: 20px;
            color: #90a4ae;
        }
        /* Query section styles */
        .search-box {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        .search-box input {
            flex: 1;
            padding: 15px 20px;
            font-size: 1.1rem;
        }
        .search-box .btn {
            padding: 15px 30px;
            white-space: nowrap;
        }
        .search-options {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .search-options .form-group {
            margin-bottom: 0;
            min-width: 150px;
        }
        .search-options label {
            font-size: 0.85rem;
        }
        .search-options input, .search-options select {
            padding: 8px 12px;
            font-size: 0.9rem;
        }
        .query-results {
            margin-top: 20px;
        }
        .query-result-item {
            background: rgba(0, 0, 0, 0.2);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 15px;
            border-left: 4px solid #4fc3f7;
        }
        .query-result-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 10px;
            flex-wrap: wrap;
            gap: 10px;
        }
        .query-result-title {
            color: #4fc3f7;
            font-weight: 600;
            font-size: 1.1rem;
        }
        .query-result-score {
            background: rgba(79, 195, 247, 0.2);
            color: #4fc3f7;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.85rem;
        }
        .query-result-path {
            color: #90a4ae;
            font-size: 0.85rem;
            margin-bottom: 10px;
        }
        .query-result-text {
            color: #e0e0e0;
            line-height: 1.6;
            font-size: 0.95rem;
        }
        .query-result-refs {
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
        }
        .query-result-refs-title {
            color: #90a4ae;
            font-size: 0.8rem;
            margin-bottom: 5px;
        }
        .query-result-refs a {
            color: #81c784;
            text-decoration: none;
            font-size: 0.85rem;
            margin-right: 10px;
        }
        .query-result-refs a:hover {
            text-decoration: underline;
        }
        .no-results {
            text-align: center;
            padding: 40px;
            color: #607d8b;
        }
        .results-summary {
            color: #90a4ae;
            margin-bottom: 15px;
            font-size: 0.9rem;
        }
        .example-queries {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 15px;
        }
        .example-query {
            background: rgba(79, 195, 247, 0.1);
            border: 1px solid rgba(79, 195, 247, 0.2);
            color: #4fc3f7;
            padding: 6px 12px;
            border-radius: 16px;
            font-size: 0.85rem;
            cursor: pointer;
            transition: all 0.2s;
        }
        .example-query:hover {
            background: rgba(79, 195, 247, 0.2);
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📜 Legal RAG System</h1>
            <p class="subtitle">Search and manage Hong Kong property law ordinances</p>
        </header>

        <div class="card">
            <h2>🔍 Search Legal Database</h2>
            <div class="search-box">
                <input type="text" id="search-query" placeholder="Ask a question about Hong Kong property law..."
                       onkeypress="if(event.key==='Enter') submitQuery()">
                <button class="btn" onclick="submitQuery()">Search</button>
            </div>

            <div class="search-options">
                <div class="form-group">
                    <label for="search-top-k">Results</label>
                    <select id="search-top-k">
                        <option value="5">5 results</option>
                        <option value="10" selected>10 results</option>
                        <option value="20">20 results</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="search-ordinance">Filter by Ordinance</label>
                    <select id="search-ordinance">
                        <option value="">All Ordinances</option>
                        <option value="Cap. 344">Cap. 344 - Building Management</option>
                        <option value="Cap. 123">Cap. 123 - Buildings Ordinance</option>
                        <option value="Cap. 572">Cap. 572 - Fire Safety (Buildings)</option>
                        <option value="Cap. 95">Cap. 95 - Fire Services</option>
                        <option value="Cap. 131">Cap. 131 - Town Planning</option>
                        <option value="Cap. 7">Cap. 7 - Landlord and Tenant</option>
                        <option value="Cap. 618">Cap. 618 - Lifts and Escalators</option>
                        <option value="Cap. 563">Cap. 563 - Urban Renewal Authority</option>
                    </select>
                </div>
            </div>

            <div class="example-queries">
                <span style="color: #607d8b; font-size: 0.85rem;">Try:</span>
                <span class="example-query" onclick="setQuery('What are the duties of the management committee?')">Management committee duties</span>
                <span class="example-query" onclick="setQuery('How to form an incorporated owners corporation?')">Form IO corporation</span>
                <span class="example-query" onclick="setQuery('Fire safety requirements for buildings')">Fire safety</span>
                <span class="example-query" onclick="setQuery('AGM voting requirements')">AGM voting</span>
            </div>

            <div class="loading" id="query-loading">
                <div class="spinner"></div>
                <p>Searching legal database...</p>
            </div>

            <div class="query-results" id="query-results"></div>
        </div>

        <div class="card">
            <h2>📤 Upload Document</h2>
            <div class="tabs">
                <div class="tab active" onclick="switchTab('file')">📁 File Upload</div>
                <div class="tab" onclick="switchTab('text')">📝 Text Input</div>
            </div>

            <div id="file-tab" class="tab-content active">
                <div class="upload-area" id="drop-area" onclick="document.getElementById('file-input').click()">
                    <div class="upload-icon">📤</div>
                    <p>Drag & drop your ordinance file here</p>
                    <p>or click to browse</p>
                    <p class="supported">Supported: XML (HKEL format), TXT</p>
                </div>
                <input type="file" id="file-input" accept=".xml,.txt" onchange="handleFileSelect(event)">

                <div class="form-group" style="margin-top: 20px;">
                    <label for="file-custom-id">Custom ID (optional)</label>
                    <input type="text" id="file-custom-id" placeholder="e.g., cap_999">
                </div>

                <button class="btn" id="upload-file-btn" onclick="uploadFile()" disabled>Upload File</button>
            </div>

            <div id="text-tab" class="tab-content">
                <div class="form-group">
                    <label for="text-title">Document Title *</label>
                    <input type="text" id="text-title" placeholder="e.g., Building Safety Guidelines 2024">
                </div>

                <div class="form-group">
                    <label for="text-type">Document Type</label>
                    <select id="text-type">
                        <option value="regulatory_statement">Regulatory Statement</option>
                        <option value="guideline">Guideline</option>
                        <option value="notice">Notice</option>
                        <option value="circular">Circular</option>
                        <option value="practice_direction">Practice Direction</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="text-content">Document Content *</label>
                    <textarea id="text-content" placeholder="Paste your regulatory statement or guideline text here...

You can use numbered sections like:
1. First provision...
2. Second provision...

Cross-references to existing ordinances (e.g., Cap. 344, Building Management Ordinance) will be automatically detected."></textarea>
                </div>

                <button class="btn" onclick="uploadText()">Upload Text</button>
            </div>

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Processing document...</p>
            </div>

            <div class="result" id="result"></div>
        </div>

        <div class="card">
            <h2>Uploaded Documents</h2>
            <div class="uploads-list" id="uploads-list">
                <p style="color: #607d8b;">Loading...</p>
            </div>
        </div>

        <div class="card">
            <h2>Index Status</h2>
            <div class="stats" id="index-stats">
                <div class="stat">
                    <div class="stat-value" id="stat-files">-</div>
                    <div class="stat-label">Chunk Files</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="stat-chunks">-</div>
                    <div class="stat-label">Total Chunks</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="stat-retriever">-</div>
                    <div class="stat-label">Retriever</div>
                </div>
            </div>
            <button class="btn" style="margin-top: 20px;" onclick="rebuildIndex()">🔄 Rebuild Index</button>
        </div>
    </div>

    <script>
        let selectedFile = null;

        // Tab switching
        function switchTab(tab) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

            document.querySelector(`.tab:nth-child(${tab === 'file' ? 1 : 2})`).classList.add('active');
            document.getElementById(`${tab}-tab`).classList.add('active');
        }

        // File drag & drop
        const dropArea = document.getElementById('drop-area');

        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, preventDefaults, false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        ['dragenter', 'dragover'].forEach(eventName => {
            dropArea.addEventListener(eventName, () => dropArea.classList.add('dragover'), false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, () => dropArea.classList.remove('dragover'), false);
        });

        dropArea.addEventListener('drop', handleDrop, false);

        function handleDrop(e) {
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                selectedFile = files[0];
                updateFileDisplay();
            }
        }

        function handleFileSelect(event) {
            selectedFile = event.target.files[0];
            updateFileDisplay();
        }

        function updateFileDisplay() {
            if (selectedFile) {
                dropArea.innerHTML = `
                    <div class="upload-icon">📄</div>
                    <p><strong>${selectedFile.name}</strong></p>
                    <p class="supported">${(selectedFile.size / 1024).toFixed(2)} KB</p>
                `;
                document.getElementById('upload-file-btn').disabled = false;
            }
        }

        // Upload functions
        async function uploadFile() {
            if (!selectedFile) return;

            showLoading();

            const formData = new FormData();
            formData.append('file', selectedFile);

            const customId = document.getElementById('file-custom-id').value;
            if (customId) formData.append('custom_id', customId);

            try {
                const response = await fetch('/api/upload/file', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (response.ok) {
                    showResult(data, true);
                } else {
                    showResult({ error_message: data.detail }, false);
                }
            } catch (error) {
                showResult({ error_message: error.message }, false);
            }

            hideLoading();
            loadUploads();
            loadIndexStatus();
        }

        async function uploadText() {
            const title = document.getElementById('text-title').value;
            const content = document.getElementById('text-content').value;
            const docType = document.getElementById('text-type').value;

            if (!title || !content) {
                alert('Please fill in title and content');
                return;
            }

            showLoading();

            try {
                const response = await fetch('/api/upload/text', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        text: content,
                        title: title,
                        doc_type: docType
                    })
                });

                const data = await response.json();

                if (response.ok) {
                    showResult(data, true);
                    document.getElementById('text-title').value = '';
                    document.getElementById('text-content').value = '';
                } else {
                    showResult({ error_message: data.detail }, false);
                }
            } catch (error) {
                showResult({ error_message: error.message }, false);
            }

            hideLoading();
            loadUploads();
            loadIndexStatus();
        }

        function showLoading() {
            document.getElementById('loading').classList.add('active');
            document.getElementById('result').className = 'result';
        }

        function hideLoading() {
            document.getElementById('loading').classList.remove('active');
        }

        function showResult(data, success) {
            const resultDiv = document.getElementById('result');

            if (success) {
                let html = `
                    <h3>✅ Upload Successful</h3>
                    <p><strong>${data.ordinance_name}</strong> (ID: ${data.ordinance_id})</p>
                    <div class="stats">
                        <div class="stat">
                            <div class="stat-value">${data.chunks_created}</div>
                            <div class="stat-label">Chunks Created</div>
                        </div>
                        <div class="stat">
                            <div class="stat-value">${data.cross_references_found}</div>
                            <div class="stat-label">Cross-References</div>
                        </div>
                        <div class="stat">
                            <div class="stat-value">${data.definitions_extracted}</div>
                            <div class="stat-label">Definitions</div>
                        </div>
                    </div>
                `;

                if (data.warnings && data.warnings.length > 0) {
                    html += `
                        <div class="warnings">
                            <h4>⚠️ Warnings</h4>
                            <ul>
                                ${data.warnings.map(w => `<li>${w}</li>`).join('')}
                            </ul>
                        </div>
                    `;
                }

                resultDiv.innerHTML = html;
                resultDiv.className = 'result success';
            } else {
                resultDiv.innerHTML = `
                    <h3>❌ Upload Failed</h3>
                    <p>${data.error_message || 'Unknown error'}</p>
                `;
                resultDiv.className = 'result error';
            }
        }

        // Load uploads list
        async function loadUploads() {
            try {
                const response = await fetch('/api/uploads');
                const data = await response.json();

                const listDiv = document.getElementById('uploads-list');

                if (data.uploads.length === 0) {
                    listDiv.innerHTML = '<p style="color: #607d8b;">No documents uploaded yet.</p>';
                    return;
                }

                listDiv.innerHTML = data.uploads.map(upload => `
                    <div class="upload-item">
                        <div class="upload-item-info">
                            <h4>${upload.ordinance_name}</h4>
                            <p>ID: ${upload.ordinance_id} | ${upload.chunks_created} chunks | Uploaded: ${new Date(upload.uploaded_at).toLocaleDateString()}</p>
                        </div>
                        <button class="delete-btn" onclick="deleteUpload('${upload.ordinance_id}')">🗑️ Delete</button>
                    </div>
                `).join('');
            } catch (error) {
                console.error('Error loading uploads:', error);
            }
        }

        async function deleteUpload(id) {
            if (!confirm('Are you sure you want to delete this document?')) return;

            try {
                await fetch(`/api/uploads/${id}`, { method: 'DELETE' });
                loadUploads();
                loadIndexStatus();
            } catch (error) {
                alert('Error deleting document: ' + error.message);
            }
        }

        // Load index status
        async function loadIndexStatus() {
            try {
                const response = await fetch('/api/index/status');
                const data = await response.json();

                document.getElementById('stat-files').textContent = data.chunk_files;
                document.getElementById('stat-chunks').textContent = data.total_chunks;
                document.getElementById('stat-retriever').textContent = data.retriever_loaded ? '✅' : '❌';
            } catch (error) {
                console.error('Error loading index status:', error);
            }
        }

        async function rebuildIndex() {
            if (!confirm('Rebuild the entire index? This may take a while.')) return;

            showLoading();

            try {
                const response = await fetch('/api/index/rebuild', { method: 'POST' });
                const data = await response.json();

                if (response.ok) {
                    alert('Index rebuilt successfully!');
                } else {
                    alert('Error: ' + data.detail);
                }
            } catch (error) {
                alert('Error rebuilding index: ' + error.message);
            }

            hideLoading();
            loadIndexStatus();
        }

        // Query functions
        function setQuery(query) {
            document.getElementById('search-query').value = query;
            submitQuery();
        }

        async function submitQuery() {
            const query = document.getElementById('search-query').value.trim();
            if (!query) {
                alert('Please enter a search query');
                return;
            }

            const topK = document.getElementById('search-top-k').value;
            const filterOrdinance = document.getElementById('search-ordinance').value;

            // Show loading
            document.getElementById('query-loading').classList.add('active');
            document.getElementById('query-results').innerHTML = '';

            try {
                const response = await fetch('/api/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        query: query,
                        top_k: parseInt(topK),
                        filter_ordinance: filterOrdinance || null,
                        expand_context: true
                    })
                });

                const data = await response.json();

                if (response.ok) {
                    displayQueryResults(data);
                } else {
                    document.getElementById('query-results').innerHTML = `
                        <div class="no-results">
                            <p>❌ Error: ${data.detail || 'Unknown error'}</p>
                        </div>
                    `;
                }
            } catch (error) {
                document.getElementById('query-results').innerHTML = `
                    <div class="no-results">
                        <p>❌ Error: ${error.message}</p>
                        <p style="margin-top: 10px; font-size: 0.9rem;">Make sure the server is running and the index is built.</p>
                    </div>
                `;
            }

            // Hide loading
            document.getElementById('query-loading').classList.remove('active');
        }

        function displayQueryResults(data) {
            const resultsDiv = document.getElementById('query-results');

            if (!data.results || data.results.length === 0) {
                resultsDiv.innerHTML = `
                    <div class="no-results">
                        <p>No results found for your query.</p>
                        <p style="margin-top: 10px; font-size: 0.9rem;">Try different keywords or remove filters.</p>
                    </div>
                `;
                return;
            }

            let html = `<div class="results-summary">Found ${data.total} results for "<strong>${data.query}</strong>"</div>`;

            data.results.forEach((result, index) => {
                const scorePercent = (result.score * 100).toFixed(1);
                const truncatedText = result.text.length > 500
                    ? result.text.substring(0, 500) + '...'
                    : result.text;

                html += `
                    <div class="query-result-item">
                        <div class="query-result-header">
                            <span class="query-result-title">${result.ordinance} ${result.section}</span>
                            <span class="query-result-score">Score: ${scorePercent}%</span>
                        </div>
                        <div class="query-result-path">📍 ${result.hierarchy_path || 'N/A'}</div>
                        <div class="query-result-text">${escapeHtml(truncatedText)}</div>
                        ${result.cross_references && result.cross_references.length > 0 ? `
                            <div class="query-result-refs">
                                <div class="query-result-refs-title">Cross-references:</div>
                                ${result.cross_references.slice(0, 5).map(ref =>
                                    `<a href="#" onclick="setQuery('${escapeHtml(ref.text || ref.href)}'); return false;">${escapeHtml(ref.text || ref.href)}</a>`
                                ).join('')}
                            </div>
                        ` : ''}
                    </div>
                `;
            });

            resultsDiv.innerHTML = html;
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // Initialize
        loadUploads();
        loadIndexStatus();
    </script>
</body>
</html>
    """


# Run with: uvicorn src.api.main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
