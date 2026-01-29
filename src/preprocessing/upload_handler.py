"""
Ordinance Upload Handler

Handles uploading and processing of new ordinance documents,
integrating them into the existing RAG system with full
chunking and cross-reference support.
"""

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import hashlib
import tempfile

from .hkel_parser import HKELParser, Ordinance
from .chunker import LegalChunker, LegalChunk


@dataclass
class UploadResult:
    """Result of an ordinance upload operation"""
    success: bool
    ordinance_id: str
    ordinance_name: str
    file_path: str
    chunks_created: int
    cross_references_found: int
    definitions_extracted: int
    error_message: Optional[str] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "ordinance_id": self.ordinance_id,
            "ordinance_name": self.ordinance_name,
            "file_path": self.file_path,
            "chunks_created": self.chunks_created,
            "cross_references_found": self.cross_references_found,
            "definitions_extracted": self.definitions_extracted,
            "error_message": self.error_message,
            "warnings": self.warnings
        }


class OrdinanceUploadHandler:
    """
    Handles upload and processing of new ordinance documents.

    Supports:
    - XML files in HKEL format
    - Plain text regulatory statements
    - PDF documents (requires additional processing)
    - Batch uploads

    All uploaded documents are:
    1. Validated for format and content
    2. Parsed and chunked using the same strategy as base dataset
    3. Added to the cross-reference graph
    4. Indexed for retrieval
    """

    def __init__(
        self,
        data_dir: Path,
        index_dir: Path,
        max_chunk_size: int = 2000
    ):
        """
        Initialize the upload handler.

        Args:
            data_dir: Directory for storing uploaded ordinances
            index_dir: Directory for index files
            max_chunk_size: Maximum chunk size for chunking
        """
        self.data_dir = Path(data_dir)
        self.index_dir = Path(index_dir)
        self.max_chunk_size = max_chunk_size

        # Create directories
        self.uploads_dir = self.data_dir / "uploads"
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.parser = HKELParser()
        self.chunker = LegalChunker(max_chunk_size=max_chunk_size)

        # Track uploaded documents
        self.upload_registry_path = self.uploads_dir / "registry.json"
        self.upload_registry = self._load_registry()

    def _load_registry(self) -> Dict:
        """Load or create upload registry"""
        if self.upload_registry_path.exists():
            with open(self.upload_registry_path, 'r') as f:
                return json.load(f)
        return {"uploads": [], "last_updated": None}

    def _save_registry(self):
        """Save upload registry"""
        self.upload_registry["last_updated"] = datetime.now().isoformat()
        with open(self.upload_registry_path, 'w') as f:
            json.dump(self.upload_registry, f, indent=2, ensure_ascii=False)

    def upload_xml(
        self,
        file_path: Union[str, Path],
        custom_id: Optional[str] = None
    ) -> UploadResult:
        """
        Upload an XML ordinance file in HKEL format.

        Args:
            file_path: Path to the XML file
            custom_id: Optional custom identifier

        Returns:
            UploadResult with processing details
        """
        file_path = Path(file_path)

        if not file_path.exists():
            return UploadResult(
                success=False,
                ordinance_id="",
                ordinance_name="",
                file_path=str(file_path),
                chunks_created=0,
                cross_references_found=0,
                definitions_extracted=0,
                error_message=f"File not found: {file_path}"
            )

        try:
            # Parse the XML file
            ordinance = self.parser.parse_file(file_path)

            # Generate ID if not provided
            if custom_id:
                ordinance_id = custom_id
            else:
                ordinance_id = self._generate_ordinance_id(ordinance)

            # Copy file to uploads directory
            dest_dir = self.uploads_dir / f"{ordinance_id}_en_c"
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_file = dest_dir / file_path.name
            shutil.copy2(file_path, dest_file)

            # Process and chunk
            result = self._process_ordinance(ordinance, ordinance_id, str(dest_file))

            # Register upload
            self._register_upload(result)

            return result

        except Exception as e:
            return UploadResult(
                success=False,
                ordinance_id=custom_id or "",
                ordinance_name="",
                file_path=str(file_path),
                chunks_created=0,
                cross_references_found=0,
                definitions_extracted=0,
                error_message=f"Processing error: {str(e)}"
            )

    def upload_text(
        self,
        text: str,
        title: str,
        doc_type: str = "regulatory_statement",
        custom_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> UploadResult:
        """
        Upload a plain text regulatory statement.

        Args:
            text: The text content
            title: Document title
            doc_type: Type of document (regulatory_statement, guideline, etc.)
            custom_id: Optional custom identifier
            metadata: Additional metadata

        Returns:
            UploadResult with processing details
        """
        # Generate ID
        if custom_id:
            doc_id = custom_id
        else:
            doc_id = self._generate_text_id(title, text)

        # Convert to pseudo-ordinance structure
        ordinance = self._text_to_ordinance(text, title, doc_id, doc_type, metadata)

        # Save as JSON (since it's not XML)
        dest_dir = self.uploads_dir / f"{doc_id}_text"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / f"{doc_id}.json"

        with open(dest_file, 'w', encoding='utf-8') as f:
            json.dump({
                "title": title,
                "doc_type": doc_type,
                "text": text,
                "metadata": metadata or {},
                "uploaded_at": datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)

        # Process and chunk
        result = self._process_ordinance(ordinance, doc_id, str(dest_file))

        # Register upload
        self._register_upload(result)

        return result

    def upload_file(
        self,
        file_content: bytes,
        filename: str,
        content_type: str,
        custom_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> UploadResult:
        """
        Upload a file (for web UI integration).

        Args:
            file_content: File content as bytes
            filename: Original filename
            content_type: MIME type
            custom_id: Optional custom identifier
            metadata: Additional metadata

        Returns:
            UploadResult with processing details
        """
        # Save to temp file first
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp:
            tmp.write(file_content)
            tmp_path = Path(tmp.name)

        try:
            if content_type == "application/xml" or filename.endswith(".xml"):
                return self.upload_xml(tmp_path, custom_id)
            elif content_type == "text/plain" or filename.endswith(".txt"):
                text = file_content.decode('utf-8')
                title = Path(filename).stem
                return self.upload_text(text, title, custom_id=custom_id, metadata=metadata)
            else:
                return UploadResult(
                    success=False,
                    ordinance_id="",
                    ordinance_name=filename,
                    file_path="",
                    chunks_created=0,
                    cross_references_found=0,
                    definitions_extracted=0,
                    error_message=f"Unsupported file type: {content_type}"
                )
        finally:
            # Clean up temp file
            tmp_path.unlink(missing_ok=True)

    def _generate_ordinance_id(self, ordinance: Ordinance) -> str:
        """Generate unique ID for an ordinance"""
        if ordinance.doc_number:
            return f"uploaded_{ordinance.doc_number}"

        # Hash-based ID for documents without number
        content_hash = hashlib.md5(
            f"{ordinance.short_title}{ordinance.long_title}".encode()
        ).hexdigest()[:8]
        return f"uploaded_{content_hash}"

    def _generate_text_id(self, title: str, text: str) -> str:
        """Generate unique ID for text document"""
        content_hash = hashlib.md5(f"{title}{text[:500]}".encode()).hexdigest()[:8]
        safe_title = "".join(c if c.isalnum() else "_" for c in title.lower())[:20]
        return f"text_{safe_title}_{content_hash}"

    def _text_to_ordinance(
        self,
        text: str,
        title: str,
        doc_id: str,
        doc_type: str,
        metadata: Optional[Dict]
    ) -> Ordinance:
        """Convert plain text to Ordinance structure for processing"""
        from .hkel_parser import LegalSection, LegalPart

        # Parse text into sections (split by numbered sections or paragraphs)
        sections = self._parse_text_sections(text, doc_id)

        ordinance = Ordinance(
            doc_name=doc_id,
            doc_type=doc_type,
            doc_number=doc_id,
            doc_status="uploaded",
            short_title=title,
            long_title=metadata.get("description", title) if metadata else title,
            effective_date=datetime.now().strftime("%Y-%m-%d"),
            file_path=""
        )

        # Create a single part containing all sections
        if sections:
            part = LegalPart(
                id=f"{doc_id}_main",
                name="main",
                num="",
                heading=title,
                sections=sections
            )
            ordinance.parts.append(part)

        return ordinance

    def _parse_text_sections(self, text: str, doc_id: str) -> List:
        """Parse plain text into section structures"""
        from .hkel_parser import LegalSection
        import re

        sections = []

        # Try to split by numbered sections (1., 2., etc.)
        section_pattern = r'(?:^|\n)(\d+)\.\s*(.+?)(?=(?:\n\d+\.)|$)'
        matches = re.findall(section_pattern, text, re.DOTALL)

        if matches:
            for num, content in matches:
                section = LegalSection(
                    id=f"{doc_id}_s{num}",
                    name=f"s{num}",
                    num=f"{num}.",
                    content=content.strip(),
                    status="operational"
                )
                section.hierarchy_path = f"{doc_id} > Section {num}"
                sections.append(section)
        else:
            # Treat entire text as single section
            section = LegalSection(
                id=f"{doc_id}_s1",
                name="s1",
                num="1.",
                content=text.strip(),
                status="operational"
            )
            section.hierarchy_path = f"{doc_id} > Section 1"
            sections.append(section)

        return sections

    def _process_ordinance(
        self,
        ordinance: Ordinance,
        ordinance_id: str,
        file_path: str
    ) -> UploadResult:
        """Process ordinance: chunk and prepare for indexing"""
        warnings = []

        # Chunk the ordinance
        chunks = self.chunker.chunk_ordinance(ordinance)

        # Count cross-references and definitions
        cross_refs = len(ordinance.all_cross_references)
        definitions = len(ordinance.all_definitions)

        # Validate cross-references
        unresolved_refs = self._validate_cross_references(ordinance)
        if unresolved_refs:
            warnings.append(
                f"Found {len(unresolved_refs)} cross-references to external documents: "
                f"{', '.join(unresolved_refs[:5])}{'...' if len(unresolved_refs) > 5 else ''}"
            )

        # Save chunks
        chunks_file = self.index_dir / "chunks" / f"{ordinance_id}_chunks.json"
        chunks_file.parent.mkdir(parents=True, exist_ok=True)

        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump([c.to_dict() for c in chunks], f, indent=2, ensure_ascii=False)

        return UploadResult(
            success=True,
            ordinance_id=ordinance_id,
            ordinance_name=ordinance.short_title or ordinance.doc_name,
            file_path=file_path,
            chunks_created=len(chunks),
            cross_references_found=cross_refs,
            definitions_extracted=definitions,
            warnings=warnings
        )

    def _validate_cross_references(self, ordinance: Ordinance) -> List[str]:
        """Check for cross-references to documents not in the system"""
        # Load existing ordinance IDs
        existing_ids = self._get_existing_ordinance_ids()

        unresolved = []
        for ref in ordinance.all_cross_references:
            if ref.target_ordinance:
                # Normalize the ordinance reference
                target_id = ref.target_ordinance.lower().replace(" ", "").replace(".", "")
                if target_id not in existing_ids:
                    unresolved.append(ref.target_ordinance)

        return list(set(unresolved))

    def _get_existing_ordinance_ids(self) -> set:
        """Get IDs of all ordinances in the system"""
        ids = set()

        # Check base data directory
        for subdir in self.data_dir.iterdir():
            if subdir.is_dir() and subdir.name.endswith("_en_c"):
                # Extract cap number
                cap_match = subdir.name.replace("_en_c", "")
                ids.add(cap_match.lower())

        # Check uploads
        for upload in self.upload_registry.get("uploads", []):
            ids.add(upload["ordinance_id"].lower())

        return ids

    def _register_upload(self, result: UploadResult):
        """Register a successful upload"""
        if result.success:
            self.upload_registry["uploads"].append({
                "ordinance_id": result.ordinance_id,
                "ordinance_name": result.ordinance_name,
                "file_path": result.file_path,
                "chunks_created": result.chunks_created,
                "uploaded_at": datetime.now().isoformat()
            })
            self._save_registry()

    def list_uploads(self) -> List[Dict]:
        """List all uploaded documents"""
        return self.upload_registry.get("uploads", [])

    def delete_upload(self, ordinance_id: str) -> bool:
        """Delete an uploaded document"""
        # Remove from registry
        uploads = self.upload_registry.get("uploads", [])
        self.upload_registry["uploads"] = [
            u for u in uploads if u["ordinance_id"] != ordinance_id
        ]
        self._save_registry()

        # Remove files
        upload_dir = self.uploads_dir / f"{ordinance_id}_en_c"
        if upload_dir.exists():
            shutil.rmtree(upload_dir)

        text_dir = self.uploads_dir / f"{ordinance_id}_text"
        if text_dir.exists():
            shutil.rmtree(text_dir)

        # Remove chunks
        chunks_file = self.index_dir / "chunks" / f"{ordinance_id}_chunks.json"
        if chunks_file.exists():
            chunks_file.unlink()

        return True


class DynamicIndexUpdater:
    """
    Updates the RAG index when new documents are uploaded.

    Handles:
    - Adding new chunks to vector index
    - Updating knowledge graph with new cross-references
    - Rebuilding BM25 index
    """

    def __init__(self, index_dir: Path):
        self.index_dir = Path(index_dir)
        self.chunks_dir = self.index_dir / "chunks"

    def update_index(self, retriever, knowledge_graph, new_chunks: List[Dict]):
        """
        Update retriever and knowledge graph with new chunks.

        Args:
            retriever: HybridRetriever instance
            knowledge_graph: LegalKnowledgeGraph instance
            new_chunks: List of new chunk dictionaries
        """
        if not new_chunks:
            return

        # Add to retriever
        existing_chunks = retriever.chunks.copy()
        existing_chunks.extend(new_chunks)

        # Re-index (incremental would be better for large systems)
        retriever.index_chunks(existing_chunks)

        # Update knowledge graph
        knowledge_graph.build_from_chunks(existing_chunks)

    def get_all_chunks(self) -> List[Dict]:
        """Load all chunks from disk"""
        all_chunks = []

        for chunk_file in self.chunks_dir.glob("*_chunks.json"):
            with open(chunk_file, 'r') as f:
                chunks = json.load(f)
                all_chunks.extend(chunks)

        return all_chunks

    def rebuild_full_index(self, retriever, knowledge_graph):
        """Rebuild the entire index from chunk files"""
        all_chunks = self.get_all_chunks()

        print(f"Rebuilding index with {len(all_chunks)} total chunks...")

        # Re-index everything
        retriever.index_chunks(all_chunks)
        knowledge_graph.build_from_chunks(all_chunks)

        print("Index rebuild complete!")


if __name__ == "__main__":
    # Test upload handler
    handler = OrdinanceUploadHandler(
        data_dir=Path("/home/user/legal_advisor_pytorch/data/property_owner_ordinances"),
        index_dir=Path("/home/user/legal_advisor_pytorch/data/index")
    )

    # Test text upload
    sample_text = """
    1. All incorporated owners shall maintain proper financial records.

    2. The management committee must hold at least one annual general meeting.

    3. Any expenditure exceeding $200,000 requires approval by resolution at a general meeting.

    4. The corporation shall comply with the Building Management Ordinance (Cap. 344).
    """

    result = handler.upload_text(
        text=sample_text,
        title="Sample Building Management Guidelines",
        doc_type="guideline",
        metadata={"source": "Test", "version": "1.0"}
    )

    print(f"Upload result: {result.to_dict()}")
    print(f"\nAll uploads: {handler.list_uploads()}")
