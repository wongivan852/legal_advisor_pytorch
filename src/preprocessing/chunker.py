"""
Legal Document Chunking Strategy

This module implements hierarchical chunking for legal documents,
preserving semantic units and maintaining legal citation accuracy.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set
from pathlib import Path
import json
import hashlib

from .hkel_parser import (
    Ordinance, LegalPart, LegalSection, Definition, CrossReference
)


@dataclass
class LegalChunk:
    """A chunk of legal text with full metadata for RAG"""

    # Unique identifiers
    chunk_id: str
    chunk_type: str  # "section", "definition", "part_summary"

    # Hierarchical location
    ordinance: str           # "Cap. 344"
    ordinance_name: str      # "Building Management Ordinance"
    part: Optional[str] = None
    part_name: Optional[str] = None
    section: Optional[str] = None
    section_name: Optional[str] = None
    subsection: Optional[str] = None
    paragraph: Optional[str] = None
    hierarchy_path: str = ""

    # Legal status
    effective_date: Optional[str] = None
    status: str = "operational"

    # Relationships
    parent_chunk_id: Optional[str] = None
    child_chunk_ids: List[str] = field(default_factory=list)
    cross_references: List[Dict] = field(default_factory=list)
    definitions_used: List[str] = field(default_factory=list)

    # Content
    text: str = ""
    heading: str = ""
    text_length: int = 0

    # For retrieval
    keywords: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    @staticmethod
    def generate_id(ordinance: str, section: str, subsection: str = "",
                    paragraph: str = "") -> str:
        """Generate a unique chunk ID"""
        parts = [ordinance.lower().replace(" ", "").replace(".", "")]
        if section:
            parts.append(section.replace(".", ""))
        if subsection:
            parts.append(subsection.replace("(", "").replace(")", ""))
        if paragraph:
            parts.append(paragraph.replace("(", "").replace(")", ""))
        return "_".join(parts)


class LegalChunker:
    """
    Chunker for legal documents that preserves semantic structure.

    Chunking Strategy:
    1. Primary chunks are SECTIONS (complete legal provisions)
    2. Large sections are split at SUBSECTION boundaries
    3. Definitions are extracted as separate chunks
    4. Cross-references are preserved in metadata
    5. Hierarchy is maintained through parent/child relationships
    """

    def __init__(self, max_chunk_size: int = 2000, min_chunk_size: int = 100):
        """
        Initialize the chunker.

        Args:
            max_chunk_size: Maximum characters per chunk (before splitting)
            min_chunk_size: Minimum characters for a standalone chunk
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def chunk_ordinance(self, ordinance: Ordinance) -> List[LegalChunk]:
        """
        Chunk an entire ordinance into retrieval units.

        Returns:
            List of LegalChunk objects
        """
        chunks = []

        # Create ordinance-level summary chunk
        summary_chunk = self._create_ordinance_summary(ordinance)
        chunks.append(summary_chunk)

        # Create definition chunks
        definition_chunks = self._create_definition_chunks(ordinance)
        chunks.extend(definition_chunks)

        # Process parts
        for part in ordinance.parts:
            part_chunks = self._chunk_part(part, ordinance)
            chunks.extend(part_chunks)

        # Process sections not in parts
        for section in ordinance.sections:
            section_chunks = self._chunk_section(section, ordinance, None)
            chunks.extend(section_chunks)

        # Set parent-child relationships
        self._link_parent_child(chunks)

        return chunks

    def _create_ordinance_summary(self, ordinance: Ordinance) -> LegalChunk:
        """Create a summary chunk for the entire ordinance"""
        summary_text = f"""
{ordinance.doc_name} - {ordinance.short_title}

{ordinance.long_title}

Status: {ordinance.doc_status}
Effective Date: {ordinance.effective_date}

This ordinance contains {len(ordinance.parts)} parts and {len(ordinance.all_definitions)} definitions.
        """.strip()

        return LegalChunk(
            chunk_id=LegalChunk.generate_id(ordinance.doc_name, "summary"),
            chunk_type="ordinance_summary",
            ordinance=ordinance.doc_name,
            ordinance_name=ordinance.short_title,
            hierarchy_path=ordinance.doc_name,
            effective_date=ordinance.effective_date,
            text=summary_text,
            heading=ordinance.short_title,
            text_length=len(summary_text),
            keywords=self._extract_keywords(summary_text)
        )

    def _create_definition_chunks(self, ordinance: Ordinance) -> List[LegalChunk]:
        """Create separate chunks for each definition"""
        chunks = []

        for name, defn in ordinance.all_definitions.items():
            text = f"Definition of '{defn.term_en}'"
            if defn.term_zh:
                text += f" ({defn.term_zh})"
            text += f":\n\n{defn.content}"

            chunk = LegalChunk(
                chunk_id=LegalChunk.generate_id(ordinance.doc_name, "def", name),
                chunk_type="definition",
                ordinance=ordinance.doc_name,
                ordinance_name=ordinance.short_title,
                section=defn.source_section,
                hierarchy_path=f"{ordinance.doc_name} > Definitions > {defn.term_en}",
                text=text,
                heading=f"Definition: {defn.term_en}",
                text_length=len(text),
                keywords=[defn.term_en] + ([defn.term_zh] if defn.term_zh else [])
            )
            chunks.append(chunk)

        return chunks

    def _chunk_part(self, part: LegalPart, ordinance: Ordinance) -> List[LegalChunk]:
        """Chunk a part of an ordinance"""
        chunks = []

        # Create part summary
        part_summary = LegalChunk(
            chunk_id=LegalChunk.generate_id(ordinance.doc_name, part.name),
            chunk_type="part_summary",
            ordinance=ordinance.doc_name,
            ordinance_name=ordinance.short_title,
            part=part.num,
            part_name=part.heading,
            hierarchy_path=f"{ordinance.doc_name} > {part.heading}",
            text=f"{part.num} - {part.heading}\n\nThis part contains {len(part.sections)} sections.",
            heading=part.heading,
            text_length=0
        )
        chunks.append(part_summary)

        # Chunk each section in the part
        for section in part.sections:
            section_chunks = self._chunk_section(section, ordinance, part)
            chunks.extend(section_chunks)

        return chunks

    def _chunk_section(self, section: LegalSection, ordinance: Ordinance,
                       part: Optional[LegalPart]) -> List[LegalChunk]:
        """
        Chunk a section, potentially splitting into subsections if too large.
        """
        chunks = []

        # Skip repealed sections (but we could optionally keep them with a flag)
        if section.status == 'repealed':
            return chunks

        # Build the full section text
        full_text = self._build_section_text(section)

        # Extract cross-references
        cross_refs = [
            {"href": ref.href, "text": ref.text, "target": ref.target_ordinance}
            for ref in section.cross_references
        ]

        # Extract definition terms used
        definitions_used = [d.term_en for d in section.definitions]

        # Create the primary section chunk
        section_chunk = LegalChunk(
            chunk_id=LegalChunk.generate_id(ordinance.doc_name, section.name),
            chunk_type="section",
            ordinance=ordinance.doc_name,
            ordinance_name=ordinance.short_title,
            part=part.num if part else None,
            part_name=part.heading if part else None,
            section=section.num,
            section_name=section.heading,
            hierarchy_path=section.hierarchy_path or f"{ordinance.doc_name} > {section.num}",
            effective_date=section.effective_date,
            status=section.status,
            text=full_text,
            heading=section.heading or section.num,
            text_length=len(full_text),
            cross_references=cross_refs,
            definitions_used=definitions_used,
            keywords=self._extract_keywords(full_text)
        )
        chunks.append(section_chunk)

        # If section is large, also create subsection chunks for finer retrieval
        if len(full_text) > self.max_chunk_size and section.children:
            for child in section.children:
                child_chunks = self._chunk_subsection(child, section_chunk, ordinance, part)
                section_chunk.child_chunk_ids.extend([c.chunk_id for c in child_chunks])
                chunks.extend(child_chunks)

        return chunks

    def _chunk_subsection(self, subsection: LegalSection, parent_chunk: LegalChunk,
                          ordinance: Ordinance, part: Optional[LegalPart]) -> List[LegalChunk]:
        """Create chunks for subsections"""
        chunks = []

        text = self._build_section_text(subsection)

        if len(text) < self.min_chunk_size:
            return chunks  # Too small to be useful as standalone

        chunk = LegalChunk(
            chunk_id=LegalChunk.generate_id(
                ordinance.doc_name,
                parent_chunk.section or "",
                subsection.num
            ),
            chunk_type="subsection",
            ordinance=ordinance.doc_name,
            ordinance_name=ordinance.short_title,
            part=part.num if part else None,
            part_name=part.heading if part else None,
            section=parent_chunk.section,
            section_name=parent_chunk.section_name,
            subsection=subsection.num,
            hierarchy_path=subsection.hierarchy_path,
            effective_date=subsection.effective_date or parent_chunk.effective_date,
            status=subsection.status,
            parent_chunk_id=parent_chunk.chunk_id,
            text=text,
            heading=f"{parent_chunk.section_name} {subsection.num}" if parent_chunk.section_name else subsection.num,
            text_length=len(text),
            cross_references=[
                {"href": ref.href, "text": ref.text, "target": ref.target_ordinance}
                for ref in subsection.cross_references
            ],
            definitions_used=[d.term_en for d in subsection.definitions],
            keywords=self._extract_keywords(text)
        )
        chunks.append(chunk)

        return chunks

    def _build_section_text(self, section: LegalSection) -> str:
        """Build the full text for a section including its children"""
        parts = []

        # Add heading if present
        if section.heading:
            parts.append(f"{section.num} {section.heading}")
        elif section.num:
            parts.append(section.num)

        # Add main content
        if section.content:
            parts.append(section.content)

        # Add children content (subsections, paragraphs)
        for child in section.children:
            child_text = self._build_section_text(child)
            if child_text:
                parts.append(child_text)

        return "\n\n".join(filter(None, parts))

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text for sparse retrieval boost"""
        # Legal keywords to prioritize
        legal_terms = [
            "owner", "owners", "corporation", "incorporated", "management",
            "committee", "meeting", "resolution", "vote", "voting",
            "building", "flat", "common parts", "deed of mutual covenant",
            "fire", "safety", "ordinance", "section", "subsection",
            "shall", "must", "may", "required", "prohibited"
        ]

        text_lower = text.lower()
        found_keywords = [term for term in legal_terms if term in text_lower]

        return found_keywords

    def _link_parent_child(self, chunks: List[LegalChunk]):
        """Establish parent-child relationships between chunks"""
        chunk_map = {c.chunk_id: c for c in chunks}

        for chunk in chunks:
            if chunk.parent_chunk_id and chunk.parent_chunk_id in chunk_map:
                parent = chunk_map[chunk.parent_chunk_id]
                if chunk.chunk_id not in parent.child_chunk_ids:
                    parent.child_chunk_ids.append(chunk.chunk_id)


def chunk_all_ordinances(ordinances: List[Ordinance],
                         output_dir: Path,
                         max_chunk_size: int = 2000) -> List[LegalChunk]:
    """
    Chunk all ordinances and save to files.

    Args:
        ordinances: List of parsed ordinances
        output_dir: Directory to save chunks
        max_chunk_size: Maximum chunk size in characters

    Returns:
        List of all chunks
    """
    chunker = LegalChunker(max_chunk_size=max_chunk_size)
    all_chunks = []

    output_dir.mkdir(parents=True, exist_ok=True)

    for ordinance in ordinances:
        chunks = chunker.chunk_ordinance(ordinance)
        all_chunks.extend(chunks)

        # Save ordinance chunks to separate file
        ordinance_file = output_dir / f"{ordinance.doc_number}_chunks.json"
        with open(ordinance_file, 'w', encoding='utf-8') as f:
            json.dump([c.to_dict() for c in chunks], f, indent=2, ensure_ascii=False)

        print(f"Chunked {ordinance.doc_name}: {len(chunks)} chunks")

    # Save all chunks to single file
    all_chunks_file = output_dir / "all_chunks.json"
    with open(all_chunks_file, 'w', encoding='utf-8') as f:
        json.dump([c.to_dict() for c in all_chunks], f, indent=2, ensure_ascii=False)

    print(f"\nTotal chunks: {len(all_chunks)}")
    print(f"Saved to: {output_dir}")

    return all_chunks


if __name__ == "__main__":
    from .hkel_parser import parse_ordinance_directory

    # Parse and chunk
    data_dir = Path("/home/user/legal_advisor_pytorch/data/property_owner_ordinances")
    output_dir = Path("/home/user/legal_advisor_pytorch/data/chunks")

    ordinances = parse_ordinance_directory(data_dir)
    chunks = chunk_all_ordinances(ordinances, output_dir)

    # Print sample chunks
    print("\n=== Sample Chunks ===")
    for chunk in chunks[:3]:
        print(f"\nID: {chunk.chunk_id}")
        print(f"Type: {chunk.chunk_type}")
        print(f"Path: {chunk.hierarchy_path}")
        print(f"Length: {chunk.text_length}")
        print(f"Text preview: {chunk.text[:200]}...")
