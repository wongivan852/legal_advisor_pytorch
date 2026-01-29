"""
Preprocessing module for Legal RAG system.

This module handles:
- XML parsing of HKEL legal documents
- Hierarchical chunking for legal accuracy
- Definition and cross-reference extraction
- Document upload and dynamic indexing
"""

from .hkel_parser import (
    HKELParser,
    Ordinance,
    LegalPart,
    LegalSection,
    Definition,
    CrossReference,
    parse_ordinance_directory
)

from .chunker import (
    LegalChunker,
    LegalChunk,
    chunk_all_ordinances
)

from .upload_handler import (
    OrdinanceUploadHandler,
    DynamicIndexUpdater,
    UploadResult
)

__all__ = [
    'HKELParser',
    'Ordinance',
    'LegalPart',
    'LegalSection',
    'Definition',
    'CrossReference',
    'parse_ordinance_directory',
    'LegalChunker',
    'LegalChunk',
    'chunk_all_ordinances',
    'OrdinanceUploadHandler',
    'DynamicIndexUpdater',
    'UploadResult'
]
