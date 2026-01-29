"""
Retrieval module for Legal RAG system.

This module implements:
- Hybrid retrieval (dense + sparse)
- Knowledge graph for cross-references
- Context expansion for legal accuracy
"""

from .hybrid_retriever import HybridRetriever
from .knowledge_graph import LegalKnowledgeGraph

__all__ = [
    'HybridRetriever',
    'LegalKnowledgeGraph'
]
