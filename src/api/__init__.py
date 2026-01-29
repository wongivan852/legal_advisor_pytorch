"""
API module for Legal RAG system.

Provides:
- FastAPI web server for uploads and queries
- REST endpoints for document management
"""

from .main import app

__all__ = ['app']
