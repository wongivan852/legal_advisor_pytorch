"""
Hybrid Retrieval for Legal RAG

Combines dense (semantic) and sparse (keyword) retrieval for
maximum accuracy in legal document retrieval.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import numpy as np
from collections import defaultdict
import re

# Optional imports - will be installed as needed
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False


@dataclass
class RetrievalResult:
    """Result from retrieval with full metadata"""
    chunk_id: str
    score: float
    text: str
    ordinance: str
    section: str
    hierarchy_path: str
    chunk_type: str
    cross_references: List[Dict]
    dense_score: float = 0.0
    sparse_score: float = 0.0


class HybridRetriever:
    """
    Hybrid retrieval combining dense and sparse methods.

    For legal documents, we need:
    1. Semantic search - find conceptually similar provisions
    2. Keyword search - exact legal term matching
    3. Knowledge graph - cross-reference traversal
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: Optional[str] = None,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.3,
        kg_weight: float = 0.2
    ):
        """
        Initialize the hybrid retriever.

        Args:
            model_name: HuggingFace model for dense embeddings
            device: torch device (auto-detected if None)
            dense_weight: Weight for dense retrieval (0-1)
            sparse_weight: Weight for sparse retrieval (0-1)
            kg_weight: Weight for knowledge graph results (0-1)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        # Retrieval weights (should sum to 1)
        total = dense_weight + sparse_weight + kg_weight
        self.dense_weight = dense_weight / total
        self.sparse_weight = sparse_weight / total
        self.kg_weight = kg_weight / total

        # Will be initialized on first use or explicit call
        self.model = None
        self.chunks = []
        self.chunk_texts = []
        self.embeddings = None
        self.bm25 = None
        self.chunk_id_to_idx = {}

        # Legal term boosting
        self.legal_terms = self._load_legal_terms()

    def _load_legal_terms(self) -> Dict[str, float]:
        """Load legal terms with boost weights"""
        # Higher weight = more important for retrieval
        return {
            # Ordinance references
            "cap": 2.0, "cap.": 2.0, "ordinance": 1.8,
            "section": 1.8, "s.": 1.8, "subsection": 1.5,

            # Building Management
            "incorporated owners": 2.0, "corporation": 1.8,
            "management committee": 1.8, "deed of mutual covenant": 2.0,
            "dmc": 1.8, "common parts": 1.5, "building manager": 1.5,

            # Fire Safety
            "fire safety": 1.8, "fire services": 1.8, "fire hazard": 1.5,

            # Meetings and voting
            "resolution": 1.5, "meeting": 1.3, "vote": 1.3, "quorum": 1.5,

            # Legal obligations
            "shall": 1.3, "must": 1.3, "required": 1.3,
            "prohibited": 1.3, "offence": 1.5, "penalty": 1.5
        }

    def load_model(self):
        """Load the embedding model"""
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )

        print(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        print(f"Model loaded on {self.device}")

    def index_chunks(self, chunks: List[Dict], save_path: Optional[Path] = None):
        """
        Index chunks for retrieval.

        Args:
            chunks: List of chunk dictionaries
            save_path: Optional path to save embeddings
        """
        if self.model is None:
            self.load_model()

        self.chunks = chunks
        self.chunk_texts = [c['text'] for c in chunks]
        self.chunk_id_to_idx = {c['chunk_id']: i for i, c in enumerate(chunks)}

        print(f"Indexing {len(chunks)} chunks...")

        # Generate dense embeddings
        self.embeddings = self._generate_embeddings(self.chunk_texts)

        # Build BM25 index
        self._build_bm25_index()

        if save_path:
            self._save_index(save_path)

        print("Indexing complete!")

    def _generate_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Generate embeddings for texts"""
        if not texts:
            return torch.empty(0, self.model.get_sentence_embedding_dimension())

        embeddings = self.model.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=True,
            batch_size=32
        )
        return F.normalize(embeddings, p=2, dim=1)

    def _build_bm25_index(self):
        """Build BM25 index for sparse retrieval"""
        if not HAS_BM25:
            print("Warning: rank_bm25 not installed. Sparse retrieval disabled.")
            return

        # Tokenize with legal term awareness
        tokenized_corpus = [self._tokenize_legal(text) for text in self.chunk_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _tokenize_legal(self, text: str) -> List[str]:
        """Tokenize text with legal term preservation"""
        # Lowercase
        text = text.lower()

        # Preserve multi-word legal terms
        for term in sorted(self.legal_terms.keys(), key=len, reverse=True):
            if ' ' in term:
                text = text.replace(term, term.replace(' ', '_'))

        # Basic tokenization
        tokens = re.findall(r'\b[\w_]+\b', text)

        return tokens

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filter_ordinance: Optional[str] = None,
        filter_status: str = "operational"
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks using hybrid search.

        Args:
            query: Search query
            top_k: Number of results to return
            filter_ordinance: Optional filter by ordinance (e.g., "Cap. 344")
            filter_status: Filter by status (default: operational)

        Returns:
            List of RetrievalResult objects
        """
        if self.model is None or self.embeddings is None:
            raise RuntimeError("Index not built. Call index_chunks() first.")

        # Dense retrieval
        dense_scores = self._dense_search(query)

        # Sparse retrieval
        sparse_scores = self._sparse_search(query) if self.bm25 else np.zeros(len(self.chunks))

        # Combine scores
        combined_scores = (
            self.dense_weight * dense_scores +
            self.sparse_weight * sparse_scores
        )

        # Apply filters
        if filter_ordinance or filter_status:
            combined_scores = self._apply_filters(
                combined_scores, filter_ordinance, filter_status
            )

        # Get top-k indices
        top_indices = np.argsort(combined_scores)[-top_k * 2:][::-1]

        # Build results
        results = []
        seen_chunks = set()

        for idx in top_indices:
            if len(results) >= top_k:
                break

            chunk = self.chunks[idx]

            # Skip duplicates
            if chunk['chunk_id'] in seen_chunks:
                continue
            seen_chunks.add(chunk['chunk_id'])

            result = RetrievalResult(
                chunk_id=chunk['chunk_id'],
                score=float(combined_scores[idx]),
                text=chunk['text'],
                ordinance=chunk['ordinance'],
                section=chunk.get('section', ''),
                hierarchy_path=chunk.get('hierarchy_path', ''),
                chunk_type=chunk['chunk_type'],
                cross_references=chunk.get('cross_references', []),
                dense_score=float(dense_scores[idx]),
                sparse_score=float(sparse_scores[idx])
            )
            results.append(result)

        return results

    def _dense_search(self, query: str) -> np.ndarray:
        """Perform dense (semantic) search"""
        query_embedding = self.model.encode(
            query,
            convert_to_tensor=True,
            show_progress_bar=False
        )
        query_embedding = F.normalize(query_embedding.unsqueeze(0), p=2, dim=1)

        # Cosine similarity
        similarities = torch.mm(query_embedding, self.embeddings.T).squeeze(0)
        return similarities.cpu().numpy()

    def _sparse_search(self, query: str) -> np.ndarray:
        """Perform sparse (BM25) search"""
        query_tokens = self._tokenize_legal(query)
        scores = self.bm25.get_scores(query_tokens)

        # Apply legal term boosting
        for i, chunk in enumerate(self.chunks):
            boost = 1.0
            text_lower = chunk['text'].lower()
            for term, weight in self.legal_terms.items():
                if term in text_lower and term in query.lower():
                    boost *= weight
            scores[i] *= boost

        # Normalize to 0-1 range
        if scores.max() > 0:
            scores = scores / scores.max()

        return scores

    def _apply_filters(
        self,
        scores: np.ndarray,
        ordinance: Optional[str],
        status: str
    ) -> np.ndarray:
        """Apply filters by setting non-matching scores to -inf"""
        for i, chunk in enumerate(self.chunks):
            if ordinance and chunk.get('ordinance') != ordinance:
                scores[i] = -np.inf
            if status and chunk.get('status', 'operational') != status:
                scores[i] = -np.inf
        return scores

    def expand_context(
        self,
        results: List[RetrievalResult],
        include_parent: bool = True,
        include_definitions: bool = True,
        max_expansions: int = 5
    ) -> List[RetrievalResult]:
        """
        Expand retrieval results with related context.

        Args:
            results: Initial retrieval results
            include_parent: Include parent sections
            include_definitions: Include relevant definitions
            max_expansions: Maximum additional chunks to add

        Returns:
            Expanded list of results
        """
        expanded = list(results)
        seen_ids = {r.chunk_id for r in results}
        expansions_added = 0

        for result in results:
            if expansions_added >= max_expansions:
                break

            chunk = self.chunks[self.chunk_id_to_idx.get(result.chunk_id, 0)]

            # Add parent chunk
            if include_parent and chunk.get('parent_chunk_id'):
                parent_idx = self.chunk_id_to_idx.get(chunk['parent_chunk_id'])
                if parent_idx is not None and chunk['parent_chunk_id'] not in seen_ids:
                    parent_chunk = self.chunks[parent_idx]
                    expanded.append(RetrievalResult(
                        chunk_id=parent_chunk['chunk_id'],
                        score=result.score * 0.8,  # Slightly lower score
                        text=parent_chunk['text'],
                        ordinance=parent_chunk['ordinance'],
                        section=parent_chunk.get('section', ''),
                        hierarchy_path=parent_chunk.get('hierarchy_path', ''),
                        chunk_type=parent_chunk['chunk_type'],
                        cross_references=parent_chunk.get('cross_references', [])
                    ))
                    seen_ids.add(parent_chunk['chunk_id'])
                    expansions_added += 1

            # Add definition chunks
            if include_definitions:
                for def_term in chunk.get('definitions_used', []):
                    def_id = f"{chunk['ordinance'].lower().replace(' ', '').replace('.', '')}_def_{def_term.lower().replace(' ', '')}"
                    def_idx = self.chunk_id_to_idx.get(def_id)
                    if def_idx is not None and def_id not in seen_ids:
                        def_chunk = self.chunks[def_idx]
                        expanded.append(RetrievalResult(
                            chunk_id=def_chunk['chunk_id'],
                            score=result.score * 0.7,
                            text=def_chunk['text'],
                            ordinance=def_chunk['ordinance'],
                            section=def_chunk.get('section', ''),
                            hierarchy_path=def_chunk.get('hierarchy_path', ''),
                            chunk_type='definition',
                            cross_references=[]
                        ))
                        seen_ids.add(def_id)
                        expansions_added += 1

        return expanded

    def _save_index(self, path: Path):
        """Save index to disk"""
        path.mkdir(parents=True, exist_ok=True)

        # Save embeddings
        torch.save(self.embeddings, path / "embeddings.pt")

        # Save chunk metadata
        with open(path / "chunks_meta.json", 'w') as f:
            json.dump(self.chunks, f, ensure_ascii=False)

        print(f"Index saved to {path}")

    def load_index(self, path: Path):
        """Load index from disk"""
        if self.model is None:
            self.load_model()

        # Load embeddings
        self.embeddings = torch.load(path / "embeddings.pt", map_location=self.device)

        # Load chunk metadata
        with open(path / "chunks_meta.json", 'r') as f:
            self.chunks = json.load(f)

        self.chunk_texts = [c['text'] for c in self.chunks]
        self.chunk_id_to_idx = {c['chunk_id']: i for i, c in enumerate(self.chunks)}

        # Rebuild BM25
        self._build_bm25_index()

        print(f"Index loaded from {path}: {len(self.chunks)} chunks")


class QueryProcessor:
    """
    Processes queries for optimal retrieval.

    Handles:
    - Legal term normalization
    - Query expansion
    - Intent classification
    """

    def __init__(self):
        self.ordinance_aliases = {
            "bmo": "Cap. 344",
            "building management ordinance": "Cap. 344",
            "bo": "Cap. 123",
            "buildings ordinance": "Cap. 123",
            "fso": "Cap. 95",
            "fire services ordinance": "Cap. 95",
            "fsbo": "Cap. 572",
            "fire safety buildings ordinance": "Cap. 572",
            "ura": "Cap. 563",
            "urban renewal authority ordinance": "Cap. 563"
        }

    def process_query(self, query: str) -> Dict:
        """
        Process a query for retrieval.

        Returns:
            Dict with processed query and metadata
        """
        # Normalize
        normalized = query.lower().strip()

        # Detect referenced ordinances
        ordinances = self._detect_ordinances(normalized)

        # Detect section references
        sections = self._detect_sections(normalized)

        # Expand query with synonyms
        expanded = self._expand_query(normalized)

        return {
            "original": query,
            "normalized": normalized,
            "expanded": expanded,
            "ordinances": ordinances,
            "sections": sections,
            "query_type": self._classify_query(normalized)
        }

    def _detect_ordinances(self, query: str) -> List[str]:
        """Detect ordinance references in query"""
        ordinances = []

        # Check aliases
        for alias, ordinance in self.ordinance_aliases.items():
            if alias in query:
                ordinances.append(ordinance)

        # Check Cap. references
        cap_pattern = r'cap\.?\s*(\d+[a-z]?)'
        for match in re.finditer(cap_pattern, query, re.IGNORECASE):
            ordinances.append(f"Cap. {match.group(1).upper()}")

        return list(set(ordinances))

    def _detect_sections(self, query: str) -> List[str]:
        """Detect section references in query"""
        sections = []

        # Pattern: section 3, s.3, s3
        section_pattern = r'(?:section|s\.?)\s*(\d+[a-z]?)'
        for match in re.finditer(section_pattern, query, re.IGNORECASE):
            sections.append(f"s.{match.group(1)}")

        return sections

    def _expand_query(self, query: str) -> str:
        """Expand query with synonyms"""
        expansions = {
            "io": "incorporated owners corporation",
            "mc": "management committee",
            "agm": "annual general meeting",
            "egm": "extraordinary general meeting",
            "dmc": "deed of mutual covenant"
        }

        expanded = query
        for abbrev, full in expansions.items():
            if re.search(rf'\b{abbrev}\b', expanded, re.IGNORECASE):
                expanded += f" {full}"

        return expanded

    def _classify_query(self, query: str) -> str:
        """Classify query type"""
        if any(word in query for word in ['what is', 'define', 'definition', 'meaning']):
            return "definition"
        if any(word in query for word in ['how to', 'procedure', 'process', 'steps']):
            return "procedural"
        if any(word in query for word in ['must', 'required', 'obligation', 'duty']):
            return "obligation"
        if any(word in query for word in ['can', 'may', 'allowed', 'permitted', 'right']):
            return "rights"
        if any(word in query for word in ['penalty', 'offence', 'fine', 'imprisonment']):
            return "penalty"
        return "general"


if __name__ == "__main__":
    # Example usage
    retriever = HybridRetriever(model_name="BAAI/bge-base-en-v1.5")

    # Load chunks
    chunks_path = Path("/home/user/legal_advisor_pytorch/data/chunks/all_chunks.json")
    if chunks_path.exists():
        with open(chunks_path) as f:
            chunks = json.load(f)

        retriever.index_chunks(chunks)

        # Test retrieval
        query = "What are the duties of the management committee under BMO?"
        results = retriever.retrieve(query, top_k=5)

        print(f"\nQuery: {query}\n")
        for r in results:
            print(f"Score: {r.score:.3f} | {r.ordinance} {r.section}")
            print(f"  {r.text[:200]}...")
            print()
