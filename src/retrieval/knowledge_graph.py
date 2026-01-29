"""
Legal Knowledge Graph for Cross-Reference Resolution

This module builds and queries a knowledge graph of legal provisions
to support accurate cross-reference resolution in RAG.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
import json
import re
from collections import defaultdict

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


@dataclass
class LegalNode:
    """A node in the legal knowledge graph"""
    id: str
    node_type: str  # ordinance, part, section, definition
    name: str
    ordinance: str
    content_preview: str = ""
    status: str = "operational"


@dataclass
class LegalEdge:
    """An edge in the legal knowledge graph"""
    source: str
    target: str
    edge_type: str  # contains, references, defines, amends, related_to
    weight: float = 1.0


class LegalKnowledgeGraph:
    """
    Knowledge graph for Hong Kong legal documents.

    Supports:
    - Cross-reference resolution
    - Related provision discovery
    - Hierarchical navigation
    - Definition linking
    """

    def __init__(self):
        if not HAS_NETWORKX:
            raise ImportError(
                "networkx not installed. Install with: pip install networkx"
            )

        self.graph = nx.DiGraph()
        self.node_index: Dict[str, LegalNode] = {}
        self.definition_index: Dict[str, List[str]] = defaultdict(list)  # term -> node_ids

    def build_from_chunks(self, chunks: List[Dict]):
        """
        Build knowledge graph from chunk data.

        Args:
            chunks: List of LegalChunk dictionaries
        """
        print("Building knowledge graph...")

        # First pass: Create nodes
        for chunk in chunks:
            self._add_chunk_node(chunk)

        # Second pass: Create edges
        for chunk in chunks:
            self._add_chunk_edges(chunk)

        # Third pass: Add semantic similarity edges (optional)
        # self._add_similarity_edges(chunks)

        print(f"Knowledge graph built: {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")

    def _add_chunk_node(self, chunk: Dict):
        """Add a node for a chunk"""
        node = LegalNode(
            id=chunk['chunk_id'],
            node_type=chunk['chunk_type'],
            name=chunk.get('heading', chunk['chunk_id']),
            ordinance=chunk['ordinance'],
            content_preview=chunk['text'][:200] if chunk['text'] else "",
            status=chunk.get('status', 'operational')
        )

        self.node_index[node.id] = node
        self.graph.add_node(
            node.id,
            node_type=node.node_type,
            name=node.name,
            ordinance=node.ordinance,
            status=node.status
        )

        # Index definitions
        if node.node_type == 'definition':
            # Extract term from chunk_id (format: cap344_def_termname)
            term_match = re.search(r'_def_(.+)$', node.id)
            if term_match:
                term = term_match.group(1).replace('_', ' ')
                self.definition_index[term.lower()].append(node.id)

    def _add_chunk_edges(self, chunk: Dict):
        """Add edges for a chunk"""
        chunk_id = chunk['chunk_id']

        # Parent-child relationship (CONTAINS)
        if chunk.get('parent_chunk_id'):
            self.graph.add_edge(
                chunk['parent_chunk_id'],
                chunk_id,
                edge_type='contains',
                weight=1.0
            )

        # Cross-references (REFERENCES)
        for ref in chunk.get('cross_references', []):
            target_id = self._resolve_reference(ref, chunk['ordinance'])
            if target_id and target_id in self.node_index:
                self.graph.add_edge(
                    chunk_id,
                    target_id,
                    edge_type='references',
                    weight=0.8
                )

        # Definition usage (USES_DEFINITION)
        for term in chunk.get('definitions_used', []):
            term_lower = term.lower()
            if term_lower in self.definition_index:
                for def_node_id in self.definition_index[term_lower]:
                    if def_node_id != chunk_id:
                        self.graph.add_edge(
                            chunk_id,
                            def_node_id,
                            edge_type='uses_definition',
                            weight=0.5
                        )

    def _resolve_reference(self, ref: Dict, source_ordinance: str) -> Optional[str]:
        """
        Resolve a cross-reference to a node ID.

        Args:
            ref: Reference dictionary with href, text, target fields
            source_ordinance: Ordinance of the source chunk

        Returns:
            Node ID if resolvable, None otherwise
        """
        href = ref.get('href', '')

        if not href:
            return None

        # Parse href patterns
        # Pattern 1: /hk/cap344/s3 -> cap344_s3
        # Pattern 2: /hk/cap344 -> cap344_summary
        # Pattern 3: section X (internal) -> use source ordinance

        if href.startswith('/hk/'):
            match = re.match(r'/hk/(cap\d+[A-Z]?)(?:/(.+))?', href, re.IGNORECASE)
            if match:
                ordinance_code = match.group(1).lower()
                section = match.group(2)

                if section:
                    # Try to find the section
                    node_id = f"{ordinance_code}_{section.replace('.', '')}"
                    if node_id in self.node_index:
                        return node_id
                else:
                    # Reference to whole ordinance
                    node_id = f"{ordinance_code}_summary"
                    if node_id in self.node_index:
                        return node_id

        # Internal reference (e.g., "section 3A")
        text = ref.get('text', '')
        if 'section' in text.lower():
            section_match = re.search(r'section\s+(\d+[A-Z]?)', text, re.IGNORECASE)
            if section_match:
                # Use source ordinance
                ordinance_code = source_ordinance.lower().replace(' ', '').replace('.', '')
                node_id = f"{ordinance_code}_s{section_match.group(1)}"
                if node_id in self.node_index:
                    return node_id

        return None

    def get_related_nodes(
        self,
        node_id: str,
        max_hops: int = 2,
        edge_types: Optional[List[str]] = None,
        max_results: int = 10
    ) -> List[Tuple[str, float, str]]:
        """
        Get related nodes via graph traversal.

        Args:
            node_id: Starting node
            max_hops: Maximum traversal depth
            edge_types: Filter by edge types (None = all)
            max_results: Maximum results to return

        Returns:
            List of (node_id, relevance_score, path_description) tuples
        """
        if node_id not in self.graph:
            return []

        related = []
        visited = {node_id}

        # BFS with depth tracking
        queue = [(node_id, 0, "")]  # (node, depth, path)

        while queue and len(related) < max_results:
            current, depth, path = queue.pop(0)

            if depth > 0:
                # Calculate relevance score (decreases with depth)
                score = 1.0 / (depth + 1)
                related.append((current, score, path))

            if depth < max_hops:
                # Explore neighbors
                for neighbor in self.graph.neighbors(current):
                    if neighbor not in visited:
                        edge_data = self.graph.edges[current, neighbor]
                        edge_type = edge_data.get('edge_type', 'unknown')

                        if edge_types is None or edge_type in edge_types:
                            visited.add(neighbor)
                            new_path = f"{path} -> {edge_type} -> {neighbor}" if path else f"{edge_type} -> {neighbor}"
                            queue.append((neighbor, depth + 1, new_path))

                # Also explore incoming edges (for bidirectional traversal)
                for predecessor in self.graph.predecessors(current):
                    if predecessor not in visited:
                        edge_data = self.graph.edges[predecessor, current]
                        edge_type = edge_data.get('edge_type', 'unknown')

                        if edge_types is None or edge_type in edge_types:
                            visited.add(predecessor)
                            new_path = f"{path} <- {edge_type} <- {predecessor}" if path else f"<- {edge_type} <- {predecessor}"
                            queue.append((predecessor, depth + 1, new_path))

        # Sort by score
        related.sort(key=lambda x: -x[1])

        return related[:max_results]

    def find_cross_references(
        self,
        node_id: str,
        direction: str = "outgoing"
    ) -> List[Dict]:
        """
        Find all cross-references from/to a node.

        Args:
            node_id: Node to search from
            direction: "outgoing", "incoming", or "both"

        Returns:
            List of reference info dictionaries
        """
        refs = []

        if node_id not in self.graph:
            return refs

        if direction in ("outgoing", "both"):
            for neighbor in self.graph.neighbors(node_id):
                edge = self.graph.edges[node_id, neighbor]
                if edge.get('edge_type') == 'references':
                    target_node = self.node_index.get(neighbor)
                    if target_node:
                        refs.append({
                            "direction": "outgoing",
                            "target_id": neighbor,
                            "target_name": target_node.name,
                            "target_ordinance": target_node.ordinance
                        })

        if direction in ("incoming", "both"):
            for predecessor in self.graph.predecessors(node_id):
                edge = self.graph.edges[predecessor, node_id]
                if edge.get('edge_type') == 'references':
                    source_node = self.node_index.get(predecessor)
                    if source_node:
                        refs.append({
                            "direction": "incoming",
                            "source_id": predecessor,
                            "source_name": source_node.name,
                            "source_ordinance": source_node.ordinance
                        })

        return refs

    def get_definitions_for_terms(
        self,
        terms: List[str]
    ) -> Dict[str, List[Dict]]:
        """
        Get definition nodes for specified terms.

        Args:
            terms: List of terms to look up

        Returns:
            Dict mapping terms to definition info
        """
        results = {}

        for term in terms:
            term_lower = term.lower()
            if term_lower in self.definition_index:
                defs = []
                for node_id in self.definition_index[term_lower]:
                    node = self.node_index.get(node_id)
                    if node:
                        defs.append({
                            "node_id": node_id,
                            "ordinance": node.ordinance,
                            "preview": node.content_preview
                        })
                results[term] = defs

        return results

    def get_section_hierarchy(self, node_id: str) -> List[str]:
        """
        Get the hierarchical path to a section.

        Returns list from ordinance -> part -> section -> subsection
        """
        hierarchy = [node_id]

        current = node_id
        while True:
            parents = [
                pred for pred in self.graph.predecessors(current)
                if self.graph.edges[pred, current].get('edge_type') == 'contains'
            ]

            if not parents:
                break

            parent = parents[0]  # Take first parent
            hierarchy.insert(0, parent)
            current = parent

        return hierarchy

    def save(self, path: Path):
        """Save knowledge graph to disk"""
        path.mkdir(parents=True, exist_ok=True)

        # Save graph structure
        nx.write_gml(self.graph, path / "graph.gml")

        # Save node index
        node_data = {k: vars(v) for k, v in self.node_index.items()}
        with open(path / "nodes.json", 'w') as f:
            json.dump(node_data, f, ensure_ascii=False, indent=2)

        # Save definition index
        with open(path / "definitions.json", 'w') as f:
            json.dump(dict(self.definition_index), f, ensure_ascii=False, indent=2)

        print(f"Knowledge graph saved to {path}")

    def load(self, path: Path):
        """Load knowledge graph from disk"""
        # Load graph structure
        self.graph = nx.read_gml(path / "graph.gml")

        # Load node index
        with open(path / "nodes.json", 'r') as f:
            node_data = json.load(f)
            self.node_index = {
                k: LegalNode(**v) for k, v in node_data.items()
            }

        # Load definition index
        with open(path / "definitions.json", 'r') as f:
            self.definition_index = defaultdict(list, json.load(f))

        print(f"Knowledge graph loaded from {path}: "
              f"{self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")

    def get_statistics(self) -> Dict:
        """Get graph statistics"""
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_types": dict(nx.get_node_attributes(self.graph, 'node_type')),
            "edge_types": defaultdict(int),
            "ordinances": set(nx.get_node_attributes(self.graph, 'ordinance').values()),
            "definitions_indexed": len(self.definition_index)
        }


if __name__ == "__main__":
    # Example usage
    kg = LegalKnowledgeGraph()

    # Load chunks and build graph
    chunks_path = Path("/home/user/legal_advisor_pytorch/data/chunks/all_chunks.json")
    if chunks_path.exists():
        with open(chunks_path) as f:
            chunks = json.load(f)

        kg.build_from_chunks(chunks)

        # Test queries
        print("\n=== Graph Statistics ===")
        stats = kg.get_statistics()
        print(f"Nodes: {stats['total_nodes']}")
        print(f"Edges: {stats['total_edges']}")
        print(f"Definitions: {stats['definitions_indexed']}")

        # Test cross-reference lookup
        test_node = "cap344_s2"
        print(f"\n=== Cross-references from {test_node} ===")
        refs = kg.find_cross_references(test_node, direction="both")
        for ref in refs[:5]:
            print(f"  {ref}")

        # Test related nodes
        print(f"\n=== Related nodes to {test_node} ===")
        related = kg.get_related_nodes(test_node, max_hops=2)
        for node_id, score, path in related[:5]:
            print(f"  {node_id} (score: {score:.2f}): {path}")
