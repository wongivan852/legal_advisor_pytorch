#!/usr/bin/env python3
"""
Build Legal RAG Index

This script processes the HKEL legal documents and builds the complete
RAG index including:
1. Parse XML documents
2. Create hierarchical chunks
3. Generate embeddings
4. Build knowledge graph

Usage:
    python scripts/build_index.py --data-dir data/property_owner_ordinances --output-dir data/index
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing import parse_ordinance_directory, chunk_all_ordinances
from retrieval import HybridRetriever, LegalKnowledgeGraph


def main():
    parser = argparse.ArgumentParser(description="Build Legal RAG Index")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/property_owner_ordinances"),
        help="Directory containing HKEL XML files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/index"),
        help="Output directory for index files"
    )
    parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=2000,
        help="Maximum chunk size in characters"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="BAAI/bge-base-en-v1.5",
        help="HuggingFace embedding model name"
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding generation (for testing)"
    )

    args = parser.parse_args()

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir = args.output_dir / "chunks"
    index_dir = args.output_dir / "vector_index"
    kg_dir = args.output_dir / "knowledge_graph"

    print("=" * 60)
    print("Legal RAG Index Builder")
    print("=" * 60)

    # Step 1: Parse XML documents
    print("\n[1/4] Parsing HKEL XML documents...")
    ordinances = parse_ordinance_directory(args.data_dir)
    print(f"      Parsed {len(ordinances)} ordinances")

    # Step 2: Create chunks
    print("\n[2/4] Creating hierarchical chunks...")
    chunks = chunk_all_ordinances(
        ordinances,
        chunks_dir,
        max_chunk_size=args.max_chunk_size
    )
    print(f"      Created {len(chunks)} chunks")

    # Step 3: Build knowledge graph
    print("\n[3/4] Building knowledge graph...")
    kg = LegalKnowledgeGraph()
    kg.build_from_chunks([c.to_dict() for c in chunks])
    kg.save(kg_dir)
    stats = kg.get_statistics()
    print(f"      Nodes: {stats['total_nodes']}, Edges: {stats['total_edges']}")

    # Step 4: Generate embeddings
    if not args.skip_embeddings:
        print("\n[4/4] Generating embeddings...")
        retriever = HybridRetriever(model_name=args.embedding_model)
        retriever.index_chunks(
            [c.to_dict() for c in chunks],
            save_path=index_dir
        )
        print("      Embeddings generated and saved")
    else:
        print("\n[4/4] Skipping embedding generation")

    print("\n" + "=" * 60)
    print("Index build complete!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  Chunks:     {chunks_dir}")
    print(f"  KG:         {kg_dir}")
    if not args.skip_embeddings:
        print(f"  Embeddings: {index_dir}")

    # Print summary statistics
    print(f"\nDataset Statistics:")
    print(f"  Ordinances:    {len(ordinances)}")
    print(f"  Total Chunks:  {len(chunks)}")

    chunk_types = {}
    for c in chunks:
        ct = c.chunk_type
        chunk_types[ct] = chunk_types.get(ct, 0) + 1
    print(f"  Chunk Types:")
    for ct, count in sorted(chunk_types.items()):
        print(f"    - {ct}: {count}")


if __name__ == "__main__":
    main()
