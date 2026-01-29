#!/usr/bin/env python3
"""
CLI for uploading ordinances and regulatory statements.

Usage:
    # Upload XML file
    python scripts/upload_ordinance.py --file path/to/ordinance.xml

    # Upload text file
    python scripts/upload_ordinance.py --file path/to/statement.txt --title "My Statement"

    # Upload text directly
    python scripts/upload_ordinance.py --text "1. First provision..." --title "Guidelines"

    # List uploads
    python scripts/upload_ordinance.py --list

    # Delete upload
    python scripts/upload_ordinance.py --delete uploaded_abc123
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing.upload_handler import OrdinanceUploadHandler, DynamicIndexUpdater


def main():
    parser = argparse.ArgumentParser(
        description="Upload ordinances and regulatory statements to the Legal RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Upload XML ordinance:
    python scripts/upload_ordinance.py --file cap_999.xml

  Upload text file:
    python scripts/upload_ordinance.py --file guidelines.txt --title "Building Guidelines"

  Upload text directly:
    python scripts/upload_ordinance.py --text "1. All owners shall..." --title "IO Rules"

  List all uploads:
    python scripts/upload_ordinance.py --list

  Delete an upload:
    python scripts/upload_ordinance.py --delete uploaded_abc123

  Rebuild index after uploads:
    python scripts/upload_ordinance.py --rebuild-index
        """
    )

    # Actions
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "--file", "-f",
        type=Path,
        help="Path to file to upload (XML or TXT)"
    )
    action_group.add_argument(
        "--text", "-t",
        type=str,
        help="Text content to upload directly"
    )
    action_group.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all uploaded documents"
    )
    action_group.add_argument(
        "--delete", "-d",
        type=str,
        metavar="ID",
        help="Delete an uploaded document by ID"
    )
    action_group.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild the search index"
    )

    # Options
    parser.add_argument(
        "--title",
        type=str,
        help="Document title (required for text uploads)"
    )
    parser.add_argument(
        "--doc-type",
        type=str,
        default="regulatory_statement",
        choices=["regulatory_statement", "guideline", "notice", "circular", "practice_direction"],
        help="Document type for text uploads"
    )
    parser.add_argument(
        "--custom-id",
        type=str,
        help="Custom identifier for the upload"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "property_owner_ordinances",
        help="Data directory"
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "index",
        help="Index directory"
    )

    args = parser.parse_args()

    # Initialize handler
    handler = OrdinanceUploadHandler(
        data_dir=args.data_dir,
        index_dir=args.index_dir
    )

    # Execute action
    if args.list:
        list_uploads(handler)
    elif args.delete:
        delete_upload(handler, args.delete)
    elif args.rebuild_index:
        rebuild_index(args.index_dir)
    elif args.file:
        upload_file(handler, args)
    elif args.text:
        upload_text(handler, args)


def upload_file(handler: OrdinanceUploadHandler, args):
    """Upload a file"""
    if not args.file.exists():
        print(f"❌ Error: File not found: {args.file}")
        sys.exit(1)

    print(f"📤 Uploading: {args.file}")

    if args.file.suffix.lower() == ".xml":
        result = handler.upload_xml(args.file, custom_id=args.custom_id)
    elif args.file.suffix.lower() == ".txt":
        if not args.title:
            print("❌ Error: --title is required for text files")
            sys.exit(1)

        with open(args.file, 'r', encoding='utf-8') as f:
            text = f.read()

        result = handler.upload_text(
            text=text,
            title=args.title,
            doc_type=args.doc_type,
            custom_id=args.custom_id
        )
    else:
        print(f"❌ Error: Unsupported file type: {args.file.suffix}")
        sys.exit(1)

    print_result(result)


def upload_text(handler: OrdinanceUploadHandler, args):
    """Upload text directly"""
    if not args.title:
        print("❌ Error: --title is required for text uploads")
        sys.exit(1)

    print(f"📤 Uploading text: {args.title}")

    result = handler.upload_text(
        text=args.text,
        title=args.title,
        doc_type=args.doc_type,
        custom_id=args.custom_id
    )

    print_result(result)


def print_result(result):
    """Print upload result"""
    if result.success:
        print("\n✅ Upload successful!")
        print(f"   ID: {result.ordinance_id}")
        print(f"   Name: {result.ordinance_name}")
        print(f"   Chunks created: {result.chunks_created}")
        print(f"   Cross-references: {result.cross_references_found}")
        print(f"   Definitions: {result.definitions_extracted}")

        if result.warnings:
            print("\n⚠️  Warnings:")
            for warning in result.warnings:
                print(f"   - {warning}")

        print(f"\n💡 To rebuild the index, run:")
        print(f"   python scripts/upload_ordinance.py --rebuild-index")
    else:
        print(f"\n❌ Upload failed: {result.error_message}")
        sys.exit(1)


def list_uploads(handler: OrdinanceUploadHandler):
    """List all uploads"""
    uploads = handler.list_uploads()

    if not uploads:
        print("📋 No documents uploaded yet.")
        return

    print(f"\n📋 Uploaded Documents ({len(uploads)} total):\n")
    print("-" * 80)

    for upload in uploads:
        print(f"ID:      {upload['ordinance_id']}")
        print(f"Name:    {upload['ordinance_name']}")
        print(f"Chunks:  {upload['chunks_created']}")
        print(f"Date:    {upload['uploaded_at']}")
        print("-" * 80)


def delete_upload(handler: OrdinanceUploadHandler, ordinance_id: str):
    """Delete an upload"""
    print(f"🗑️  Deleting: {ordinance_id}")

    success = handler.delete_upload(ordinance_id)

    if success:
        print("✅ Deleted successfully")
    else:
        print("❌ Delete failed - document not found")
        sys.exit(1)


def rebuild_index(index_dir: Path):
    """Rebuild the search index"""
    print("🔄 Rebuilding index...")

    try:
        from retrieval import HybridRetriever, LegalKnowledgeGraph

        updater = DynamicIndexUpdater(index_dir=index_dir)
        chunks = updater.get_all_chunks()

        if not chunks:
            print("❌ No chunks found. Please upload some documents first.")
            sys.exit(1)

        print(f"   Found {len(chunks)} chunks")

        # Create retriever and KG
        retriever = HybridRetriever()
        kg = LegalKnowledgeGraph()

        # Build index
        print("   Building vector index...")
        retriever.index_chunks(chunks, save_path=index_dir / "vector_index")

        print("   Building knowledge graph...")
        kg.build_from_chunks(chunks)
        kg.save(index_dir / "knowledge_graph")

        print("\n✅ Index rebuilt successfully!")
        print(f"   Total chunks: {len(chunks)}")
        print(f"   KG nodes: {kg.graph.number_of_nodes()}")
        print(f"   KG edges: {kg.graph.number_of_edges()}")

    except ImportError as e:
        print(f"❌ Error: Missing dependencies - {e}")
        print("   Install with: pip install sentence-transformers networkx")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error rebuilding index: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
