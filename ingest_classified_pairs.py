#!/usr/bin/env python3
"""
Batch ingestion script for pre-classified RFP-Proposal pairs
Uses files from input/rfps/ and input/proposals/ directories
"""

from gcp_rag_multimodal_vertex import VertexAIMultimodalRAG
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# Configuration
RFP_DIR = Path("input/rfps")
PROPOSAL_DIR = Path("input/proposals")
BUCKET_NAME = os.getenv("BUCKET_NAME", "neuracities-rfps-ne1")
INDEX_ID = os.getenv("INDEX_ID")
ENDPOINT_ID = os.getenv("ENDPOINT_ID")
DEPLOYED_INDEX_ID = "deployed_multimodal_rag"

def ingest_all_pairs(dry_run: bool = True):
    """
    Ingest all RFP-Proposal pairs from input directories

    Args:
        dry_run: If True, only prints what would be processed without actual ingestion
    """
    print("=" * 80)
    print("RFP-Proposal Pairs Ingestion (Pre-Classified)")
    print("=" * 80)

    if not RFP_DIR.exists() or not PROPOSAL_DIR.exists():
        print(f"❌ Error: Directories not found")
        print(f"  RFP dir: {RFP_DIR}")
        print(f"  Proposal dir: {PROPOSAL_DIR}")
        return

    # Initialize RAG (skip if dry run and no IDs set)
    rag = None
    if not dry_run:
        if not INDEX_ID or not ENDPOINT_ID:
            print("❌ Error: INDEX_ID and ENDPOINT_ID must be set!")
            print("\nSet these environment variables:")
            print("  export VERTEX_INDEX_ID='projects/.../indexes/...'")
            print("  export VERTEX_ENDPOINT_ID='projects/.../indexEndpoints/...'")
            return

        print(f"\n📦 Initializing RAG with:")
        print(f"  Bucket: {BUCKET_NAME}")
        print(f"  Index: {INDEX_ID}")
        print(f"  Endpoint: {ENDPOINT_ID}")

        rag = VertexAIMultimodalRAG(
            bucket_name=BUCKET_NAME,
            index_id=INDEX_ID,
            endpoint_id=ENDPOINT_ID,
            deployed_index_id=DEPLOYED_INDEX_ID,
            enable_vision_ai=True
        )
        print("✓ RAG initialized\n")

    # Get all RFP files and match with proposals
    rfp_files = sorted(RFP_DIR.glob("project*_rfp.pdf"))

    print(f"\n📁 Found {len(rfp_files)} RFP-Proposal pairs\n")

    stats = {
        "total_pairs": 0,
        "rfps_processed": 0,
        "proposals_processed": 0,
        "errors": 0,
        "total_chunks": 0
    }

    for idx, rfp_file in enumerate(rfp_files, 1):
        # Extract project number from filename (e.g., project1_rfp.pdf -> 1)
        project_num = rfp_file.stem.replace("_rfp", "").replace("project", "")
        project_id = f"project{project_num}"

        # Find matching proposal
        proposal_file = PROPOSAL_DIR / f"project{project_num}_proposal.pdf"

        if not proposal_file.exists():
            print(f"\n[{idx}/{len(rfp_files)}] ⚠️  Missing proposal for {project_id}")
            continue

        print(f"\n[{idx}/{len(rfp_files)}] {project_id}")
        print("-" * 80)

        # Process RFP
        rfp_size_mb = rfp_file.stat().st_size / (1024 * 1024)
        print(f"\n  📄 RFP: {rfp_file.name}")
        print(f"     Size: {rfp_size_mb:.2f} MB")

        if dry_run:
            print(f"     [DRY RUN] Would process this file")
        else:
            try:
                print(f"     Processing... ", end="", flush=True)

                result = rag.process_and_index_document(
                    local_pdf_path=str(rfp_file),
                    project_id=project_id,
                    document_type="rfp"
                )

                print(f"✓ Done")
                print(f"     Chunks: {result['total_chunks']} total")
                print(f"       - Text: {result['text_chunks']}")
                print(f"       - Tables: {result['table_chunks']}")
                print(f"       - Images: {result['image_chunks']}")

                stats['total_chunks'] += result['total_chunks']
                stats['rfps_processed'] += 1

            except Exception as e:
                print(f"❌ Error: {str(e)}")
                stats['errors'] += 1

        # Process Proposal
        proposal_size_mb = proposal_file.stat().st_size / (1024 * 1024)
        print(f"\n  📝 Proposal: {proposal_file.name}")
        print(f"     Size: {proposal_size_mb:.2f} MB")

        if dry_run:
            print(f"     [DRY RUN] Would process this file")
        else:
            try:
                print(f"     Processing... ", end="", flush=True)

                result = rag.process_and_index_document(
                    local_pdf_path=str(proposal_file),
                    project_id=project_id,
                    document_type="proposal"
                )

                print(f"✓ Done")
                print(f"     Chunks: {result['total_chunks']} total")
                print(f"       - Text: {result['text_chunks']}")
                print(f"       - Tables: {result['table_chunks']}")
                print(f"       - Images: {result['image_chunks']}")

                stats['total_chunks'] += result['total_chunks']
                stats['proposals_processed'] += 1

            except Exception as e:
                print(f"❌ Error: {str(e)}")
                stats['errors'] += 1

        stats['total_pairs'] += 1

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if dry_run:
        print("\n[DRY RUN MODE] - No files were actually processed")
        print(f"\nFound:")
        print(f"  • {len(rfp_files)} RFP-Proposal pairs")
        print(f"  • {len(rfp_files) * 2} PDF files total")
        print("\nTo actually ingest, run:")
        print("  python ingest_classified_pairs.py --process")
    else:
        print(f"\n✓ Processed {stats['total_pairs']} pairs")
        print(f"  • RFPs: {stats['rfps_processed']}")
        print(f"  • Proposals: {stats['proposals_processed']}")
        print(f"  • Total chunks: {stats['total_chunks']}")
        print(f"  • Errors: {stats['errors']}")

        if stats['errors'] > 0:
            print("\n⚠️  Some files failed to process. Check logs above.")

        print("\n✅ Ingestion complete! You can now query your data.")
        print("\nTry:")
        print("  python example_queries.py")

if __name__ == "__main__":
    import sys

    # Check if --process flag is passed
    dry_run = "--process" not in sys.argv

    if dry_run:
        print("\n🔍 Running in DRY RUN mode - no files will be processed")
        print("To actually process files, run: python ingest_classified_pairs.py --process\n")
    else:
        print("\n⚡ PROCESSING MODE - files will be ingested into Vertex AI\n")

    ingest_all_pairs(dry_run=dry_run)
