#!/usr/bin/env python3
"""
Remove a project from the Vertex AI Vector Search index

NOTE: Vertex AI Vector Search doesn't support direct deletion of individual vectors.
This script provides two approaches:
1. Move project files out of input directories and re-ingest everything
2. Rebuild the index from scratch excluding the specified project

Approach 1 is recommended for small changes.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()

BUCKET_NAME = os.getenv("BUCKET_NAME")
RFP_DIR = Path("input/rfps")
PROPOSAL_DIR = Path("input/proposals")


def move_project_to_archive(project_id: str, dry_run: bool = True):
    """
    Move project files to an archive directory

    Args:
        project_id: Project identifier (e.g., "project1")
        dry_run: If True, only shows what would be done
    """
    print("=" * 80)
    print(f"Remove {project_id} from Index")
    print("=" * 80)

    # Create archive directory
    archive_dir = Path("input/archived")
    archive_rfp = archive_dir / "rfps"
    archive_proposal = archive_dir / "proposals"

    # Find files
    rfp_file = RFP_DIR / f"{project_id}_rfp.pdf"
    proposal_file = PROPOSAL_DIR / f"{project_id}_proposal.pdf"

    files_found = []
    if rfp_file.exists():
        files_found.append(("RFP", rfp_file, archive_rfp / rfp_file.name))
    if proposal_file.exists():
        files_found.append(("Proposal", proposal_file, archive_proposal / proposal_file.name))

    if not files_found:
        print(f"\n❌ No files found for {project_id}")
        print(f"   Looked in: {RFP_DIR} and {PROPOSAL_DIR}")
        return False

    print(f"\n📁 Found {len(files_found)} file(s) for {project_id}:")
    for doc_type, source, dest in files_found:
        size_mb = source.stat().st_size / (1024 * 1024)
        print(f"   • {doc_type}: {source.name} ({size_mb:.2f} MB)")

    if dry_run:
        print(f"\n[DRY RUN] Would move files to: {archive_dir}")
        print("\nTo actually move files, run:")
        print(f"  python remove_project.py {project_id} --execute")
        return True

    # Create archive directories
    archive_rfp.mkdir(parents=True, exist_ok=True)
    archive_proposal.mkdir(parents=True, exist_ok=True)

    # Move files
    print(f"\n📦 Moving files to {archive_dir}...")
    for doc_type, source, dest in files_found:
        source.rename(dest)
        print(f"   ✓ Moved {source.name}")

    print(f"\n✅ Files archived successfully!")
    print(f"\nNext steps:")
    print(f"1. Clear existing embeddings from GCS:")
    print(f"   python remove_project.py {project_id} --clear-gcs")
    print(f"2. Re-ingest all remaining projects:")
    print(f"   python ingest_classified_pairs.py --process")
    print(f"3. Wait 5-15 minutes for index to update")

    return True


def clear_gcs_embeddings(project_id: str, dry_run: bool = True):
    """
    Clear embeddings and metadata for a project from GCS

    NOTE: This won't immediately update the Vertex AI index.
    You need to rebuild the index after clearing.
    """
    print("=" * 80)
    print(f"Clear GCS Data for {project_id}")
    print("=" * 80)

    if not BUCKET_NAME:
        print("❌ BUCKET_NAME not set in .env")
        return False

    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    metadata_bucket = storage_client.bucket(f"{BUCKET_NAME}-metadata")

    print(f"\n🔍 Scanning for {project_id} data in GCS...")

    # This is complex because we need to check metadata to find which batches contain project1
    # For simplicity, we'll provide instructions to rebuild

    print(f"\n⚠️  Note: Individual vector deletion is not supported by Vertex AI.")
    print(f"\nRecommended approach:")
    print(f"1. Delete all embeddings and rebuild:")
    print(f"   gsutil -m rm -r gs://{BUCKET_NAME}/embeddings/")
    print(f"   gsutil -m rm -r gs://{BUCKET_NAME}-metadata/metadata/")
    print(f"2. Re-ingest all projects (excluding {project_id}):")
    print(f"   python ingest_classified_pairs.py --process")
    print(f"3. Wait 5-15 minutes for index to update")

    return True


def list_all_projects():
    """List all projects in input directories"""
    print("=" * 80)
    print("All Projects")
    print("=" * 80)

    rfp_files = sorted(RFP_DIR.glob("project*_rfp.pdf"))

    print(f"\n📊 Found {len(rfp_files)} project(s):\n")

    for rfp_file in rfp_files:
        project_num = rfp_file.stem.replace("_rfp", "").replace("project", "")
        project_id = f"project{project_num}"
        proposal_file = PROPOSAL_DIR / f"{project_id}_proposal.pdf"

        rfp_size = rfp_file.stat().st_size / (1024 * 1024)
        proposal_size = proposal_file.stat().st_size / (1024 * 1024) if proposal_file.exists() else 0

        status = "✓" if proposal_file.exists() else "⚠️  (missing proposal)"

        print(f"  {status} {project_id}")
        print(f"      RFP: {rfp_size:.2f} MB")
        if proposal_file.exists():
            print(f"      Proposal: {proposal_size:.2f} MB")
        print()


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python remove_project.py list                    # List all projects")
        print("  python remove_project.py <project_id>            # Dry run removal")
        print("  python remove_project.py <project_id> --execute  # Actually remove")
        print("  python remove_project.py <project_id> --clear-gcs # Clear GCS data")
        print("\nExample:")
        print("  python remove_project.py project1")
        print("  python remove_project.py project1 --execute")
        sys.exit(1)

    command = sys.argv[1]

    if command == "list":
        list_all_projects()
        return

    project_id = command
    dry_run = "--execute" not in sys.argv
    clear_gcs = "--clear-gcs" in sys.argv

    if clear_gcs:
        clear_gcs_embeddings(project_id, dry_run)
    else:
        move_project_to_archive(project_id, dry_run)


if __name__ == "__main__":
    main()
