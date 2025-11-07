#!/usr/bin/env python3
"""
Working example queries with progress indicators
"""
import sys
import os
from dotenv import load_dotenv
from gcp_rag_multimodal_vertex import VertexAIMultimodalRAG

# Load environment variables from .env file
load_dotenv()

# Get variables from environment
BUCKET_NAME = os.getenv("BUCKET_NAME")
INDEX_ID = os.getenv("INDEX_ID")
ENDPOINT_ID = os.getenv("ENDPOINT_ID")

def print_flush(msg):
    """Print and flush immediately"""
    print(msg)
    sys.stdout.flush()



print_flush("Initializing RAG system...")
rag = VertexAIMultimodalRAG(
    bucket_name=BUCKET_NAME,
    index_id=INDEX_ID,
    endpoint_id=ENDPOINT_ID,
    enable_vision_ai=True  # Now uses GCP Gemini Vision (no external API needed)
)
print_flush("✓ RAG initialized\n")

# Example 1
print_flush("=" * 80)
print_flush("Example 1: Search Across All Projects")
print_flush("=" * 80)
print_flush("Querying... (this may take a moment)")
sys.stdout.flush()

results = rag.query("What are the public engagement requirements?", top_k=5)

print_flush(f"\n✓ Found {len(results)} results:\n")
for i, result in enumerate(results, 1):
    print_flush(f"[{i}] Score: {result['score']:.3f}")
    print_flush(f"Type: {result['content_type']} | Doc: {result['document_type']} | Project: {result.get('project_id', 'N/A')}")
    print_flush(f"Content: {result['content'][:200]}...")
    print_flush(f"Source: Page {result.get('page', '?')}\n")

# Example 2
print_flush("\n" + "=" * 80)
print_flush("Example 2: Filter by RFPs Only")
print_flush("=" * 80)
print_flush("Querying RFPs...")
sys.stdout.flush()

results = rag.query("What are the budget and timeline requirements?", top_k=3, filters={"document_type": "rfp"})

print_flush(f"\n✓ Found {len(results)} RFP results:\n")
for i, result in enumerate(results, 1):
    print_flush(f"[{i}] Score: {result['score']:.3f}")
    print_flush(f"Project: {result.get('project_id', 'N/A')}")
    print_flush(f"Content: {result['content'][:150]}...\n")

# Example 3
print_flush("\n" + "=" * 80)
print_flush("Example 3: Filter by Specific Project (project7)")
print_flush("=" * 80)
print_flush("Querying project7...")
sys.stdout.flush()

results = rag.query("methodology and approach", top_k=3, filters={"project_id": "project7"})

print_flush(f"\n✓ Found {len(results)} results in project7:\n")
for i, result in enumerate(results, 1):
    print_flush(f"[{i}] Score: {result['score']:.3f}")
    print_flush(f"Type: {result['document_type']}/{result['content_type']}")
    print_flush(f"Content: {result['content'][:150]}...\n")

# Example 4
print_flush("\n" + "=" * 80)
print_flush("Example 4: Search Tables Only")
print_flush("=" * 80)
print_flush("Querying tables...")
sys.stdout.flush()

results = rag.query("cost breakdowns and budget", top_k=3, filters={"content_type": "table"})

print_flush(f"\n✓ Found {len(results)} table results:\n")
for i, result in enumerate(results, 1):
    print_flush(f"[{i}] Score: {result['score']:.3f}")
    print_flush(f"Project: {result.get('project_id', 'N/A')} | Doc: {result['document_type']}")
    print_flush(f"Table: {result['content'][:200]}...\n")

print_flush("\n" + "=" * 80)
print_flush("✅ All Examples Complete!")
print_flush("=" * 80)
print_flush("\nThe RAG system is working correctly.")
print_flush("Results are being retrieved from your indexed data.")
