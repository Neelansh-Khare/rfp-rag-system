#!/usr/bin/env python3
"""
Enhanced GCP RAG with Vertex AI Vector Search + Multimodal Processing

Advantages over Firestore:
- Purpose-built for vector similarity search
- Better performance and scalability
- Supports both batch and streaming updates
- More sophisticated filtering capabilities
- Production-grade for high-volume workloads
"""

from google.cloud import storage, aiplatform
from google.cloud.aiplatform import MatchingEngineIndex, MatchingEngineIndexEndpoint
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel, Part
import vertexai
import json
import uuid
from typing import List, Dict, Optional
import os
import logging

# Multimodal processing imports
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import tabula
import camelot
import io
import base64

logger = logging.getLogger(__name__)

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "neuracities-pilot")  # Changed default
LOCATION = "northamerica-northeast1"  # Changed to match bucket location
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "neuracities-rfps-ne1")  # Changed default

vertexai.init(project=PROJECT_ID, location=LOCATION)
aiplatform.init(project=PROJECT_ID, location=LOCATION)


class VertexAIMultimodalRAG:
    """Enhanced RAG using Vertex AI Vector Search with full multimodal capabilities"""

    def __init__(
        self,
        bucket_name: str,
        index_id: Optional[str] = None,
        endpoint_id: Optional[str] = None,
        deployed_index_id: Optional[str] = None,
        enable_vision_ai: bool = True
    ):
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(bucket_name)
        self.bucket_name = bucket_name
        self.embedding_model = TextEmbeddingModel.from_pretrained(
            "text-embedding-004"  # Available in northamerica-northeast1
        )

        # Vector Search components
        self.index_id = index_id
        self.endpoint_id = endpoint_id
        self.deployed_index_id = deployed_index_id or "deployed_multimodal_rag"
        self.index = None
        self.endpoint = None

        # Load existing resources if IDs provided
        if self.index_id:
            try:
                self.index = aiplatform.MatchingEngineIndex(self.index_id)
                logger.info(f"Loaded existing index: {self.index_id}")
            except Exception as e:
                logger.warning(f"Could not load index: {e}")

        if self.endpoint_id:
            try:
                self.endpoint = aiplatform.MatchingEngineIndexEndpoint(self.endpoint_id)
                logger.info(f"Loaded existing endpoint: {self.endpoint_id}")
            except Exception as e:
                logger.warning(f"Could not load endpoint: {e}")

        # Metadata storage (Cloud Storage for document metadata)
        self.metadata_bucket = f"{bucket_name}-metadata"
        self._ensure_metadata_bucket()

        # Vision AI setup - using GCP Gemini Vision (no external API needed)
        self.enable_vision_ai = enable_vision_ai
        if enable_vision_ai:
            logger.info("Vision AI enabled with Vertex AI Gemini Vision")

    def _ensure_metadata_bucket(self):
        """Create metadata bucket if it doesn't exist"""
        try:
            bucket = self.storage_client.bucket(self.metadata_bucket)
            if not bucket.exists():
                bucket = self.storage_client.create_bucket(
                    self.metadata_bucket,
                    location=LOCATION
                )
                logger.info(f"Created metadata bucket: {self.metadata_bucket}")
        except Exception as e:
            logger.warning(f"Metadata bucket check: {e}")

    # ==================== VERTEX AI VECTOR SEARCH SETUP ====================

    def create_vector_index(
        self,
        display_name: str = "multimodal-rag-index",
        dimensions: int = 768
    ) -> str:
        """Create a new Vertex AI Vector Search index"""

        logger.info("Creating Vertex AI Vector Search index (this takes ~30 minutes)...")

        # Create the index
        index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
            display_name=display_name,
            contents_delta_uri=f"gs://{self.bucket.name}/embeddings",
            dimensions=dimensions,
            approximate_neighbors_count=100,
            distance_measure_type="COSINE_DISTANCE",
            leaf_node_embedding_count=1000,
            leaf_nodes_to_search_percent=100,
            description="Multimodal RAG Index with text, tables, and images"
        )

        self.index_id = index.resource_name
        self.index = index
        logger.info(f"Created index: {self.index_id}")
        return self.index_id

    def create_index_endpoint(
        self,
        display_name: str = "multimodal-rag-endpoint"
    ) -> str:
        """Create and deploy index endpoint"""

        logger.info("Creating index endpoint...")

        endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
            display_name=display_name,
            description="Endpoint for multimodal RAG queries",
            public_endpoint_enabled=True
        )

        self.endpoint_id = endpoint.resource_name
        self.endpoint = endpoint
        logger.info(f"Created endpoint: {self.endpoint_id}")

        # Deploy the index to the endpoint if index exists
        if self.index:
            logger.info("Deploying index to endpoint (takes ~10 minutes)...")
            endpoint.deploy_index(
                index=self.index,
                deployed_index_id=self.deployed_index_id,
                display_name=display_name,
                machine_type="e2-standard-16",  # Changed from e2-standard-2 (required for SHARD_SIZE_MEDIUM)
                min_replica_count=1,
                max_replica_count=2
            )
            logger.info("Index deployed successfully")

        return self.endpoint_id

    # ==================== CORE GCS & UPLOAD ====================

    def upload_pdf(self, local_path: str, gcs_path: str) -> str:
        """Upload PDF to Cloud Storage"""
        try:
            blob = self.bucket.blob(gcs_path)
            blob.upload_from_filename(local_path)
            gcs_uri = f"gs://{self.bucket.name}/{gcs_path}"
            logger.info(f"Uploaded to {gcs_uri}")
            return gcs_uri
        except Exception as e:
            logger.error(f"Error uploading PDF: {e}")
            raise

    # ==================== IMAGE PROCESSING ====================

    def extract_images_from_page(self, page: fitz.Page) -> List[Dict]:
        """Extract all images from a PDF page with robust colorspace handling"""
        images = []
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                pix = fitz.Pixmap(page.parent, xref)

                pil_image = None
                img_data = None

                # Try direct PNG conversion
                try:
                    img_data = pix.tobytes("png")
                    pil_image = Image.open(io.BytesIO(img_data))
                except Exception:
                    # Convert CMYK/other colorspaces to RGB
                    try:
                        if pix.colorspace and pix.colorspace.n == 4:  # CMYK
                            pix_rgb = fitz.Pixmap(fitz.csRGB, pix)
                            pix = pix_rgb
                        img_data = pix.tobytes("png")
                        pil_image = Image.open(io.BytesIO(img_data))
                    except Exception:
                        # Fallback: Extract raw image
                        try:
                            base_image = page.parent.extract_image(xref)
                            image_bytes = base_image["image"]
                            pil_image = Image.open(io.BytesIO(image_bytes))
                            if pil_image.mode in ("CMYK", "LAB", "YCbCr"):
                                pil_image = pil_image.convert("RGB")
                            buffer = io.BytesIO()
                            pil_image.save(buffer, format="PNG")
                            img_data = buffer.getvalue()
                        except Exception as e:
                            logger.error(f"All extraction methods failed: {e}")
                            continue

                if pil_image is None:
                    continue

                # Skip small decorative images
                if pil_image.width < 100 or pil_image.height < 100:
                    continue

                bbox = page.get_image_bbox(img)
                image_info = {
                    "index": img_index,
                    "width": pil_image.width,
                    "height": pil_image.height,
                    "bbox": bbox,
                    "pil_image": pil_image,
                    "image_data": img_data
                }
                images.append(image_info)

            except Exception as e:
                logger.error(f"Error extracting image {img_index}: {e}")
                continue

        return images

    def perform_ocr(self, pil_image: Image.Image) -> str:
        """Extract text from image using OCR with enhancement"""
        try:
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            # Enhance image for OCR
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            denoised = cv2.fastNlMeansDenoising(gray)
            enhanced = cv2.convertScaleAbs(denoised, alpha=1.5, beta=0)
            enhanced_image = Image.fromarray(cv2.cvtColor(
                cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR),
                cv2.COLOR_BGR2RGB
            ))

            # Perform OCR
            text = pytesseract.image_to_string(enhanced_image, config='--psm 6')
            text = ' '.join(text.split())

            return text.strip() if len(text.strip()) >= 3 else ""
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return ""

    def analyze_image_with_vision_ai(self, pil_image: Image.Image, context: str = "") -> str:
        """Analyze image content using Vertex AI Gemini Vision (GCP-native)"""
        try:
            # Convert PIL Image to bytes
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            image_bytes = buffered.getvalue()

            # Create Gemini Vision model
            vision_model = GenerativeModel("gemini-1.5-flash")

            prompt = """Analyze this image from an RFP or Proposal document. Describe:
1. Type of content (chart, diagram, table, screenshot, etc.)
2. Key information and data points
3. Important text or labels
4. Relationships or processes shown
5. Relevant business/technical details

Focus on extracting actionable information for project planning, technical requirements,
budget allocation, and process flows. Be concise but comprehensive."""

            if context:
                prompt += f"\n\nContext: {context}"

            # Create image part from bytes
            image_part = Part.from_data(image_bytes, mime_type="image/png")

            # Generate content with image
            response = vision_model.generate_content(
                [prompt, image_part],
                generation_config={
                    "max_output_tokens": 500,
                    "temperature": 0.2,
                }
            )

            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini Vision AI failed: {e}")
            return ""

    def classify_image_content(self, pil_image: Image.Image) -> Dict:
        """Classify image type based on properties"""
        width, height = pil_image.size
        aspect_ratio = width / height

        if aspect_ratio > 2.0:
            img_type = "timeline_or_process"
        elif 0.8 <= aspect_ratio <= 1.2:
            img_type = "chart_or_diagram"
        elif width > 600 and height > 400:
            img_type = "screenshot_or_detailed_diagram"
        else:
            img_type = "small_chart_or_icon"

        return {
            "type": img_type,
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio
        }

    def process_image_comprehensive(self, pil_image: Image.Image, page_context: str = "") -> Dict:
        """Comprehensive image processing combining all methods"""
        results = {
            "ocr_text": "",
            "vision_analysis": "",
            "classification": {},
            "combined_description": ""
        }

        try:
            results["classification"] = self.classify_image_content(pil_image)
            results["ocr_text"] = self.perform_ocr(pil_image)

            if self.enable_vision_ai:
                results["vision_analysis"] = self.analyze_image_with_vision_ai(
                    pil_image, page_context
                )

            # Create combined description
            parts = [f"Image type: {results['classification']['type']}"]
            if results["ocr_text"]:
                parts.append(f"Text: {results['ocr_text']}")
            if results["vision_analysis"]:
                parts.append(f"Analysis: {results['vision_analysis']}")

            results["combined_description"] = " | ".join(parts)
            return results

        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            return results

    # ==================== TABLE EXTRACTION ====================

    def extract_tables_from_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract tables using multiple methods (Camelot, Tabula, PyMuPDF)"""
        all_tables = []

        # Method 1: Camelot (best for well-formatted tables)
        try:
            lattice_tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
            for table in lattice_tables:
                if self._is_valid_table(table.df):
                    all_tables.append({
                        'page': table.page,
                        'dataframe': table.df,
                        'accuracy': table.accuracy,
                        'method': 'camelot_lattice'
                    })
        except Exception as e:
            logger.warning(f"Camelot extraction failed: {e}")

        # Method 2: Tabula
        try:
            dfs = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
            for i, df in enumerate(dfs):
                if self._is_valid_table(df):
                    all_tables.append({
                        'page': i + 1,
                        'dataframe': df,
                        'accuracy': 0.7,
                        'method': 'tabula'
                    })
        except Exception as e:
            logger.warning(f"Tabula extraction failed: {e}")

        # Deduplicate
        return self._deduplicate_tables(all_tables)

    def _is_valid_table(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame is a valid table"""
        if df is None or df.empty or df.shape[0] < 2 or df.shape[1] < 2:
            return False

        content_ratio = sum(
            1 for col in df.columns
            for val in df[col]
            if pd.notna(val) and str(val).strip() and len(str(val).strip()) > 1
        )
        total_cells = df.shape[0] * df.shape[1]
        return (content_ratio / total_cells) > 0.3

    def _deduplicate_tables(self, tables: List[Dict]) -> List[Dict]:
        """Remove duplicate tables, keeping best quality"""
        if not tables:
            return []

        by_page = {}
        for table in tables:
            page = table['page']
            if page not in by_page:
                by_page[page] = []
            by_page[page].append(table)

        deduplicated = []
        for page_tables in by_page.values():
            best = max(page_tables, key=lambda t: t['accuracy'])
            deduplicated.append(best)

        return sorted(deduplicated, key=lambda x: x['page'])

    def convert_table_to_text(self, table_info: Dict) -> str:
        """Convert table to searchable text"""
        df = table_info['dataframe']
        page = table_info['page']
        method = table_info['method']

        description = f"Table on page {page} (via {method}):\n"
        description += f"{df.shape[0]} rows × {df.shape[1]} columns\n"

        try:
            df_clean = df.fillna('')
            table_str = df_clean.to_string(index=False, max_colwidth=50)
            description += f"Content:\n{table_str}"
        except Exception as e:
            logger.error(f"Table conversion failed: {e}")

        return description

    # ==================== DOCUMENT PROCESSING ====================

    def process_pdf_comprehensive(self, pdf_path: str) -> Dict:
        """Extract all content: text, images, tables"""
        doc = fitz.open(pdf_path)

        content = {
            "text": "",
            "images": [],
            "tables": []
        }

        try:
            # Extract text
            full_text = []
            for page_num in range(doc.page_count):
                page = doc[page_num]
                page_text = page.get_text()
                full_text.append(f"--- Page {page_num + 1} ---\n{page_text}")

                # Extract images
                page_images = self.extract_images_from_page(page)
                for img_info in page_images:
                    processed = self.process_image_comprehensive(
                        img_info["pil_image"],
                        f"Page {page_num + 1}"
                    )
                    img_info.update(processed)
                    img_info["page"] = page_num + 1
                    content["images"].append(img_info)

            content["text"] = "\n".join(full_text)

            # Extract tables
            content["tables"] = self.extract_tables_from_pdf(pdf_path)

            logger.info(f"Extracted: {len(content['images'])} images, {len(content['tables'])} tables")

        finally:
            doc.close()

        return content

    # ==================== CHUNKING & EMBEDDING ====================

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Chunk text with overlap"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap
        return chunks

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Vertex AI"""
        try:
            embeddings = []
            batch_size = 5
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self.embedding_model.get_embeddings(batch)
                embeddings.extend([emb.values for emb in batch_embeddings])
            return embeddings
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise

    # ==================== VERTEX AI STORAGE ====================

    def store_embeddings_batch(self, chunks: List[Dict]) -> str:
        """Store embeddings in format for Vertex AI Vector Search"""

        embeddings_data = []
        metadata_docs = {}

        for chunk in chunks:
            # Generate embedding
            embedding = self.generate_embeddings([chunk["content"]])[0]

            # Create vector search record
            vector_id = chunk["chunk_id"]
            embeddings_data.append({
                "id": vector_id,
                "embedding": embedding,
                # Vertex AI filtering via restricts
                "restricts": [
                    {"namespace": "project_id", "allow": [chunk.get("project_id", "")]},
                    {"namespace": "document_type", "allow": [chunk.get("document_type", "")]},
                    {"namespace": "content_type", "allow": [chunk.get("content_type", "text")]}
                ]
            })

            # Store full metadata separately
            metadata_docs[vector_id] = {
                "content": chunk["content"],
                "project_id": chunk.get("project_id"),
                "document_type": chunk.get("document_type"),
                "content_type": chunk.get("content_type"),
                "source": chunk.get("source"),
                "page": chunk.get("page", 0),
                "metadata": chunk.get("metadata", {})
            }

        # Save embeddings in JSON format for Vertex AI (JSONL content with .json extension)
        timestamp = uuid.uuid4().hex
        embeddings_path = f"embeddings/batch_{timestamp}.json"

        blob = self.bucket.blob(embeddings_path)
        jsonl_content = "\n".join([json.dumps(record) for record in embeddings_data])
        blob.upload_from_string(jsonl_content, content_type='application/json')

        # Save metadata
        metadata_path = f"metadata/batch_{timestamp}.json"
        metadata_bucket = self.storage_client.bucket(self.metadata_bucket)
        metadata_blob = metadata_bucket.blob(metadata_path)
        metadata_blob.upload_from_string(json.dumps(metadata_docs))

        logger.info(f"Stored {len(chunks)} embeddings and metadata")

        # Update index (if exists)
        if self.index:
            try:
                self.index.update_embeddings(
                    contents_delta_uri=f"gs://{self.bucket.name}/embeddings"
                )
                logger.info("Index updated with new embeddings")
            except Exception as e:
                logger.warning(f"Could not update index automatically: {e}")

        return embeddings_path

    # ==================== QUERY & SEARCH ====================

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        filters: Dict = None
    ) -> List[Dict]:
        """Query using Vertex AI Vector Search"""

        if not self.endpoint:
            raise ValueError("No endpoint configured. Create one with create_index_endpoint()")

        # Generate query embedding
        query_embedding = self.generate_embeddings([query_text])[0]

        # Query the endpoint
        try:
            # Vertex AI Vector Search doesn't support native filtering in find_neighbors
            # We'll retrieve more results and filter in post-processing
            response = self.endpoint.find_neighbors(
                deployed_index_id=self.deployed_index_id,
                queries=[query_embedding],
                num_neighbors=top_k * 10 if filters else top_k  # Get more results if filtering
            )
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise

        # Retrieve metadata and build results
        results = []
        for neighbors in response:
            for neighbor in neighbors:
                vector_id = neighbor.id
                distance = neighbor.distance

                # Retrieve metadata from storage
                metadata = self._get_metadata(vector_id)
                if metadata:
                    # Apply post-processing filters
                    if filters:
                        matches = True
                        for field, value in filters.items():
                            if metadata.get(field) != value:
                                matches = False
                                break
                        if not matches:
                            continue

                    results.append({
                        "content": metadata.get("content", ""),
                        "source": metadata.get("source", ""),
                        "project_id": metadata.get("project_id", ""),
                        "document_type": metadata.get("document_type", ""),
                        "content_type": metadata.get("content_type", "text"),
                        "page": metadata.get("page", 0),
                        "score": 1 - distance  # Convert distance to similarity
                    })

                    # Stop if we have enough filtered results
                    if len(results) >= top_k:
                        break
            if len(results) >= top_k:
                break

        return results[:top_k]

    def _get_metadata(self, vector_id: str) -> Dict:
        """Retrieve metadata from storage"""
        try:
            # Search for metadata file containing this vector_id
            bucket = self.storage_client.bucket(self.metadata_bucket)
            for blob in bucket.list_blobs(prefix="metadata/"):
                content = json.loads(blob.download_as_text())
                if vector_id in content:
                    return content[vector_id]
        except Exception as e:
            logger.error(f"Failed to retrieve metadata for {vector_id}: {e}")
        return {}

    def generate_answer(
        self,
        query_text: str,
        top_k: int = 5,
        filters: Dict = None,
        model_name: str = "gemini-2.0-flash-001"
    ) -> Dict:
        """Generate answer using RAG with Gemini"""
        try:
            contexts = self.query(query_text, top_k, filters)

            if not contexts:
                return {
                    "answer": "No relevant information found.",
                    "sources": [],
                    "contexts": []
                }

            # Build context with content type indicators
            context_parts = []
            for ctx in contexts:
                content_type = ctx.get("content_type", "text")
                prefix = f"[{content_type.upper()}]" if content_type != "text" else ""
                context_parts.append(f"{prefix} {ctx['content']}")

            context = "\n\n".join(context_parts)

            model = GenerativeModel(model_name)
            prompt = f"""Based on the following context from RFP and Proposal documents, answer the question.

Context:
{context}

Question: {query_text}

Answer (be specific and cite sources):"""

            response = model.generate_content(prompt)

            return {
                "answer": response.text,
                "sources": [ctx["source"] for ctx in contexts],
                "contexts": contexts
            }
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            raise

    # ==================== FULL PIPELINE ====================

    def process_and_index_document(
        self,
        local_pdf_path: str,
        project_id: str,
        document_type: str
    ) -> Dict:
        """Full pipeline with multimodal processing"""
        try:
            # Upload to GCS
            gcs_path = f"pdfs/{project_id}_{document_type}.pdf"
            gcs_uri = self.upload_pdf(local_pdf_path, gcs_path)

            # Extract all content
            content = self.process_pdf_comprehensive(local_pdf_path)

            # Chunk text
            text_chunks = self.chunk_text(content["text"])

            all_chunks = []

            # Process text chunks
            for i, chunk in enumerate(text_chunks):
                all_chunks.append({
                    "chunk_id": f"{project_id}_{document_type}_text_{i}",
                    "content": chunk,
                    "project_id": project_id,
                    "document_type": document_type,
                    "content_type": "text",
                    "source": f"{project_id}_{document_type}.pdf",
                    "metadata": {}
                })

            # Process table chunks
            for i, table in enumerate(content["tables"]):
                table_text = self.convert_table_to_text(table)
                all_chunks.append({
                    "chunk_id": f"{project_id}_{document_type}_table_{i}",
                    "content": table_text,
                    "project_id": project_id,
                    "document_type": document_type,
                    "content_type": "table",
                    "source": f"{project_id}_{document_type}.pdf",
                    "page": table["page"],
                    "metadata": {"method": table["method"]}
                })

            # Process image chunks
            for i, img in enumerate(content["images"]):
                if img.get("combined_description"):
                    all_chunks.append({
                        "chunk_id": f"{project_id}_{document_type}_image_{i}",
                        "content": f"IMAGE: {img['combined_description']}",
                        "project_id": project_id,
                        "document_type": document_type,
                        "content_type": "image",
                        "source": f"{project_id}_{document_type}.pdf",
                        "page": img.get("page", 0),
                        "metadata": {
                            "image_type": img["classification"]["type"],
                            "has_ocr": bool(img.get("ocr_text"))
                        }
                    })

            # Store in Vertex AI Vector Search
            self.store_embeddings_batch(all_chunks)

            logger.info(f"Indexed {len(all_chunks)} chunks: "
                       f"{len(text_chunks)} text, "
                       f"{len(content['tables'])} tables, "
                       f"{len(content['images'])} images")

            return {
                "success": True,
                "total_chunks": len(all_chunks),
                "text_chunks": len(text_chunks),
                "table_chunks": len(content["tables"]),
                "image_chunks": len(content["images"])
            }

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise


# ==================== SETUP HELPER ====================

def print_setup_instructions():
    """Print setup instructions"""
    print("""
=================================================================
Vertex AI Vector Search Setup Instructions
=================================================================

1. Enable APIs:
   gcloud services enable aiplatform.googleapis.com

2. Set environment variables:
   export GCP_PROJECT_ID="your-project-id"
   export GCS_BUCKET_NAME="your-bucket-name"
   export OPENAI_API_KEY="your-openai-key"  # Optional

3. Initialize and create index (ONE TIME - takes ~30 minutes):

   from gcp_rag_multimodal_vertex import VertexAIMultimodalRAG

   rag = VertexAIMultimodalRAG(bucket_name="your-bucket")
   index_id = rag.create_vector_index()

   # Save this: {index_id}

4. Create endpoint (ONE TIME - takes ~10 minutes):

   endpoint_id = rag.create_index_endpoint()

   # Save this: {endpoint_id}

5. Use for queries (use saved IDs):

   rag = VertexAIMultimodalRAG(
       bucket_name="your-bucket",
       index_id="saved-index-id",
       endpoint_id="saved-endpoint-id"
   )

   # Process documents
   result = rag.process_and_index_document(
       "path/to/doc.pdf", "project1", "rfp"
   )

   # Query
   results = rag.query("What are the requirements?", top_k=5)

=================================================================
""")


if __name__ == "__main__":
    # Print setup instructions
    print_setup_instructions()

    # Example usage (uncomment to use)
    # rag = VertexAIMultimodalRAG(
    #     bucket_name=BUCKET_NAME,
    #     openai_api_key=os.getenv("OPENAI_API_KEY"),
    #     enable_vision_ai=True
    # )

    # First time: Create index and endpoint
    # print("Creating index (takes ~30 minutes)...")
    # index_id = rag.create_vector_index()
    # print(f"Index ID: {index_id}")

    # print("Creating endpoint (takes ~10 minutes)...")
    # endpoint_id = rag.create_index_endpoint()
    # print(f"Endpoint ID: {endpoint_id}")

    # print("\nSave these IDs for future use!")
