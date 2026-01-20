"""
This module provides the VertexAIMultimodalRAG class for interacting with the
GCP Vertex AI-based RAG system.
"""
import os
import json
import logging
from typing import Dict, List, Any, Optional

from google.cloud import aiplatform
from google.cloud.aiplatform.gapic.schema import predict
from google.cloud.aiplatform_v1.services import llm_utility_service
from google.cloud.aiplatform_v1.types import llm_utility as llm_utility_types
from google.cloud import storage
import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel, Part


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VertexAIMultimodalRAG:
    """
    A class to handle RAG operations using Google Cloud Vertex AI, including
    multimodal queries against a vector index and answer generation with Gemini.
    """
    def __init__(
        self,
        bucket_name: str,
        index_id: str,
        endpoint_id: str,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        enable_vision_ai: bool = True,
    ):
        """
        Initializes the RAG system.

        Args:
            bucket_name: Name of the GCS bucket for metadata.
            index_id: The ID of the Vertex AI Vector Search index.
            endpoint_id: The ID of the index endpoint.
            project_id: The GCP project ID.
            location: The GCP region.
            enable_vision_ai: Flag to enable Gemini Vision for image analysis.
        """
        if not project_id or not location:
            # Autodiscover from environment
            from google.auth import default
            _, project_id = default()
            location = os.getenv("GCP_REGION", "northamerica-northeast1")
            
        self.project_id = project_id
        self.location = location
        self.bucket_name = bucket_name
        self.index_id = index_id
        self.endpoint_id = endpoint_id
        self.enable_vision_ai = enable_vision_ai

        vertexai.init(project=self.project_id, location=self.location)

        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(self.bucket_name)

        self.index_endpoint = aiplatform.MatchingEngineIndexEndpoint(
            index_endpoint_name=self.endpoint_id
        )
        self.embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
        self.vision_model = GenerativeModel("gemini-1.5-flash-001") if self.enable_vision_ai else None
        
        logger.info("VertexAIMultimodalRAG initialized successfully.")
        logger.info(f"Project: {self.project_id}, Location: {self.location}")

    def query(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Performs a semantic search query against the Vertex AI Vector Search index.
        """
        logger.info(f"Querying with top_k={top_k} and filters={filters}")
        query_embedding = self._get_embedding(query)

        filter_restricts = []
        if filters:
            for key, value in filters.items():
                filter_restricts.append(
                    {"namespace": key, "allow_list": [value]}
                )

        response = self.index_endpoint.find_neighbors(
            queries=[query_embedding],
            num_neighbors=top_k,
            filter=filter_restricts,
        )

        results = []
        if not response or not response[0]:
            return results

        for match in response[0]:
            try:
                metadata = self._get_metadata_from_gcs(match.id)
                if metadata:
                    results.append(
                        {
                            "id": match.id,
                            "score": match.distance,
                            **metadata,
                        }
                    )
            except Exception as e:
                logger.error(f"Error processing match {match.id}: {e}")

        # Sort by score (descending) as find_neighbors gives distance
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def generate_answer(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Generates an answer to a query using RAG from the indexed documents.
        """
        logger.info("Generating answer...")
        search_results = self.query(query, top_k=top_k, filters=filters)

        if not search_results:
            return {"answer": "Could not find any relevant information.", "sources": []}

        context = ""
        for i, res in enumerate(search_results, 1):
            context += f"Source [{i}]:\n"
            context += f"Document: {res.get('document_type', 'N/A')}, Project: {res.get('project_id', 'N/A')}\n"
            context += f"Content: {res.get('content', '')}\n\n"

        prompt = f"""
        You are a helpful AI assistant. Answer the user's question based on the
        provided sources. Cite the sources you use in your answer using the [1], [2], etc. notation.

        Question: {query}

        Sources:
        {context}

        Answer:
        """

        response = self.vision_model.generate_content(prompt)
        
        return {
            "answer": response.text,
            "sources": search_results,
        }

    def _get_embedding(self, text: str) -> List[float]:
        """Generates an embedding for the given text."""
        return self.embedding_model.get_embeddings([text])[0].values

    def _get_metadata_from_gcs(self, vector_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves metadata for a given vector ID from GCS."""
        try:
            blob_name = f"embeddings/{vector_id}.json"
            blob = self.bucket.blob(blob_name)
            if blob.exists():
                metadata_str = blob.download_as_text()
                return json.loads(metadata_str)
        except Exception as e:
            logger.error(f"Failed to retrieve metadata for {vector_id}: {e}")
        return None

if __name__ == "__main__":
    # Example usage when run directly
    from dotenv import load_dotenv
    load_dotenv()

    BUCKET_NAME = os.getenv("BUCKET_NAME")
    INDEX_ID = os.getenv("INDEX_ID")
    ENDPOINT_ID = os.getenv("ENDPOINT_ID")

    if not all([BUCKET_NAME, INDEX_ID, ENDPOINT_ID]):
        print("Error: BUCKET_NAME, INDEX_ID, and ENDPOINT_ID must be set in .env file.")
    else:
        print("Initializing RAG system for direct script execution example...")
        rag = VertexAIMultimodalRAG(
            bucket_name=BUCKET_NAME,
            index_id=INDEX_ID,
            endpoint_id=ENDPOINT_ID,
        )

        # 1. Basic Query
        print("\n--- Example 1: Basic Query ---")
        results = rag.query("What are the public engagement requirements?", top_k=3)
        print(f"Found {len(results)} results.")
        for res in results:
            print(f"  - Score: {res['score']:.3f}, Project: {res.get('project_id', 'N/A')}, Type: {res.get('document_type')}")
        
        # 2. Filtered Query
        print("\n--- Example 2: Filtered Query (RFPs only) ---")
        results = rag.query("budget", top_k=2, filters={"document_type": "rfp"})
        print(f"Found {len(results)} RFP results.")
        for res in results:
            print(f"  - Score: {res['score']:.3f}, Project: {res.get('project_id', 'N/A')}")
            
        # 3. Answer Generation
        print("\n--- Example 3: Answer Generation ---")
        answer_obj = rag.generate_answer("What are the common engagement approaches?")
        print("Generated Answer:")
        print(answer_obj['answer'])