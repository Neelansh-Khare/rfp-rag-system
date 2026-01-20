# -*- coding: utf-8 -*-
#
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may
# may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This script demonstrates how to use the Vertex AI RAG API for multimodal search.
"""

import argparse
import os

from google.cloud import aiplatform
from google.cloud.aiplatform.private_preview import rag


def main(project_id: str, location: str, rag_corpus_id: str, question: str):
    """Uses the Vertex AI RAG API to answer a question.

    Args:
        project_id: The Google Cloud project ID.
        location: The Google Cloud region.
        rag_corpus_id: The ID of the RAG corpus to search.
        question: The question to ask.
    """
    aiplatform.init(project=project_id, location=location)

    # Create a RAG client
    rag_client = rag.RagClient()

    # Get the RAG corpus
    rag_corpus = rag_client.get_rag_corpus(rag_corpus_id)

    # Ask the question
    response = rag.rag_retrieval(rag_corpus=rag_corpus, query=question)

    print(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demonstrates how to use the Vertex AI RAG API."
    )
    parser.add_argument("project_id", help="The Google Cloud project ID.")
    parser.add_argument("location", help="The Google Cloud region.")
    parser.add_argument("rag_corpus_id", help="The ID of the RAG corpus to search.")
    parser.add_argument("question", help="The question to ask.")
    args = parser.parse_args()
    main(args.project_id, args.location, args.rag_corpus_id, args.question)
