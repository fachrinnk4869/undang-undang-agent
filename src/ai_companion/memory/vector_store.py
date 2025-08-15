import os
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import List, Optional
import uuid

from ai_companion.settings import settings
from ai_companion.storage.vector import PineconeStorage, ChromaStorage
from sentence_transformers import SentenceTransformer
from ai_companion.memory.embed import Embedder
from pinecone import Pinecone, ServerlessSpec
from ai_companion.memory.memory_template import Memory


class VectorStore:
    """A class to handle vector storage operations using Qdrant."""

    REQUIRED_ENV_VARS = []
    # Qwen/Qwen3-Embedding-8B / all-MiniLM-L6-v2
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    INDEX_NAME = "long_term_memory"
    SIMILARITY_THRESHOLD = 0.9  # Threshold for considering memories as similar
    EMBED_DIM = 2048  # Dimension of the embedding model
    TOP_K = 20

    _instance: Optional["VectorStore"] = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs) -> "VectorStore":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, storage, mapper) -> None:
        if not self._initialized:
            self._validate_env_vars()
            # self.model = Embedder(self.EMBEDDING_MODEL)
            self.model = SentenceTransformer(self.EMBEDDING_MODEL)
            self.storage = storage  # or ChromaStorage()
            self.mapper = mapper
            self._initialized = True

    def _validate_env_vars(self) -> None:
        """Validate that all required environment variables are set."""
        missing_vars = [
            var for var in self.REQUIRED_ENV_VARS if not os.getenv(var)]
        if missing_vars:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_vars)}")

    # def _index_exists(self) -> bool:
    #     """Check if the memory index exists."""
    #     return self.client.has_index(self.INDEX_NAME)

    # def _create_index(self) -> None:
    #     """Create a new collection for storing memories."""
    #     self.client.create_index(
    #         name=self.INDEX_NAME,
    #         vector_type="dense",
    #         dimension=self.EMBED_DIM,
    #         metric="cosine",
    #         spec=ServerlessSpec(
    #             cloud="aws",
    #             region="us-east-1"
    #         ),
    #         deletion_protection="disabled"
    #     )

    def find_similar_memory(self, text: str) -> Optional[Memory]:
        """Find if a similar memory already exists.

        Args:
            text: The text to search for

        Returns:
            Optional Memory if a similar one is found
        """
        results = self.search_memories(text, k=1)
        if results and results[0].score >= self.SIMILARITY_THRESHOLD:
            return results[0]
        return None

    def generate_embedding(self, text: str):
        try:
            # get dense embedding
            dense_item = {
                "values": self.model.encode(text)
            }
            return dense_item
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def store_memory(self, text: str, metadata: dict) -> None:
        """Store a new memory in the vector store or update if similar exists.

        Args:
            text: The text content of the memory
            metadata: Additional information about the memory (timestamp, type, etc.)
        """
        # if not self._index_exists():
        #     self._create_index()

        # Check if similar memory exists
        # print('Finding similar memory...')
        similar_memory = self.find_similar_memory(text)
        if similar_memory and similar_memory.id:
            print(
                f"Similar memory found (ID: {similar_memory.id}, Skip storing.")
            return
            metadata["id"] = similar_memory.id  # Keep same ID for update

        embedding = self.model.encode(text)
        self.storage.save({
            "id": metadata.get("id") or str(uuid.uuid4()),
            "values": embedding,
            "metadata": metadata
        })

    def search_memories(self, query: str, k: int = 5) -> List[Memory]:
        """Search for similar memories in the vector store.

        Args:
            query: Text to search for
            k: Number of results to return

        Returns:
            List of Memory objects
        """
        query_embedding = self.model.encode(query).tolist()
        raw_results = self.storage.search(query_embedding, top_k=k)
        return self.mapper.map_results(raw_results)


@lru_cache
def get_vector_store(storage, mapper) -> VectorStore:
    """Get or create the VectorStore singleton instance."""
    return VectorStore(storage, mapper)
