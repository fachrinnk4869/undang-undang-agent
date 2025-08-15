from ai_companion import settings
from ai_companion.memory.memory_template import Memory
from pinecone import Pinecone, ServerlessSpec
import os


class PineconeStorage():
    """Embedder class for Pinecone using SentenceTransformer."""
    REQUIRED_ENV_VARS = []

    def __init__(self):
        self._validate_env_vars()
        self.client = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.dense_index = self.client.Index(
            host=settings.HOST_PINECONE_DENSE)

    def _validate_env_vars(self) -> None:
        """Validate that all required environment variables are set."""
        missing_vars = [
            var for var in self.REQUIRED_ENV_VARS if not os.getenv(var)]
        if missing_vars:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_vars)}")

    def search(self, query_embedding: str, top_k: int = 10):
        """Search for similar documents in Pinecone."""
        result = self.dense_index.query(
            namespace=settings.NAMESPACE,
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            include_values=False,
            # filter=filter_query if filter_query else None
        )
        return [
            Memory(
                text=hit.metadata.values,
                metadata=hit.metadata,
                score=hit.score,
            )
            for hit in result.matches
        ]

    def save(self, vectors: dict) -> None:
        """Save a text and its metadata to Pinecone."""
        self.dense_index.upsert(
            namespace=settings.NAMESPACE,
            vectors=vectors,
        )
