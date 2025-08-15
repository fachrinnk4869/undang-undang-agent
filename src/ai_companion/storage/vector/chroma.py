from ai_companion import settings
import chromadb

import os

from ai_companion.memory.memory_template import Memory


class ChromaStorage():
    """Embedder class for Pinecone using SentenceTransformer."""
    REQUIRED_ENV_VARS = []

    def __init__(self, vectorstore_path: str):
        self._validate_env_vars()
        self.VECTORSTORE_PATH = vectorstore_path
        # Persistent storage biar data nggak hilang
        self.client = chromadb.PersistentClient(path=self.VECTORSTORE_PATH)

        # Bikin collection (kalau sudah ada, dia auto get)
        self.collection = self.client.get_or_create_collection(
            name="my_collection"
        )

    def _validate_env_vars(self) -> None:
        """Validate that all required environment variables are set."""
        missing_vars = [
            var for var in self.REQUIRED_ENV_VARS if not os.getenv(var)]
        if missing_vars:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_vars)}")

    def search(self, query_embedding: str, top_k: int = 10):
        """Search for similar documents in Chroma DB."""
        return self.collection.query(
            query_embedding, n_results=top_k)

    def save(self, vectors: dict) -> None:
        """Save a text and its metadata to Chroma DB."""
        self.collection.add(
            embeddings=[vectors['values']],
            metadatas=[vectors.get('metadata', {})],
            ids=[vectors['id']]
        )
