from ai_companion.settings import settings
import requests


class Embedder:
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-8B", dim_size: int = 2048):
        self.model_name = model_name
        self.dim_size = dim_size
        self.headers = {
            "Authorization": f"Bearer {settings.SILICONFLOW_API_KEY}",
            "Content-Type": "application/json"
        }

    def encode(self, text: str) -> list[float]:
        payload = {
            "model": self.model_name,
            "input": text,
            "encoding_format": "float",
            "dimensions": self.dim_size
        }

        response = requests.post(
            settings.SILICONFLOW_URL_EMBEDDING, json=payload, headers=self.headers)

        return response.json()['data'][0]['embedding']

    def decode(self, embedding: list[float]) -> str:
        """Convert an embedding back to text (if applicable)."""
        # Placeholder for actual decoding logic
        return "Decoded text from embedding"
