from functools import lru_cache
from typing import List
import re


class TextChunker:
    def __init__(self, chunk_size: int = 200):
        self.chunk_size = chunk_size

    def chunk(self, full_text: str) -> List[str]:
        """Memotong teks berdasarkan 'Pasal' dan mengembalikan list of chunks."""

        # Pola Regex untuk memisahkan teks setiap kali menemukan 'Pasal X' di awal baris.
        pattern = r"(?=Pasal \d+)"

        raw_chunks = re.split(pattern, full_text)

        chunks = []

        for i, raw_chunk in enumerate(raw_chunks):
            cleaned_chunk = raw_chunk.strip()
            if not cleaned_chunk or len(cleaned_chunk) < 20:
                continue

            # Kita hanya butuh konten teksnya untuk Causal LM Fine-tuning
            chunks.append(cleaned_chunk)

        # Menangani kasus jika dokumen tidak punya 'Pasal'
        if not chunks and len(full_text) > 100:
            return [full_text.strip()]
        return chunks


@lru_cache
def get_chunker() -> TextChunker:
    """Fungsi untuk mendapatkan instance TextChunker."""
    return TextChunker(chunk_size=200)
