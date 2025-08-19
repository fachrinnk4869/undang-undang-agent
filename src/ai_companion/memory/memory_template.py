from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Memory:
    """Represents a memory entry in the vector store."""

    text: str
    metadata: dict
    score: Optional[float] = None

    @property
    def id(self) -> Optional[str]:
        return self.metadata.get("id")

    @property
    def timestamp(self) -> Optional[datetime]:
        ts = self.metadata.get("timestamp")
        return datetime.fromisoformat(ts) if ts else None


@dataclass
class ContextMemory(Memory):
    """Represents a context entry in the vector store."""

    text: str
    metadata: dict
    score: Optional[float] = None

    @property
    def id(self) -> Optional[str]:
        return self.metadata.get("id")

    @property
    def file_name(self) -> Optional[str]:
        return self.metadata.get("file_name")

    @property
    def nomor(self) -> Optional[str]:
        return self.metadata.get("nomor")

    @property
    def tahun(self) -> Optional[str]:
        return self.metadata.get("tahun")

    @property
    def timestamp(self) -> Optional[datetime]:
        ts = self.metadata.get("timestamp")
        return datetime.fromisoformat(ts) if ts else None

    def __str__(self) -> str:
        """Convert context into a human-readable string for prompt injection."""
        parts = []
        if self.nomor and self.tahun:
            parts.append(f"Pasal nomor {self.nomor} pada {self.tahun}")
        elif self.nomor:
            parts.append(f"Pasal nomor {self.nomor}")
        if self.file_name:
            parts.append(f"dari dokumen '{self.file_name}'")
        if self.timestamp:
            parts.append(f"(timestamp: {self.timestamp.isoformat()})")
        # print("hasil text", self.text)
        parts.append(f"Isi: {self.text.strip()}")
        return " | ".join(parts)
