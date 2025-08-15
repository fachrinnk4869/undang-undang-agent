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
    def pasal(self) -> Optional[str]:
        return self.metadata.get("pasal")

    @property
    def tahun(self) -> Optional[str]:
        return self.metadata.get("tahun")

    @property
    def timestamp(self) -> Optional[datetime]:
        ts = self.metadata.get("timestamp")
        return datetime.fromisoformat(ts) if ts else None
