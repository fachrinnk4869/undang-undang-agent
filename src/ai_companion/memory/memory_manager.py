import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import os
import re
from langchain.document_loaders import PyPDFLoader
from pydantic import BaseModel, Field
from ai_companion.core.prompts import CONTEXT_ANALYSIS_PROMPT, MEMORY_ANALYSIS_PROMPT
from ai_companion.graph.utils.helpers import get_chat_model
from ai_companion.memory.mapper import ChromaContextMapper, ChromaMemoryMapper
from ai_companion.memory.memory_template import ContextMemory, Memory
from ai_companion.storage.vector.chroma import ChromaStorage
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage
from ai_companion.settings import settings
from ai_companion.memory.vector_store import get_vector_store
from ai_companion.memory.chunker import get_chunker
import logging
import uuid
from datetime import datetime
from typing import List, Optional


class MemoryAnalysis(BaseModel):
    """Result of analyzing a message for memory-worthy content."""

    is_important: bool = Field(
        ...,
        description="Whether the message is important enough to be stored as a memory",
    )
    formatted_memory: Optional[str] = Field(...,
                                            description="The formatted memory to be stored")


class ContextAnalysis(BaseModel):
    """Result of analyzing a message for memory-worthy content."""

    is_important: bool = Field(
        ...,
        description="Whether the message is important enough to be stored as a memory",
    )
    type: Optional[str] = Field(
        ...,
        description="The type of context to be stored (e.g., 'list', 'direct', 'comparison', 'summary', 'count', 'exists', 'mixed')",
    )
    filters: Optional[dict] = Field(
        ...,
        description="Filters to apply when retrieving this context in the future",
    )


class Manager:
    """Manager class for handling memory operations."""

    def __init__(self, storage, mapper) -> None:
        """Initialize the MemoryManager with a vector store and chunker."""
        self.vector_store = get_vector_store(storage, mapper)
        self.chunker = get_chunker()
        self.logger = logging.getLogger(__name__)

    def get_relevant_memories(self, context: str) -> List[str]:
        """Retrieve relevant memories based on the current context."""
        memories = self.vector_store.search_memories(
            context, k=settings.MEMORY_TOP_K)
        if memories:
            for memory in memories:
                self.logger.debug(
                    f"Memory: '{memory.text}' (score: {memory.score:.2f})")
        return [memory.text for memory in memories]

    def format_memories_for_prompt(self, memories: List[str]) -> str:
        """Format retrieved memories as bullet points."""
        if not memories:
            return ""
        return "\n".join(f"- {memory}" for memory in memories)


class MemoryManager(Manager):
    """Manager class for handling long-term memory operations."""

    def __init__(self, storage, mapper) -> None:
        """Initialize the MemoryManager with a vector store and chunker."""
        super().__init__(storage, mapper)
        self.llm = get_chat_model(
            temperature=0.1).with_structured_output(MemoryAnalysis)

    def find_memory(self, embeddings):
        raw_results = self.vector_store.search(embeddings)
        return self.map_results(raw_results)

    async def _analyze_memory(self, message: str) -> MemoryAnalysis:
        """Analyze a message to determine importance and format if needed."""
        prompt = MEMORY_ANALYSIS_PROMPT.format(message=message)
        return await self.llm.ainvoke(prompt)

    async def extract_and_store_memories(self, message: BaseMessage) -> None:
        """Extract important information from a message and store in vector store."""
        if message.type != "human":
            return

        # Analyze the message for importance and formatting
        analysis = await self._analyze_memory(message.content)
        if analysis.is_important and analysis.formatted_memory:
            # Check if similar memory exists
            similar = self.vector_store.find_similar_memory(
                analysis.formatted_memory)
            if similar:
                # Skip storage if we already have a similar memory
                self.logger.info(
                    f"Similar memory already exists: '{analysis.formatted_memory}'")
                return

            # Store new memory
            self.logger.info(
                f"Storing new memory: '{analysis.formatted_memory}'")
            self.vector_store.store_memory(
                text=analysis.formatted_memory,
                metadata={
                    "id": str(uuid.uuid4()),
                    "timestamp": datetime.now().isoformat(),
                },
            )


class ContextManager(Manager):

    def __init__(self, storage, mapper) -> None:
        """Initialize the ContextManager with a vector store and chunker."""
        super().__init__(storage, mapper)
        self.llm = get_chat_model(
            temperature=0.1)

    async def _analyze_context(self, message: str) -> ContextAnalysis:
        """Analyze a message to determine importance and format if needed."""
        prompt = f"{CONTEXT_ANALYSIS_PROMPT} {message}"
        return await self.llm.ainvoke(prompt)

    async def extract_context(self, message: BaseMessage) -> dict:
        """Extract important information from a message and store in vector store."""
        if message.type != "human":
            return

        # Analyze the message for importance and formatting
        analysis = await self._analyze_context(message.content)
        # biasanya result.content masih string JSON
        raw_json = analysis.content if isinstance(
            analysis, dict) else str(analysis.content)
        print(f"Raw JSON: {raw_json}")
        # bersihin ```json ... ```
        cleaned = raw_json.replace("```json", "").replace("```", "").strip()

        # parse ke Pydantic
        analysis = ContextAnalysis.model_validate(json.loads(cleaned))
        print(f"Analysis result: {analysis}")

        return {
            "type": analysis.type,
            "filters": analysis.filters,
            "is_important": analysis.is_important
        }

    def add_document(self, full_text: str, source_filename: str, metadata: dict) -> List[str]:
        """Chunk and store a document in the vector store."""
        chunks = self.chunker.chunk(full_text)
        if not chunks:
            self.logger.warning(
                f"No valid chunks found in document: {source_filename}")
            return []

        for chunk in tqdm(chunks):
            # print(f"Storing chunk: {chunk[:4]}...")  # Log first 50 chars
            self.vector_store.store_memory(
                text=chunk,
                metadata={
                    **metadata,
                    'text': chunk,
                    "id": str(uuid.uuid4()),
                    "file_name": source_filename,
                    "timestamp": datetime.now().isoformat(),
                },
            )
        self.logger.info(f"Stored {len(chunks)} chunks from {source_filename}")
        return chunks

    def add_documents(self) -> None:
        """Add multiple documents to the vector store."""
        folder_path = Path(
            settings.DATA_UU_PATH)  # Ganti dengan path folder PDF Anda
        pdf_files = list(folder_path.glob("*.pdf"))

        all_semantic_chunks = []
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs with PyPDFLoader"):
            # try:
            # 1. Gunakan PyPDFLoader untuk memuat PDF per halaman
            loader = PyPDFLoader(str(pdf_file))
            pages = loader.load()  # Hasilnya adalah list of Document (satu per halaman)

            # 2. Gabungkan konten teks dari semua halaman menjadi satu string
            full_text = "\n".join([page.page_content for page in pages])

            # 3. Lakukan Semantic Chunking (per pasal) pada teks yang sudah digabung
            match = re.search(
                r'uu(\d+)-(\d{4})', pdf_file.name, re.IGNORECASE)
            nomor, tahun = match.groups() if match else ("-", "-")
            print(
                f"Processing {pdf_file.name} - Nomor: {nomor}, Tahun: {tahun}")
            semantic_chuks = self.add_document(
                full_text=full_text,
                source_filename=pdf_file.name,
                metadata={
                    "nomor": nomor,
                    "tahun": tahun
                }
            )
            all_semantic_chunks.extend(semantic_chuks)
            print(
                f"✅ Berhasil memproses {pdf_file.name} dengan {len(semantic_chuks)} semantic chunks.")
            # except Exception as e:
            #     print(f"Error processing {pdf_file.name}: {e}")
            #     continue
        self.logger.info(
            f"Proses selesai. Ditemukan {len(all_semantic_chunks)} semantic chunks.")


def get_memory_manager() -> Manager:
    """Get a MemoryManager instance for memory."""
    return MemoryManager(ChromaStorage('memory_vectorstore'), ChromaMemoryMapper())


def get_context_manager() -> Manager:
    """Get a MemoryManager for context."""
    return ContextManager(ChromaStorage('uu_vectorstore'), ChromaContextMapper())


if __name__ == "__main__":
    # Example usage
    memory_manager = get_context_manager()
    # Add documents to vector store
    memory_manager.add_documents()
    print("Documents added to vector store.")
