from langchain.document_loaders import PyPDFLoader
import re
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
from langchain.schema import Document

# ==============================================================================
# FUNGSI SEMANTIC CHUNKING (TETAP SAMA)
# ==============================================================================


def chunk_text_by_pasal(full_text, source_filename):
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
    # print(f"Total chunks found in {source_filename}: {len(chunks)}")
    # Print first 100 chars of first chunk
    # print(f"Example chunk: {chunks[1]}...")
    # Menangani kasus jika dokumen tidak punya 'Pasal'
    if not chunks and len(full_text) > 100:
        return [full_text.strip()]

    return chunks

# ==============================================================================
# PROSES UTAMA (MENGGUNAKAN PyPDFLoader)
# ==============================================================================


output_filename = "chunks-with-metadata.pkl"

if not os.path.exists(output_filename):
    print("File chunks tidak ditemukan. Memulai proses loading dan chunking PDF...")

    folder_path = Path("assets/scrap-fix")  # Ganti dengan path folder PDF Anda
    pdf_files = list(folder_path.glob("*.pdf"))

    all_semantic_chunks = []
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs with PyPDFLoader"):
        try:
            # 1. Gunakan PyPDFLoader untuk memuat PDF per halaman
            loader = PyPDFLoader(str(pdf_file))
            pages = loader.load()  # Hasilnya adalah list of Document (satu per halaman)

            # 2. Gabungkan konten teks dari semua halaman menjadi satu string
            full_text = "\n".join([page.page_content for page in pages])

            # 3. Lakukan Semantic Chunking (per pasal) pada teks yang sudah digabung
            file_chunks = chunk_text_by_pasal(full_text, pdf_file.name)
            match = re.search(r'uu(\d+)-(\d{4})', pdf_file.name, re.IGNORECASE)
            nomor, tahun = match.groups() if match else ("-", "-")
            chunked_documents = []
            for chunk_text in file_chunks:
                doc = Document(
                    page_content=chunk_text,
                    metadata={
                        "nomor": nomor,
                        "tahun": tahun,
                        "file_name": pdf_file.name
                    }
                )
                chunked_documents.append(doc)
            # 4. Tambahkan hasilnya ke list utama
            all_semantic_chunks.extend(chunked_documents)

        except Exception as e:
            print(f"Error processing {pdf_file.name}: {e}")
            continue

    print(
        f"Proses selesai. Ditemukan {len(all_semantic_chunks)} semantic chunks.")

    # 5. Simpan hasil chunking
    np.save(output_filename, all_semantic_chunks)

    chunks = all_semantic_chunks
else:
    print(f"File '{output_filename}' ditemukan. Memuat chunks dari file...")
    chunks = np.load(output_filename, allow_pickle=True).tolist()

print(f"Total chunks siap untuk training: {len(chunks)}")
print("\nContoh chunk pertama:")
print("="*30)
if chunks:
    print(chunks[2] + "...")
print("="*30)

# ... sisa skrip Anda untuk membuat Dataset Hugging Face dan training bisa dilanjutkan dari sini ...
