# %%
import concurrent.futures
from typing import Optional
import numpy as np
from tqdm import tqdm  # Untuk progress bar, install dengan: pip install tqdm
import pickle
import json
import json  # Pastikan json diimpor
from typing import List
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
load_dotenv()
# %%
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
)

# %%
prompt = PromptTemplate.from_template("""
Buatkan 5 set instruction, context, dan output dalam format JSON sebagai array untuk nanti di tranining boleh apapun khususnya pertanyaan yang berkaitan dengan undang undang, dengan struktur berikut:

[
  {{
    "instruction": "<instruction>",
    "context": "<context>",
    "output": "<output>"
  }},
  ...
]


Data ini adalah isi undang-undang dari peraturan Nomor {nomor} Tahun {tahun}. Teks dasar yang digunakan adalah:

"{text}"

Pastikan JSON valid dan hasilnya berjumlah 5. Perhatikan jangan gunakan pertanyaan undang undang kata 'ini' sebagai pertanyaan tapi gunakan nomor dan tahun yang sudah disediakan.
""")


# %%


class ResponseDetail(BaseModel):
    """Mendefinisikan struktur untuk satu item dalam daftar respons."""
    instruction: str = Field(description="Instruksi untuk diikuti oleh model.")
    context: str = Field(
        description="Konteks yang digunakan model untuk responsnya.")
    output: str = Field(
        description="Output atau jawaban yang dihasilkan oleh model.")


class ResponseList(BaseModel):
    """Mendefinisikan struktur untuk daftar (array) dari respons."""
    responses: List[ResponseDetail]

# %%
# State ini akan menyimpan histori pesan dan respons akhir yang terstruktur


class AgentState(MessagesState):
    final_response: ResponseList = None
    nomor: Optional[str] = None
    tahun: Optional[str] = None


# %%
parser = PydanticOutputParser(pydantic_object=ResponseList)

# %%
# Fungsi ini mengambil input pengguna, memformat prompt, dan memanggil LLM.


def call_model(state: AgentState):
    """Memanggil LLM untuk menghasilkan data JSON berdasarkan input."""
    # print("---NODE: Memanggil LLM---")
    # Mengambil pesan terakhir (input dari pengguna)
    user_input = state['messages'][-1].content
    nomor = state.get('nomor')
    tahun = state.get('tahun')
    # Memformat prompt dengan input pengguna dan instruksi format dari parser
    formatted_prompt = prompt.format(
        text=user_input,
        nomor=nomor if nomor else "-",
        tahun=tahun if tahun else "-",
        parser_instructions=parser.get_format_instructions()
    )
    # Memanggil LLM dengan prompt yang sudah diformat
    response = llm.invoke(formatted_prompt)
    # print("Respons mentah dari LLM diterima.")
    # Menambahkan respons dari LLM ke dalam histori pesan
    return {"messages": [response]}


# %%
model_with_structured_output = llm.with_structured_output(ResponseDetail)

# %%

# Node 2: Mem-parsing output dan menyelesaikan alur (Versi Diperbaiki)
# Fungsi ini mengambil output string dari LLM dan mengubahnya menjadi struktur data Python.


def parse_response(state: AgentState):
    """Mem-parsing respons dari LLM dan menyimpannya ke state akhir."""
    # print("---NODE: Mem-parsing Respons---")
    # Mengambil konten dari pesan terakhir (respons dari LLM)
    response_content = state['messages'][-1].content.strip()

    # Menghapus ```json, ```, dan markdown lain jika ada
    if response_content.startswith("```json"):
        response_content = response_content[7:]
    if response_content.endswith("```"):
        response_content = response_content[:-3]
    response_content = response_content.strip()

    try:
        # PERBAIKAN: Periksa apakah outputnya adalah array mentah.
        # Jika ya, bungkus secara manual agar sesuai dengan skema ResponseList.
        if response_content.startswith("[") and response_content.endswith("]"):
            # print("Output terdeteksi sebagai array mentah. Membungkusnya dalam objek...")
            # Membuat string JSON yang valid sesuai yang diharapkan parser
            response_content = f'{{"responses": {response_content}}}'
            # print(response_content)

        # Menggunakan parser untuk mengubah string JSON menjadi objek Python
        structured_response = parser.parse(response_content)
        # print("Parsing berhasil.")
        # Menyimpan hasil parsing ke dalam state akhir
        return {"final_response": structured_response}
    except Exception as e:
        print(f"Error saat parsing: {e}")
        # Jika terjadi error, kita bisa menambahkan logika penanganan error di sini
        return {}


# %%
workflow = StateGraph(AgentState)

# Menambahkan node-node ke dalam graph
workflow.add_node("llm", call_model)
workflow.add_node("parser", parse_response)

# Menentukan node awal (entry point)
workflow.set_entry_point("llm")

# Menambahkan alur (edge) antar node
workflow.add_edge("llm", "parser")
workflow.add_edge("parser", END)

# Meng-compile graph menjadi aplikasi yang bisa dijalankan
graph = workflow.compile()

# %%
try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception as e:
    print(f"Tidak dapat menampilkan visualisasi graph: {e}")
    print("Pastikan library `pygraphviz` atau `pydot` terinstal jika diperlukan.")


# %%
# Teks input yang akan digunakan
input_text = """Pasal 1.
Dengan menyimpang seperlunya dari Peraturan Presiden Republik Indonesia tertanggal
10 Oktober 1945 No. 2, menetapkan, bahwa peraturan-peraturan hukum pidana yang
sekarang berlaku, ialah peraturan-peraturan hukum pidana yang ada pada tanggal 8
Maret 1942."""

# Menjalankan graph dengan input
# Inputnya adalah dictionary yang sesuai dengan struktur AgentState
final_state = graph.invoke(
    input={
        "messages": [HumanMessage(content=input_text)],
        "nomor": "1",
        "tahun": "1945"
    }
)

# %%
# Mengambil dan menampilkan jawaban akhir
answer = final_state.get('final_response')

# %%
# Menampilkan hasil akhir
if answer:
    # Mengubah objek Pydantic menjadi dictionary untuk dicetak sebagai JSON
    print(json.dumps(answer.dict(), indent=2, ensure_ascii=False))
else:
    print("Tidak mendapatkan jawaban akhir.")

# %%
final_state

# %%

# %%
# --- Konfigurasi ---
input_pickle_file = 'chunks-with-metadata.npy'
# %%
# --- 1. Muat Data Chunks ---
try:
    chunks_data = np.load(input_pickle_file, allow_pickle=True).tolist()
    print(
        f"Berhasil memuat {len(chunks_data)} chunks dari {input_pickle_file}")
except FileNotFoundError:
    print(
        f"Error: File '{input_pickle_file}' tidak ditemukan. Pastikan file ada di direktori yang sama.")
    chunks_data = []  # Beri list kosong agar skrip tidak error
# ==============================================================================
# FUNGSI PEKERJA (WORKER FUNCTION) - VERSI MODIFIKASI
# ==============================================================================


def process_chunk_doc(task_data):
    """
    Fungsi ini memproses SATU chunk_doc, mengekstrak metadata, dan memanggil graph.
    Dirancang untuk dijalankan secara paralel di thread terpisah.
    """
    index, chunk_doc = task_data
    try:
        # 1. Ekstrak data dari objek chunk_doc
        chunk = chunk_doc.page_content.strip()
        # Default ke '-' jika tidak ada
        nomor = chunk_doc.metadata.get('nomor', '-')
        # Default ke '-' jika tidak ada
        tahun = chunk_doc.metadata.get('tahun', '-')

        # 2. Logika validasi baru: lewati jika chunk terlalu pendek
        if len(chunk) < 50:
            return {"index": index, "status": "skipped", "data": f"Chunk terlalu pendek ({len(chunk)} karakter)"}

        # 3. Panggil Graph dengan menyertakan metadata
        final_state = graph.invoke(
            {
                "messages": [HumanMessage(content=chunk)],
                "nomor": str(nomor),  # Pastikan tipe data string
                "tahun": str(tahun)  # Pastikan tipe data string
            }
        )

        generated_pairs = final_state.get('final_response')

        if generated_pairs and generated_pairs.responses:
            result_data = [pair.model_dump()
                           for pair in generated_pairs.responses]
            return {"index": index, "status": "success", "data": result_data}
        else:
            return {"index": index, "status": "failed", "data": "Tidak ada respons valid dari LLM."}

    except Exception as e:
        return {"index": index, "status": "error", "data": str(e)}
# ==============================================================================
# SKRIP UTAMA: BATCH PROCESSING PARALEL DENGAN METADATA & RESUME
# ==============================================================================


# --- Konfigurasi ---
output_jsonl_file = 'training_data_with_metadata.jsonl'
progress_file = 'progress_metadata.txt'
MAX_WORKERS = 1  # Jumlah 'pekerja' paralel. Sesuaikan sesuai kebutuhan.

print(f"\n--- MEMULAI PROSES BATCH PARALEL DENGAN METADATA---")
print(f"Jumlah thread/pekerja: {MAX_WORKERS}")
# --- Logika Resume: Baca semua indeks yang sudah selesai ---
processed_indices = set()
try:
    with open(progress_file, 'r') as f:
        processed_indices = {int(line.strip()) for line in f}
    print(
        f"Ditemukan {len(processed_indices)} chunk yang sudah selesai. Akan melewatinya.")
except (FileNotFoundError, ValueError):
    print("File progres tidak ditemukan. Memulai dari awal.")

# Siapkan tugas hanya untuk data yang BELUM diproses
tasks_to_do = [(i, doc) for i, doc in enumerate(
    chunks_data) if i not in processed_indices]

if not tasks_to_do:
    print("\nTidak ada data baru untuk diproses. Semua sudah selesai.")
else:
    print(f"Akan memproses {len(tasks_to_do)} dokumen baru.")
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        with open(output_jsonl_file, 'a', encoding='utf-8') as outfile, \
                open(progress_file, 'a', encoding='utf-8') as p_file:

            # Kirim semua tugas ke thread pool
            future_to_task = {executor.submit(
                process_chunk_doc, task): task for task in tasks_to_do}

            progress_bar = tqdm(
                concurrent.futures.as_completed(future_to_task),
                total=len(tasks_to_do),
                desc="Memproses Dokumen Paralel"
            )

            for future in progress_bar:
                task_data = future_to_task[future]
                original_index = task_data[0]

                try:
                    result = future.result()

                    if result['status'] == 'success':
                        for item in result['data']:
                            outfile.write(json.dumps(
                                item, ensure_ascii=False) + '\n')
                        # Tandai indeks ini sebagai selesai
                        p_file.write(f"{original_index}\n")
                        p_file.flush()
                    elif result['status'] != 'skipped':
                        progress_bar.write(
                            f"Gagal memproses dokumen indeks {original_index}: {result['data']}")

                except Exception as exc:
                    progress_bar.write(
                        f"Dokumen indeks {original_index} menghasilkan error tak terduga: {exc}")

    print("\nProses paralel selesai.")
