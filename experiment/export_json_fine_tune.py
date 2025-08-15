# %%
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
output_jsonl_file = 'training_data.jsonl'
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

# %%
chunks_data

# %%
# --- 2. Proses Setiap Chunk dan Simpan Hasilnya ---
progress_file = 'last_chunk.txt'
start_index = 0
try:
    with open(progress_file, 'r') as f:
        last_processed_index = int(f.read().strip())
        start_index = last_processed_index + 1
        print(
            f"Proses sebelumnya berhenti di indeks {last_processed_index}. Melanjutkan dari indeks {start_index}.")
except (FileNotFoundError, ValueError):
    print("File progres tidak ditemukan. Memulai dari awal.")
    start_index = 0
if chunks_data:
    # Buka file output dalam mode append ('a')
    with open(output_jsonl_file, 'a', encoding='utf-8') as outfile:
        # Gunakan tqdm untuk menampilkan progress bar
        data_to_process = chunks_data[start_index:]
        progress_bar = tqdm(
            enumerate(data_to_process, start=start_index),
            desc="Memproses Chunks",
            initial=start_index,
            total=len(chunks_data),
            unit="chunk"
        )

        for i, chunk_doc in progress_bar:
            chunk = chunk_doc.page_content.strip()
            nomor = chunk_doc.metadata.get('nomor', '-')
            tahun = chunk_doc.metadata.get('tahun', '-')
            if len(chunk) < 50:
                print(
                    f"Chunk terlalu pendek ({len(chunk)} karakter), melewatkan chunk: '{chunk[:50]}...'")
                continue
            try:
                # Lewati jika chunk kosong atau hanya berisi spasi
                if not chunk or chunk.isspace():
                    print(f"Melewatkan chunk kosong.")
                    continue

                # --- 3. Panggil Graph ---
                final_state = graph.invoke(
                    input={"messages": [HumanMessage(content=chunk)],
                           "nomor": nomor, "tahun": tahun}
                )

                # --- 4. Ekstrak Hasil ---
                generated_pairs = final_state.get('final_response')

                if generated_pairs and generated_pairs.responses:
                    # --- 5. Tulis setiap pasangan ke file .jsonl ---
                    for pair in generated_pairs.responses:
                        # Ubah objek Pydantic menjadi dictionary, lalu ke string JSON
                        line_to_write = json.dumps(
                            pair.model_dump(), ensure_ascii=False)
                        outfile.write(line_to_write + '\n')
                else:
                    print(
                        f"Peringatan: Tidak ada respons yang dihasilkan untuk chunk: '{chunk[:50]}...'")
                # Simpan progres SETELAH berhasil
                with open(progress_file, 'w') as pf:
                    pf.write(str(i))
            except Exception as e:
                # --- 6. Penanganan Error per Chunk ---
                print(
                    f"Terjadi error saat memproses chunk: '{chunk[:50]}...'. Error: {e}. Melanjutkan ke chunk berikutnya.")

    print(f"\nProses selesai.")
    print(
        f"Total {total_dihasilkan} set data training telah disimpan di '{output_jsonl_file}'.")
else:
    print("Tidak ada data untuk diproses.")
