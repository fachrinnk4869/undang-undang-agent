import json
from functools import partial
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextIteratorStreamer, BitsAndBytesConfig
from langchain.schema import Document
import torch
from peft import PeftModel
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFacePipeline
from tqdm import tqdm
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
import uuid
import re
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
# ================== SETUP ====================

# -----> chunks loader


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


# -----> Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="LazarusNLP/all-indo-e5-small-v4")

# -----> Load or create vectorstore with metadata
vectorstore_path = "uu_vectorstore_new"
if not os.path.exists(vectorstore_path):
    try:
        vectorstore = Chroma(persist_directory=vectorstore_path,
                             embedding_function=embeddings)
    except:
        vectorstore = Chroma.from_documents(
            [], embeddings, persist_directory=vectorstore_path)
    # Example: Load all PDFs in folder and add metadata
    folder_path = Path("assets/scrap-fix")
    pdf_files = list(folder_path.glob("*.pdf"))

    for pdf_file in tqdm(pdf_files, desc="Loading PDFs"):
        try:
            loader = PyPDFLoader(str(pdf_file))
            match = re.search(r'uu(\d+)-(\d{4})', pdf_file.stem)
            nomor, tahun = match.groups() if match else ("-", "-")
            pages = loader.load()  # Hasilnya adalah list of Document (satu per halaman)

            # 2. Gabungkan konten teks dari semua halaman menjadi satu string
            full_text = "\n".join([page.page_content for page in pages])

            # 3. Lakukan Semantic Chunking (per pasal) pada teks yang sudah digabung
            file_chunks = chunk_text_by_pasal(full_text, pdf_file.name)
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

            vectorstore.add_documents(chunked_documents)
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")
            continue
else:
    print("Vectorstore already exists, loading...")
    vectorstore = Chroma(
        persist_directory=vectorstore_path,
        embedding_function=embeddings
    )
# vectorstore.persist()

# retriever = vectorstore.as_retriever()

# -----> Load Hugging Face LLM
base_model = "google/gemma-3-4b-it"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # atau "fp4"
    bnb_4bit_compute_dtype=torch.bfloat16,  # Sangat disarankan untuk Gemma
    bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    trust_remote_code=False,
    device_map="auto",
    quantization_config=quantization_config,
    # load_in_4bit=True,
    torch_dtype=torch.bfloat16,
)
# # Load adapter if available
# adapter_dir = "./phi3-pdf-indo-lora/checkpoint-49830"
# model = PeftModel.from_pretrained(model, adapter_dir)
# model = model.merge_and_unload()

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    max_new_tokens=512,
    # top_k=50,
    # temperature=1,
)
llm = HuggingFacePipeline(pipeline=pipe)

streamer = TextIteratorStreamer(
    tokenizer, skip_prompt=True, skip_special_tokens=True)


def generate(pipe, prompt):
    # hf = HuggingFacePipeline(pipeline=pipe)
    for chunk in llm.stream(prompt):
        yield chunk


# -----> Load Google Gemini LLM
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash",
# )
model_style = "gemini" if isinstance(llm, ChatGoogleGenerativeAI) else "hf"

# ================== GRAPH ====================


class RAGState(dict):
    input: str
    nomor: str = None
    tahun: str = None
    docs: list = None
    output: str = None


graph = StateGraph(RAGState)


# -----> Node 1: Extract nomor & tahun
# def extract_nomor_tahun(state):
#     question = state["input"]
#     prompt = (
#         f"Extract nomor dan tahun undang undang yang secara jelas ada di pertanyaan, jika tidak yakin maka kembalikan saja None:\n"
#         f"Format jawaban: nomor=[nomor], tahun=[tahun]"
#         f"Kalau tidak ada nomor atau tahun, gunakan None.\n"
#         f"Contoh:\n"
#         f"Pertanyaan: Apa isi Undang-Undang Nomor 33 Tahun 2014?\n"
#         f"Jawaban: nomor=33, tahun=2014\n"
#         f"Pertanyaan: Apa isi Undang-Undang tentang pajak?\n"
#         f"Jawaban: None\n"
#         f"Pertanyaan: Messi menang pada tahun berapa?\n"
#         f"Jawaban: None\n"
#         f"Pertanyaan: Undang-Undang apa saja yang membahas andministrasi publik?\n"
#         f"Jawaban: None\n"
#         f"Pertanyaan: {question}\n"
#         f"Jawaban:"
#     )
#     if model_style == "gemini":
#         response = llm.invoke(prompt).content
#     else:
#         response = llm.invoke(prompt)
#     match = re.search(r"nomor\s*=\s*(\d+).+tahun\s*=\s*(\d+)",
#                       response, re.IGNORECASE | re.DOTALL)
#     if match:
#         nomor, tahun = match.groups()
#     else:
#         nomor, tahun = None, None
#     return {"nomor": nomor, "tahun": tahun}


# def extract_nomor_tahun(state):
#     question = state["input"]
#     # Prompt diubah untuk meminta output JSON yang ketat
#     prompt = (
#         f"Analisis pertanyaan berikut. Ekstrak nomor dan tahun undang-undang HANYA JIKA disebutkan secara eksplisit. JANGAN gunakan pengetahuan eksternal.\n"
#         f"Format jawaban HARUS dalam format JSON yang valid.\n"
#         f"Gunakan 'null' untuk nilai yang tidak ditemukan.\n\n"
#         f"Contoh:\n"
#         f"Pertanyaan: Apa isi Undang-Undang Nomor 33 Tahun 2014?\n"
#         f"Jawaban: {{\"nomor\": 33, \"tahun\": 2014}}\n\n"
#         f"Pertanyaan: Undang-Undang apa saja yang membahas sengketa pajak?\n"
#         f"Jawaban: {{\"nomor\": null, \"tahun\": null}}\n\n"
#         f"Pertanyaan: {question}\n"
#         f"Jawaban:"
#     )

#     if model_style == "gemini":
#         response_text = llm.invoke(prompt).content
#     else:
#         response_text = llm.invoke(prompt)

#     try:
#         # Membersihkan output LLM dari markdown code block
#         clean_response = re.sub(r"```json\n?|\n?```",
#                                 "", response_text).strip()
#         data = json.loads(clean_response)
#         nomor = data.get("nomor")
#         tahun = data.get("tahun")
#     except (json.JSONDecodeError, AttributeError):
#         # Fallback jika LLM gagal menghasilkan JSON yang valid
#         nomor, tahun = None, None

#     return {"nomor": nomor, "tahun": tahun}
def extract_nomor_tahun(state):
    """
    Mengekstrak nomor dan tahun UU secara langsung dari pertanyaan menggunakan regex.
    Ini jauh lebih andal dan efisien daripada menggunakan LLM untuk tugas ini.
    """
    question = state["input"]

    # Pola regex untuk mencari "Nomor X Tahun Y" dalam berbagai variasinya
    # - (?:Undang-Undang|UU) -> Mencari "Undang-Undang" atau "UU"
    # - \s+ -> Spasi
    # - (?:Nomor|No\.?) -> Mencari "Nomor", "No.", atau "No"
    # - (\d+) -> Menangkap angka (ini grup 1: nomor)
    # - (?:Tahun|Thn\.?) -> Mencari "Tahun", "Thn.", atau "Thn"
    # - (\d{4}) -> Menangkap 4 digit angka (ini grup 2: tahun)
    # re.IGNORECASE -> Mengabaikan besar kecilnya huruf

    pattern = r"(?:Undang-Undang|UU)\s+(?:Nomor|No|Nomer\.?)\s+(\d+)\s+(?:Tahun|Thn\.?)\s+(\d{4})"

    match = re.search(pattern, question, re.IGNORECASE)

    nomor, tahun = (None, None)
    if match:
        nomor = match.group(1)
        tahun = match.group(2)
        print(f"✅ Regex found: Nomor={nomor}, Tahun={tahun}")
    else:
        print("Regex found no specific UU number/year.")

    return {"nomor": nomor, "tahun": tahun}


graph.add_node("extract_nomor_tahun", extract_nomor_tahun)


# -----> Node 2: Retrieve docs by metadata filter
# def retrieve_docs(state):
#     nomor, tahun = state.get("nomor"), state.get("tahun")
#     results = retriever.invoke(state["input"], filters={
#                                "nomor": nomor, "tahun": tahun})

#     return {"docs": results}
def retrieve_docs(state):
    nomor, tahun = state.get("nomor"), state.get("tahun")
    # filters = {"nomor": nomor, "tahun": tahun} if nomor and tahun else None
    filters = {
        "$and": [
            {"nomor": {"$eq": nomor}},
            {"tahun": {"$eq": tahun}}
        ]
    } if nomor and tahun else None
    print(f"Retrieving documents with filters: {filters}")
    if filters is None:
        results = vectorstore.similarity_search(
            state["input"], k=5)
    else:
        results = vectorstore.similarity_search(
            state["input"], filter=filters, k=5)
    print(f"Retrieving documents found: {results}")
    return {"docs": results}


graph.add_node("retrieve_docs", retrieve_docs)

# -----> Node 3: Generate answer
# def generate_answer(state):

#     docs = state.get("docs", [])
#     combined_docs = "\n\n".join(doc.page_content for doc in docs)
#     prompt = (
#         f"Berikut adalah dokumen terkait Undang-Undang yang ditemukan:\n"
#         f"{combined_docs}\n\n"
#         f"Pertanyaan: {state['input']}\n"
#         f"Jawaban:"
#     )
#     if model_style == "gemini":
#         response = llm.invoke(prompt).content
#     else:
#         # response = llm.invoke(prompt)
#         response = ""
#         for chunk in generate(pipe, prompt):
#             print(chunk, end="", flush=True)
#             response += chunk
#     return {"output": response}
# --- 4. Corrected generate_answer Function ---


def generate_answer_node(state, model, tokenizer):
    """
    A self-contained, stateless node for generating answers with streaming.
    It creates a new streamer and pipeline for each call.
    """
    print("--- Entering generate_answer node ---")

    # 1. Create a NEW streamer for this specific run
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True)

    # 2. Create a NEW pipeline that uses the new streamer
    # This is fast because the model and tokenizer are already in memory
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        streamer=streamer,
        max_new_tokens=512
    )

    docs = state.get("docs", [])
    combined_docs = "\n\n".join(
        f"Pada Undang undang nomor {doc.metadata.get('nomor', '-')}, tahun {doc.metadata.get('tahun', '-')}, dan file sumber {doc.metadata.get('file_name', '-')}: {doc.page_content}" for doc in docs)
    prompt = (
        f"Berikut adalah dokumen terkait Undang-Undang yang ditemukan:\n"
        f"{combined_docs}\n\n"
        f"Pertanyaan: {state['input']} dan sebutkan nama file nya di awal jawaban berdasarkan file sumber yang saya berikan\n"
        f"Jawaban:"
    )

    # 3. Define the thread target with the CORRECT pipeline call
    def generation_thread_target():
        # Call the pipeline directly with the prompt
        pipe(prompt)

    # Start the generation in a separate thread
    thread = Thread(target=generation_thread_target)
    thread.start()

    # 4. Stream the response
    response = ""
    # Iterate over the new, single-use streamer
    for chunk in streamer:
        print(chunk, end="", flush=True)
        response += chunk

    # Wait for the thread to finish
    thread.join()
    print("\n--- Exiting generate_answer node ---")

    return {"output": response}


generate_answer_with_models = partial(
    generate_answer_node, model=model, tokenizer=tokenizer)
graph.add_node("generate_answer", generate_answer_with_models)


# -----> Define edges
graph.set_entry_point("extract_nomor_tahun")
graph.add_edge("extract_nomor_tahun", "retrieve_docs")
graph.add_edge("retrieve_docs", "generate_answer")
graph.add_edge("generate_answer", END)


# -----> Compile with checkpoint
# checkpointer = SqliteSaver("graph_checkpoints.db")
# app = graph.compile(checkpointer=checkpointer)
app = graph.compile()


# ================== RUN ====================
if __name__ == "__main__":
    # query = "Jelaskan isi Undang-Undang Nomor 33 Tahun 2014!"
    # query = "Jelaskan isi Undang-Undang Nomor 33 Tahun 2015!"
    query = "Undang-Undang apa saja yang membahas sengketa pajak! jawab dengan poin per poin nomor undang-undangnya,tahunnya dan penjelasan singkat isinya!"
    result = app.invoke(
        {"input": query},
        # configurable={"checkpoint_id": str(uuid.uuid4())}
    )
    # print("✅ Pertanyaan:", query)
    # print("\n📜 Jawaban:\n", result.get("output"))
