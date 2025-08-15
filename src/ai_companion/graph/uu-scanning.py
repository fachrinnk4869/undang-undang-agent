import torch
from peft import PeftModel
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFacePipeline
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
import uuid
import re
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
# ================== SETUP ====================

# -----> Embedding model
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

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
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)

    for pdf_file in tqdm(pdf_files, desc="Loading PDFs"):
        try:
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()
            match = re.search(r'uu(\d+)-(\d{4})', pdf_file.stem)
            nomor, tahun = match.groups() if match else ("-", "-")

            for doc in documents:
                doc.metadata["nomor"] = nomor
                doc.metadata["tahun"] = tahun
                doc.metadata["file_name"] = pdf_file.name

            splits = text_splitter.split_documents(documents)
            vectorstore.add_documents(splits)
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
# base_model = "microsoft/Phi-3-mini-4k-instruct"
# tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(
#     base_model,
#     trust_remote_code=False,
#     device_map="auto",
#     # load_in_4bit=True,
#     torch_dtype=torch.float16,
# )
# # Load adapter if available
# adapter_dir = "./phi3-pdf-indo-lora/checkpoint-49830"
# model = PeftModel.from_pretrained(model, adapter_dir)
# model = model.merge_and_unload()

# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     device_map="auto",
#     max_new_tokens=500,
#     top_k=50,
#     temperature=0.1,
# )
# llm = HuggingFacePipeline(pipeline=pipe)

# -----> Load Google Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
)
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
def extract_nomor_tahun(state):
    question = state["input"]
    prompt = (
        f"Ambil nomor dan tahun undang undang dari pertanyaan berikut dengan akurat:\n"
        f"{question}\n\n"
        f"Format jawaban: nomor=[nomor], tahun=[tahun]"
    )
    if model_style == "gemini":
        response = llm.invoke(prompt).content
    else:
        response = llm.invoke(prompt)
    match = re.search(r"nomor\s*=\s*(\d+).+tahun\s*=\s*(\d+)",
                      response, re.IGNORECASE | re.DOTALL)
    if match:
        nomor, tahun = match.groups()
    else:
        nomor, tahun = None, None
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
            state["input"], k=10)
    else:
        results = vectorstore.similarity_search(
            state["input"], filter=filters)
    print(f"Retrieving documents found: {results}")
    return {"docs": results}


graph.add_node("retrieve_docs", retrieve_docs)


# -----> Node 3: Generate answer
def generate_answer(state):
    docs = state.get("docs", [])
    combined_docs = "\n\n".join(doc.page_content for doc in docs)
    prompt = (
        f"Berikut adalah dokumen terkait Undang-Undang yang ditemukan:\n"
        f"{combined_docs}\n\n"
        f"Pertanyaan: {state['input']}\n"
        f"Jawaban:"
    )
    if model_style == "gemini":
        response = llm.invoke(prompt).content
    else:
        response = llm.invoke(prompt)
    return {"output": response}


graph.add_node("generate_answer", generate_answer)


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
    query = "Undang-Undang apa saja yang membahas sengketa pajak! jawab dengan singkat poin per poin nomor undang-undangnya dan tahunnya!"
    result = app.invoke(
        {"input": query},
        # configurable={"checkpoint_id": str(uuid.uuid4())}
    )
    print("✅ Pertanyaan:", query)
    print("\n📜 Jawaban:\n", result.get("output"))
