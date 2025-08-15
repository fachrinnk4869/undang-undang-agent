from typing import TypedDict
from tqdm import tqdm
from pathlib import Path
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline

import re

# ------------------ SETUP ------------------
PDF_FOLDER = "assets/scrap"
DB_PATH_CONTENT = "chroma_db_content"
DB_PATH_TITLES = "chroma_db_titles"
CHECKPOINT_PATH = "graph_checkpoints.db"
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"


folder_path = Path(PDF_FOLDER)
pdf_files = list(folder_path.glob("*.pdf"))

all_docs = []
all_titles = []

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
if not os.path.exists(DB_PATH_CONTENT):
    for pdf_file in tqdm(pdf_files, desc="Loading PDFs"):
        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()

        title = pdf_file.stem.replace("_", " ")
        all_titles.append(Document(page_content=title))

        for doc in docs:
            splits = splitter.split_text(doc.page_content)
            for split in splits:
                all_docs.append(Document(page_content=split))
    vectorstore_content = Chroma.from_documents(
        all_docs, embedding=embeddings, persist_directory=DB_PATH_CONTENT)
    vectorstore_titles = Chroma.from_documents(
        all_titles, embedding=embeddings, persist_directory=DB_PATH_TITLES)
else:
    print("Database already exists, skipping PDF loading.")
    vectorstore_content = Chroma(
        embedding_function=embeddings,
        persist_directory=DB_PATH_CONTENT)
    vectorstore_titles = Chroma(
        embedding_function=embeddings,
        persist_directory=DB_PATH_TITLES)

vectorstore_content.persist()
vectorstore_titles.persist()


# ------------------ STEP 3: MODEL PIPELINE ------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, trust_remote_code=False, device_map="auto")
pipe = pipeline("text-generation", model=model,
                tokenizer=tokenizer, max_new_tokens=300, temperature=0.1)
llm = HuggingFacePipeline(pipeline=pipe)

# ------------------ STEP 4: LANGGRAPH ------------------


def extract_topic_node(state):
    prompt = f"""
Tentukan Undang-Undang apa yang relevan dari pertanyaan berikut:
Pertanyaan: {state['input']}
Jawab hanya dengan judul atau nomor Undang-Undang yang relevan.
"""
    topic = llm.invoke(prompt)
    state["topic"] = topic
    return state


def retrieve_title_node(state):
    retriever = vectorstore_titles.as_retriever(search_kwargs={"k": 1})
    results = retriever.invoke(state["topic"])

    if results:
        state["matched_title"] = results[0].page_content
    else:
        state["matched_title"] = ""

    return state


def retrieve_content_node(state):
    retriever = vectorstore_content.as_retriever(search_kwargs={"k": 10})
    results = retriever.invoke(state["matched_title"])

    context = "\n\n".join([doc.page_content for doc in results])
    state["context"] = context
    return state


def generate_answer_node(state):
    prompt = f"""
Berikut adalah isi terkait Undang-Undang:
{state.get("context", "Tidak ditemukan.")}

Pertanyaan: {state['input']}
Jawaban:
"""
    output = llm.invoke(prompt)
    state["output"] = output
    return state

# Create the structure of the schema for the graph.


class MyGraphState(TypedDict):
    input: str
    topic: str
    matched_title: str
    context: str
    output: str


# Graph definition
graph = StateGraph(MyGraphState)
graph.add_node("extract_topic", extract_topic_node)
graph.add_node("retrieve_title", retrieve_title_node)
graph.add_node("retrieve_content", retrieve_content_node)
graph.add_node("generate", generate_answer_node)

graph.set_entry_point("extract_topic")
graph.add_edge("extract_topic", "retrieve_title")
graph.add_edge("retrieve_title", "retrieve_content")
graph.add_edge("retrieve_content", "generate")
graph.add_edge("generate", END)

# store = SqliteSaver(CHECKPOINT_PATH)
# app = graph.compile(checkpointer=store)
app = graph.compile()

# Example usage:
query = "Jelaskan isi Undang-Undang Nomor 51 Tahun 2009!"
result = app.invoke({"input": query})
print(result["output"])
