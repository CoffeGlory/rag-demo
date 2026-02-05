import io
import streamlit as st
import hashlib
from pypdf import PdfReader
from openai import OpenAI
from pinecone.grpc import PineconeGRPC as Pinecone
from pathlib import Path
import requests 

# ----------------------------
# Configuration
# set_page_config should be called only once, at the beginning
# its purpose is to set the title and layout of the Streamlit app
st.set_page_config(page_title="My RAG App", layout="wide")
DEFAULT_PDF_PATH = Path("data/NASM_CPT7_Study_Guide.pdf")

# ----------------------------
# Secrets (Streamlit Cloud uses st.secrets)
# Local dev: you can set these in .streamlit/secrets.toml too
# ----------------------------
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_INDEX = st.secrets["PINECONE_INDEX"]          # nasmrag
PINECONE_NAMESPACE = st.secrets.get("PINECONE_NAMESPACE", "__default__")
oai = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Connect pinecone index via host (stable)
@st.cache_resource
def get_index():
    indexes = pc.list_indexes()
    host = next((i["host"] for i in indexes if i["name"] == PINECONE_INDEX), None)
    if not host:
        raise RuntimeError(f"Pinecone index '{PINECONE_INDEX}' not found")
    return pc.Index(host=host)

# ----------------------------
# RAG functions
# Default: use 1024-dim embeddings for Pinecone, has to match the index config
# ----------------------------
def embed_1024(texts: list[str]) -> list[list[float]]:
    resp = oai.embeddings.create(
        model="text-embedding-3-large",
        input=texts,
        dimensions=1024,
    )
    return [d.embedding for d in resp.data]


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> list[str]:
    # 简单稳定：按字符切（先别做token切，等你上线再优化）
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + max_chars, n)
        chunks.append(text[i:j])
        i = j - overlap if j - overlap > i else j
    return [c.strip() for c in chunks if c.strip()]


# -----------------------------
# preprocess PDF, upsert to Pinecone, query from Pinecone
# -----------------------------
def read_pdf_bytes_to_text(pdf_bytes: bytes) -> str:
    # io here is to wrap bytes into a file-like object for PdfReader, previous version used file directly
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for p in reader.pages:
        pages.append(p.extract_text() or "")
    return "\n".join(pages)

def load_default_pdf():
    if not DEFAULT_PDF_PATH.exists():
        return None, None, None
    pdf_bytes = DEFAULT_PDF_PATH.read_bytes()
    doc_id = hashlib.sha1(pdf_bytes).hexdigest()[:12]
    return pdf_bytes, doc_id, DEFAULT_PDF_PATH.name

def upsert_doc(doc_id: str, chunks: list[str]):
    index = get_index()
    vecs = embed_1024(chunks)
    vectors = []
    for i, (c, v) in enumerate(zip(chunks, vecs)):
        vectors.append({
            "id": f"{doc_id}-{i}",
            "values": v,
            "metadata": {"text": c, "doc_id": doc_id, "chunk_id": i}
        })
    index.upsert(vectors=vectors, namespace=PINECONE_NAMESPACE)


def query_doc(question: str, top_k: int):
    index = get_index()
    qv = embed_1024([question])[0]
    res = index.query(
        vector=qv,
        top_k=top_k,
        include_metadata=True,
        namespace=PINECONE_NAMESPACE,
    )
    matches = res.get("matches", [])
    return [m["metadata"].get("text", "") for m in matches]   

def generate_answer(question: str, chunks: list[str]) -> str:
    context = "\n\n".join([f"[Chunk {i+1}]\n{c}" for i, c in enumerate(chunks)])

    resp = oai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer the user's question using ONLY the provided context. "
                    "If the answer is not in the context, say you don't know."
                ),
            },
            {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content









# -----------------------------
# Debug Test for serects access
# -----------------------------
def test_secrets_access():
    try:
        _ = OPENAI_API_KEY
        st.sidebar.success("Secrets access: OK")
        st.sidebar.success("OpenAI Pinecone: OK")
    except Exception as e:
        st.sidebar.error(f"Secrets access error: {e}")


# ----------------------------
# UI
# ----------------------------
st.title("RAG Demo (Streamlit + Pinecone)")
st.caption("Upload PDF → index to Pinecone → ask questions")
test_secrets_access()

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top K", 1, 10, 5)
    st.divider()

# -----------------------------
# PDF uploader
# -----------------------------
uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

pdf_bytes = None
doc_id = None
filename = None
source = None  # "upload" or "default"

if uploaded is not None:
    pdf_bytes = uploaded.read()
    doc_id = hashlib.sha1(pdf_bytes).hexdigest()[:12]
    filename = uploaded.name
    source = "upload"
else:
    b, did, name = load_default_pdf()
    if b is not None:
        pdf_bytes, doc_id, filename = b, did, name
        source = "default"
        st.info(f"Default PDF: {filename}")
    else:
        st.caption("Tip: Put a demo file at data/default.pdf to enable default loading.")
# Show current doc info (if any)
# TODO : hide this message if no PDF loaded
if pdf_bytes is not None:
    st.success(f"Loaded: {filename} | doc_id: {doc_id} | source: {source}")

# -----------------------------
# Small optimization: avoid re-indexing the same doc_id
# -----------------------------
if "indexed_docs" not in st.session_state:
    st.session_state["indexed_docs"] = set()

# Index button works for BOTH upload and default
# -----------------------------
# Pinecone Index
# if it's already in pinecone session_state, skip indexing
if pdf_bytes is not None:
    if st.button("Index this PDF to Pinecone"):
        if doc_id in st.session_state["indexed_docs"]:
            st.warning("This document is already indexed in this session.")
        else:
            with st.spinner("Parsing PDF..."):
                text = read_pdf_bytes_to_text(pdf_bytes)

            chunks = chunk_text(text)
            st.write(f"Total chunks: {len(chunks)}")

            with st.spinner("Embedding + Upserting to Pinecone..."):
                upsert_doc(doc_id, chunks)

            st.session_state["indexed_docs"].add(doc_id)
            st.success("Indexed to Pinecone successfully!")

# -----------------------------
# Chat state
# -----------------------------
if "messages" not in st.session_state:
    # Each message: {"role": "user"|"assistant", "content": "..."}
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello! I'm your RAG assistant. Please upload a PDF and ask me anything about it. (Mock)"}
    ]

# What's the purpose of last_retrieved?
# st.session_state["last_retrieved"] seems to be intended to store the last set of retrieved chunks from Pinecone for potential display or further processing.
if "last_retrieved" not in st.session_state:
    st.session_state["last_retrieved"] = []

# Render chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
# -----------------------------
# Chat input

prompt = st.chat_input("Ask a question about the PDF")
if prompt:
    # Default pdf exist

    # 1) 先显示用户消息
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2) 检索 + 生成（在 assistant 气泡里完成）
    with st.chat_message("assistant"):
        # if uploaded is None:
        #     st.warning("Please upload a PDF first, then click 'Index this PDF to Pinecone'.")
        #     st.stop()

        # st.spinner is a context manager that shows a spinner while the code inside is running
        with st.spinner("Retrieving from Pinecone..."):
            chunks = query_doc(prompt, top_k)

        # last_retrieved means the last set of chunks retrieved from Pinecone
        st.session_state["last_retrieved"] = chunks

        with st.spinner("Generating answer..."):
            answer = generate_answer(prompt, chunks)

        st.markdown(answer)

    # 3) 把助手回答写入历史
    st.session_state["messages"].append({"role": "assistant", "content": answer})

# -----------------------------
# 把 Retrieved Chunks 放到侧边栏/折叠区，不打断聊天流
# -----------------------------
with st.sidebar:
    st.divider()
    st.subheader("Last Retrieved Chunks")
    chunks = st.session_state.get("last_retrieved", [])
    if not chunks:
        st.caption("No chunks retrieved yet.")
    else:
        for i, c in enumerate(chunks, 1):
            with st.expander(f"Chunk #{i}", expanded=False):
                st.write(c)


