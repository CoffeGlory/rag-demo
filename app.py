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

# -----------------------------
# Query from Pinecone, generate answer, and filter
# -----------------------------
# input variables: 
# question:str | user's question
# top_k:int | number of top similar chunks to retrieve
# output: list of (score, text) tuples
def query_doc(question: str, top_k: int):
    index = get_index()
    qv = embed_1024([question])[0]

    active_doc_id = st.session_state.get("active_doc_id")
    if not active_doc_id:  
        return []
    
    res = index.query(
        vector=qv,
        top_k=top_k,
        include_metadata=True,
        namespace=PINECONE_NAMESPACE,
        filter={"doc_id": active_doc_id}, #why there is a comma here, make it a tuple

    )
    matches = res.get("matches", [])
    # return (score, text) tuples
    return [(m.get("score", 0.0), m["metadata"].get("text", "")) for m in matches]


def generate_answer(question: str, chunks: list[str]) -> str:
    context = "\n\n".join([f"[Chunk {i+1}]\n{c}" for i, c in enumerate(chunks)])

    resp = oai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a rigorous RAG assistant.\n"
                    "Use ONLY the provided context. Do NOT use outside knowledge.\n"
                    "If the context does not explicitly define a term, you may still give a brief description "
                    "based on what the context says about it, and quote the exact phrases you relied on.\n"
                    "If the context provides no relevant information at all, say 'I don't know.'\n"
                    "Keep the answer short and factual."
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


# [Strict file based] This sections support openai only generate based on target PDF  
if "indexed_docs" not in st.session_state:
    st.session_state["indexed_docs"] = set()
# -----------------------------


with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top K", 1, 10, 5)
    # st.divider()

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

# [Strict file based] This sections support openai only generate based on target PDF  
# Store active doc_id in session_state
st.session_state["active_doc_id"] = doc_id
# -----------------------------

# -----------------------------
# Sidebar info
# -----------------------------
# Show current doc info (if any)
# TODO : hide this message if no PDF loaded
if pdf_bytes is not None:
    st.success(f"Loaded: {filename} | doc_id: {doc_id} | source: {source}")

active_doc_id = st.session_state.get("active_doc_id")
is_indexed = (
    active_doc_id in st.session_state["indexed_docs"]
    if active_doc_id
    else False
)


test_secrets_access()
with st.sidebar:
    # st.divider()
    st.caption(f"Active doc_id: {active_doc_id if active_doc_id else '(none)'}")
    # st.caption("Indexed: " + ("✅" if is_indexed else "❌ (not indexed yet)"))
    if is_indexed:
        st.sidebar.success("Indexed: YES")
    else:
        st.sidebar.warning("Indexed: NO (click Index)")

# -----------------------------
# Pinecone Insert button
# optimization: avoid re-indexing the same doc_id
# -----------------------------
# Pinecone Index button
# if it's already in pinecone session_state, skip indexing
if "indexed_docs" not in st.session_state:
    st.session_state["indexed_docs"] = set()

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
            st.rerun() # Refresh to update sidebar status
#TODO: this index is per session only, need to persist in real app


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

# -----------------------------
# Prompt section
# -----------------------------
prompt = st.chat_input("Ask a question about the PDF")
if prompt:
    st.sidebar.write("DEBUG: got prompt")
    st.sidebar.write("DEBUG active_doc_id:", st.session_state.get("active_doc_id"))
    st.sidebar.write("DEBUG indexed_docs size:", len(st.session_state.get("indexed_docs", set())))

    #Guard start
    active_doc_id = st.session_state.get("active_doc_id")
    # 严格模式 guard 1：必须有 active_doc_id
    if not active_doc_id:
        st.session_state["messages"].append({"role": "assistant", "content": "No active PDF loaded."})
        st.rerun()

    # 严格模式 guard 2：必须先 index
    if active_doc_id not in st.session_state.get("indexed_docs", set()):
        st.session_state["messages"].append({
            "role": "assistant",
            "content": "This PDF is not indexed yet. Please click **Index this PDF to Pinecone** first."
        })
        st.rerun()
    #Guard end

    
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
            retrieved = query_doc(prompt, top_k)
            st.sidebar.write("DEBUG: retrieved len:", len(retrieved))

            

        # last_retrieved means the last set of chunks retrieved from Pinecone
        st.session_state["last_retrieved"] = retrieved
        chunks_texts = [t for _, t in retrieved]
        with st.spinner("Generating answer..."):
            answer = generate_answer(prompt, chunks_texts)

        st.markdown(answer)
        st.sidebar.write("DEBUG: generated answer length:", len(answer) if answer else 0)


    # 3) 把助手回答写入历史
    st.session_state["messages"].append({"role": "assistant", "content": answer})


# -----------------------------
# 把 last Retrieved Chunks 放到侧边栏/折叠区，不打断聊天流
# -----------------------------
with st.sidebar:
    st.divider()
    st.subheader("Last Retrieved Chunks")
    retrieved = st.session_state.get("last_retrieved", [])
    if not retrieved:
        st.caption("No chunks retrieved yet.")
    else:
        for i, (score, text) in enumerate(retrieved, 1):
            with st.expander(f"Chunk #{i} (score: {score:.3f})", expanded=False):
                st.write(text)


