import streamlit as st
import hashlib
from pypdf import PdfReader
from openai import OpenAI
from pinecone.grpc import PineconeGRPC as Pinecone
import requests

st.set_page_config(page_title="My RAG App", layout="wide")

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
_indexes = pc.list_indexes()
_host = next((i["host"] for i in _indexes if i["name"] == PINECONE_INDEX), None)
if not _host:
    st.error(f"Pinecone index '{PINECONE_INDEX}' not found.")
    st.stop()

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

def read_pdf_bytes_to_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(pdf_bytes)
    pages = []
    for p in reader.pages:
        pages.append(p.extract_text() or "")
    return "\n".join(pages)

def upsert_doc(doc_id: str, chunks: list[str]):
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
    qv = embed_1024([question])[0]
    res = index.query(
        vector=qv,
        top_k=top_k,
        include_metadata=True,
        namespace=PINECONE_NAMESPACE,
    )
    matches = res.get("matches", [])
    return [m["metadata"].get("text", "") for m in matches]   

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="My RAG App", layout="wide")
st.title("RAG Demo (Streamlit + Pinecone)")
st.caption("Upload PDF → index to Pinecone → ask questions")

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top K", 1, 10, 5)
    st.divider()

uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded:
    pdf_bytes = uploaded.read()
    doc_id = hashlib.sha1(pdf_bytes).hexdigest()[:12]
    st.success(f"Uploaded: {uploaded.name}  |  doc_id: {doc_id}")

    if st.button("Index this PDF to Pinecone"):
        with st.spinner("Parsing PDF..."):
            text = read_pdf_bytes_to_text(pdf_bytes)
        chunks = chunk_text(text)
        st.write(f"Total chunks: {len(chunks)}")

        with st.spinner("Embedding + Upserting to Pinecone..."):
            upsert_doc(doc_id, chunks)

        st.success("Indexed to Pinecone ✅")

question = st.text_input("Ask a question about the PDF")
if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Retrieving from Pinecone..."):
        chunks = query_doc(question, top_k)

    st.subheader("Retrieved Chunks")
    for i, c in enumerate(chunks, 1):
        with st.expander(f"Chunk #{i}", expanded=False):
            st.write(c)


# #---------------
# # Mock RAG answer (no API)
# # Later will replace with retrieve + LLM
# #---------------
# def rag_answer_mock(question: str, top_k: int, uploaded_file):
#     resp = requests.post(
#         "http://127.0.0.1:8000/ask_with_file",
#         data={
#             "question": question,
#             "top_k": str(top_k),
#         },
#         files={
#             "file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")
#         },
#         timeout=60,
#     )

#     # if not 200, raise error
#     if resp.status_code != 200:
#         raise RuntimeError(
#             f"Backend error: {resp.status_code}\n"
#             f"{resp.text}"
#         )

#     #  get answer and chunks at 200
#     data = resp.json()
#     return data["answer"], data["chunks"]

# #---------------
# # Sidebar: setting
# #---------------
# with st.sidebar:
#     st.header("Settings")
#     top_k = st.slider("Top K", min_value=1, max_value=20, value=5)
#     use_mock = st.checkbox("Use mock answer", value=True)
#     st.divider()
#     if st.button("Clear chat", use_container_width=True):
#         st.session_state.pop("messages", None)
#         st.session_state.pop("last_chunks", None)
#         st.rerun()

# # -----------------------------
# # Title
# # -----------------------------
# st.title("Rag Demo App")
# st.caption("ui by Streamlit + backend by FastAPI")


# # -----------------------------
# # PDF uploader
# # -----------------------------
# uploaded = st.file_uploader("Upload a PDF file", type=["pdf"])
# if uploaded is not None:
#     st.success(f"Uploaded: {uploaded.name} ({uploaded.size} bytes)")

# # -----------------------------
# # Chat state
# # -----------------------------
# if "messages" not in st.session_state:
#     # Each message: {"role": "user"|"assistant", "content": "..."}
#     st.session_state["messages"] = [
#         {"role": "assistant", "content": "Hello! I'm your RAG assistant. Please upload a PDF and ask me anything about it. (Mock)"}
#     ]

# # Render chat history
# for msg in st.session_state["messages"]:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])

# # -----------------------------
# # Chat input
# # -----------------------------
# prompt = st.chat_input("Ask a question about the PDF...")

# if prompt:
#     # Guard: must upload PDF first
#     if uploaded is None:
#         with st.chat_message("assistant"):
#             st.error("Please upload a PDF file first.")
#         st.stop()

#     # 1) Show user message
#     st.session_state["messages"].append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)
    
#     # 2) Get RAG answer
#     with st.chat_message("assistant"):
#         answer = ""
#         chunks = []
#         try:
#             with st.spinner("Thinking..."):
#                 if use_mock:
#                     answer, chunks = rag_answer_mock(prompt, top_k, uploaded)
#                 else:  
#                     answer = "turn on the backend to get real answer."
#                     chunks =[]
#                 st.markdown(answer)
#         except Exception as e:
#             st.error("Failed to call backend.")
#             st.code(str(e))
#             chunks = []
        

#         #Store chunks for optional display
#         st.session_state["last_chunks"] = chunks

#         #Expandable retrieved chunks
#         if chunks:
#             with st.expander("Retrieved Chunks", expanded=False):
#                 for i, c in enumerate(chunks, start=1):
#                     st.markdown(f"**Chunk #{i}**")
#                     st.write(c)
#     # 3) Save assistant message
#     st.session_state["messages"].append({"role": "assistant", "content": answer})