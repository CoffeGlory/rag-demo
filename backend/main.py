import io
import os
# from dotenv import load_dotenv
from openai import OpenAI
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional, List
from pypdf import PdfReader
from pinecone.grpc import PineconeGRPC as Pinecone

# load_dotenv()

app = FastAPI()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")         # nasmrag
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "pdf")

# client
oai = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# find host based on list_indexes
indexes = pc.list_indexes()
host = next((i["host"] for i in indexes if i["name"] == PINECONE_INDEX), None)
if not host:
    raise RuntimeError(f"Index '{PINECONE_INDEX}' not found in this Pinecone project.")

pine_index = pc.Index(host=host)


# what is class in python?
# class is a blueprint for creating objects. 
# It defines a set of attributes and methods that the created objects (instances) will have. 
# Classes allow for encapsulation of data and functionality, enabling code reuse and organization in object-oriented programming.

# AskRequest method is used to define the structure of the request body for the /ask endpoint.
def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> List[str]:
    #TODO tokenize later
    text = text.replace("\r\n", "\n")
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

def embed_1024(texts: List[str]) -> List[List[float]]:
    resp = oai.embeddings.create(
        model="text-embedding-3-large",
        input=texts,
        dimensions=1024,
    )
    return [d.embedding for d in resp.data]

# Models
class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    doc_id: Optional[str] = None


# ====== API Endpoints ======
# health check endpoint
@app.get("/")
def root():
    return {
        "msg": "RAG backend is running",
        "endpoints": ["/health", "/ask", "/ask_with_file", "/docs"]
    }

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/ask")
def ask(req: AskRequest):
    # 1) embed question
    qv = embed_1024([req.question])[0]

    # 2) pinecone query
    query_kwargs = dict(
        namespace=PINECONE_NAMESPACE,
        vector=qv,
        top_k=req.top_k,
        include_metadata=True,
    )
    if req.doc_id:
        query_kwargs["filter"] = {"doc_id": req.doc_id}

    # 3) extract chunks and scores
    res = pine_index.query(**query_kwargs)
    matches = res.get("matches", [])
    chunks = [m.get("metadata", {}).get("text", "") for m in matches]
    scores = [m.get("score", 0.0) for m in matches]

    return {"answer": "(Retrieval) returned chunks from Pinecone", "chunks": chunks, "scores": scores}


@app.post("/ask_with_file")
def ask_with_file(
    question: str = Form(...),
    top_k: int = Form(...),
    file: UploadFile = File(...)
):
    # 1) read pdf bytes from UploadFile
    pdf_bytes = file.file.read()

    # 2) extract text from pdf bytes
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text_parts = []
    for page in reader.pages:
        text_parts.append(page.extract_text() or "")
    text = "\n".join(text_parts)

    # 3) chunk text and return top k chunks
    chunks_all = chunk_text(text)
    chunks = chunks_all[:top_k]

    answer = (
        f"question={question}\n"
        f"filename={file.filename}\n"
        f"total_chunks={len(chunks_all)}"
    )

    return {
        "answer": answer,
        "chunks": chunks
    }
