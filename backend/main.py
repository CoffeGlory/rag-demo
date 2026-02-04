import io
import os
from dotenv import load_dotenv
from openai import OpenAI
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional, List
from pypdf import PdfReader
from pinecone.grpc import PineconeGRPC as Pinecone

load_dotenv()

app = FastAPI()


OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX = os.environ["PINECONE_INDEX"]          # nasmrag
PINECONE_NAMESPACE = os.environ.get("PINECONE_NAMESPACE", "pdf")

# OpenAI client
oai = OpenAI(api_key=OPENAI_API_KEY)

# Pinecone client
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
# ====== Models ======
class AskRequest(BaseModel):
    question: str
    top_k: int
    filename: Optional[str] | None = None
# Optional means that the field can be omitted when creating an instance of the class.
# The type hint Optional[str] | None indicates that the field can either be a string or None.

# ====== Utils ======
def exact_text_from_pdf_path(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    parts = []
    for page in reader.pages:
        text = page.extract_text() or ""
        parts.append(text)
    return "\n".join(parts)

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

# ====== API Endpoints ======
# health check endpoint
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/ask")
def ask(req: AskRequest):
    # For now, return a mock answer
    # 你先把一个 pdf 放到项目根目录，比如: E:\projects\rag-demo\data\demo.pdf
    pdf_path = r"E:\projects\rag-demo\data\NASM_CPT7_Study_Guide.pdf"

    text = extract_text_from_pdf_path(pdf_path)
    chunks_all = chunk_text(text)

    # 先把前 top_k chunks 返回给前端展示
    chunks = chunks_all[: req.top_k]

    answer = (
        "(Stage1) 我已经能从 PDF 抽取文本并切 chunk 了。\n"
        f"question={req.question}\n"
        f"file={req.filename}\n"
        f"total_chunks={len(chunks_all)}"
    )

    return {"answer": answer, "chunks": chunks}

@app.post("/ask_with_file")
def ask_with_file(
    question: str = Form(...),
    top_k: int = Form(...),
    file: UploadFile = File(...)
):
    # 1) 读取 PDF 文件内容（字节）
    pdf_bytes = file.file.read()

    # 2) 用 pypdf 从内存解析（不落地）
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text_parts = []
    for page in reader.pages:
        text_parts.append(page.extract_text() or "")
    text = "\n".join(text_parts)

    # 3) 切 chunk（用你现有的 chunk_text）
    chunks_all = chunk_text(text)
    chunks = chunks_all[:top_k]

    answer = (
        "(Stage1-Upload) 已解析你上传的 PDF\n"
        f"question={question}\n"
        f"filename={file.filename}\n"
        f"total_chunks={len(chunks_all)}"
    )

    return {
        "answer": answer,
        "chunks": chunks
    }

class AskRequest(BaseModel):
    question: str
    top_k: int = 5

@app.post("/ask")
def ask(req: AskRequest):
    # 1) embed question
    qv = embed_1024([req.question])[0]

    # 2) pinecone query
    res = pine_index.query(
        namespace=PINECONE_NAMESPACE,
        vector=qv,
        top_k=req.top_k,
        include_metadata=True,
    )

    # 3) extract chunks
    matches = res.get("matches", [])
    chunks = [m["metadata"].get("text", "") for m in matches]

    return {
        "answer": "(Stage: Retrieval) returned top_k chunks from Pinecone.",
        "chunks": chunks,
    }


