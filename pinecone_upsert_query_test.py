import os
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone
from openai import OpenAI

load_dotenv()

# --- keys ---
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = os.environ["PINECONE_INDEX"]

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# --- connect pinecone index via host ---
indexes = pc.list_indexes()
host = next((i["host"] for i in indexes if i["name"] == index_name), None)
if not host:
    raise RuntimeError(f"Index '{index_name}' not found.")
index = pc.Index(host=host)

def embed_1024(texts: list[str]) -> list[list[float]]:
    # 这个模型输出 1024 维（用于匹配你 index 的 dimension=1024）
    resp = client.embeddings.create(
        model="text-embedding-3-large",
        input=texts,
        dimensions=1024,
    )
    return [d.embedding for d in resp.data]

docs = [
    "Progressive overload means gradually increasing training stimulus over time.",
    "The OPT model is a NASM training framework with phases for stabilization, strength, and power.",
]

# 1) embed docs -> upsert
doc_vecs = embed_1024(docs)
index.upsert(
    vectors=[
        {"id": "demo-0", "values": doc_vecs[0], "metadata": {"text": docs[0]}},
        {"id": "demo-1", "values": doc_vecs[1], "metadata": {"text": docs[1]}},
    ]
)

# 2) embed query -> query
q = "what is progressive overload?"
qv = embed_1024([q])[0]

res = index.query(
    vector=qv,
    top_k=2,
    include_metadata=True,
)

print("Matches:")
for m in res["matches"]:
    print("-", m["score"], m["id"], m["metadata"]["text"])
