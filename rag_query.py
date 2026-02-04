import os
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

import chromadb

#1) read .env
load_dotenv()

#2) local embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

#3) local ChromaDB
db_path = r"E:\projects\rag-demo\chroma_db"
client = chromadb.PersistentClient(path=db_path)
collection = client.get_collection(name="docs")

#4) OpenAI client
# question = "What is RAG and why do people use it?"
question = input("Ask a question: ").strip()

#5) get embedding for question 1d vector
q_emb = embed_model.encode(question).tolist()

#6) query ChromaDB for top 3 similar docs
res = collection.query(
    query_embeddings=[q_emb],
    n_results=3,
    include=["documents", "metadatas", "distances"]
)

docs = res["documents"][0]
metas = res["metadatas"][0]
dists = res["distances"][0]

context_lines = []
sources_lines = []
for i, (d,m,dist) in enumerate(zip(docs, metas, dists), start = 1):
    context_lines.append(f"[{i}] {d}")
    sources_lines.append(f"[{i}] source={m.get('source')} page={m.get('page', '-')}, chunk={m.get('chunk', '-')}, dist={dist:.4f}")
context = "\n".join(context_lines)

print("=== Retrieved Context ===")
print(context)

print("\n=== Retrieved Sources (debug) ===")
print("\n".join(sources_lines))

#6) generate answer with OpenAI
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise SystemExit("Error: OPENAI_API_KEY not found in environment variables. Check your .env file.")

client_ai = OpenAI(api_key=api_key)

prompt = f"""
You are answering with citations.

Rules:
- Use ONLY the context. Do not add outside facts.
- Write a clear explanation in 3-5 bullet points.
- Each bullet must end with citations like [1] or [1][2].
- If the context lacks details to explain "how it works", explicitly say what is missing.

Context:
{context}

Question: 
{question}
""".strip()

resp = client_ai.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.2,
)
answer = resp.choices[0].message.content
print("\n=== Answer ===")
print(answer)

print("\n=== Sources ===")
# print source details for debugging
for line in sources_lines:
    print(line)