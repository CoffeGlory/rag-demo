import os
from dotenv import load_dotenv
from openai import OpenAI

# Load local embedding model
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


text = "RAG first step: embedding this sentence"
res = client.embeddings.create(
    model = "text-embedding-3-small",
    input = text
)

vec = res.data[0].embedding
print("vector length:", len(vec))
print("first 5 numbers:", vec[:5])