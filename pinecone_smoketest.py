import os
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone

load_dotenv()

api_key = os.environ["PINECONE_API_KEY"]
index_name = os.environ.get("PINECONE_INDEX", "nasmrag")

pc = Pinecone(api_key = api_key)

# 1) list indexes to verify connection
print("Pinecone Indexes:", pc.list_indexes())

# 2) describe index to verify access
index = pc.Index(index_name)

# 3) fetch index status
stats = index.describe_index_stats()
print("Index Stats:", stats)