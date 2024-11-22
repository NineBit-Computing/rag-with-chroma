import os
import chromadb
from chromadb.config import Settings

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "default_collection")
PDF_PATH = "/home/khushi/pincone/laundry_center.pdf" 

chroma_client = chromadb.HttpClient(
    host=CHROMA_HOST,
    port=CHROMA_PORT,
    settings=Settings()
)

collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

results = collection.query(
    query_texts=["What are the warnings"],
    n_results=1
)

print(results)
"""
for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
    print(f"Text: {chunk[0].page_content}")  
    print(f"Metadata: {chunk[0].metadata}") 

for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
    print(f"Chunk {i + 1}:")
    print(f"Text: {chunk[7].page_content}")  
    #print(f"Embedding (Vector): {embedding}")  
    print("-" * 50)  
 
"""