# Firstly give command as --------- chroma run
import os
import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings


# Step 1: Configuration
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "default_collection")
PDF_PATH = "/home/khushi/pincone/laundry_center.pdf"  # Update with your PDF file path

chroma_client = chromadb.HttpClient(
    host=CHROMA_HOST,
    port=CHROMA_PORT,
    settings=Settings()
)

# Step 2: Load PDF
if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(f"PDF file not found at {PDF_PATH}")

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()
print("PDF loaded successfully.")

# Step 3: Split PDF into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)
print(f"PDF split into {len(chunks)} chunks.")
"""
for i, chunk in enumerate(chunks):
    print(f"Chunk {i + 1}:")
    print(chunk.page_content) 
    print("-" * 50)  
"""

# Step 4: Embed chunks

embedding_function = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
embeddings = [embedding_function.encode(chunk.page_content) for chunk in chunks]
print("Text chunks embedded into vectors.")


# Step 5: Store vectors in ChromaDB
try:
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    if collection:
        print(f"Collection '{COLLECTION_NAME}' successfully initialized.")
    else:
        raise ValueError(f"Failed to initialize or retrieve collection '{COLLECTION_NAME}'.")

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    records = [
        {
            "id": chunk_id,  
            "document": chunk.page_content,
            "metadata": {"source": chunk.metadata.get("source", "")},
            "embedding": embedding
        }
        for chunk_id, chunk, embedding in zip(ids, chunks, embeddings)
    ]
    
    # Extract lists for ids, embeddings, and metadata from the records
    record_ids = [record['id'] for record in records]
    record_embeddings = [record['embedding'] for record in records]
    record_metadatas = [record['metadata'] for record in records]

    # Batch-add all embeddings to the collection
    collection.add(
        ids=record_ids,
        embeddings=record_embeddings,
        metadatas=record_metadatas
    )
    
    print(f"Stored {len(chunks)} chunks in ChromaDB collection '{COLLECTION_NAME}'.")
    
except Exception as e:
    print(f"Error storing chunks in ChromaDB: {e}")

results = collection.query(
    query_texts=["What are the warnings"],
    n_results=1
)

print(results)

for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
    print(f"Chunk {i + 1}:")
    print(f"Text: {chunk.page_content}")  
    print(f"Embedding (Vector): {embedding}")  
    print("-" * 50)  

query_text = "What are the warnings"
n_results = 1  

try:
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )

    # Print the most relevant chunks for the query
    print(f"Query: {query_text}")
    if results['documents']:
        for i, document in enumerate(results['documents'][0]):
            print(f"Result {i + 1}:")
            print(document)  # Prints the relevant chunk's text
            print("-" * 50)
    else:
        print("No relevant results found.")
        
except Exception as e:
    print(f"Error querying ChromaDB: {e}")
