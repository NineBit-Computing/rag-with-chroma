import os
import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Step 1: Configuration
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "default_collection")
PDF_PATHS = [
    "/home/khushi/pincone/laundry_center.pdf",
    "/home/khushi/pincone/Mixer.pdf",
    "/home/khushi/pincone/Mobile.pdf"
]

chroma_client = chromadb.HttpClient(
    host=CHROMA_HOST,
    port=CHROMA_PORT,
    settings=Settings()
)

chroma_client.delete_collection(name=COLLECTION_NAME)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
print(f"Collection '{COLLECTION_NAME}' successfully initialized.")

embedding_function = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# Step 2: Process PDFs
for pdf_path in PDF_PATHS:
    if not os.path.exists(pdf_path):
        print(f"PDF file not found at {pdf_path}")
        continue
    
    pdf_name = os.path.basename(pdf_path).split(".")[0]
    print(f"Processing PDF: {pdf_name}")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    chunks = text_splitter.split_documents(documents)
    print(f"PDF split into {len(chunks)} chunks.")

    embeddings = [embedding_function.encode(chunk.page_content) for chunk in chunks]
    print(f"Embedded {len(chunks)} chunks.")

    chunk_ids = [f"{pdf_name}_chunk_{i}" for i in range(len(chunks))]
    metadata = [{"source": pdf_name} for _ in range(len(chunks))]

    collection.add(
        ids=chunk_ids,
        embeddings=embeddings,
        metadatas=metadata
    )
    print(f"Added {len(chunks)} chunks from {pdf_name} to the collection.")

print("All PDFs processed and stored.")

try:
    query_text = "What are common washing problems"
    results = collection.query(
        query_texts=[query_text],
        n_results=1
    )

    print(f"Query Results: {results}")

    chunk_id = results['ids'][0][0]
    chunk_index = int(chunk_id.split('_')[-1]) - 1

    if 0 <= chunk_index < len(chunks):
        print(f"Chunk {chunk_index + 1}:")
        print(f"Text: {chunks[chunk_index].page_content}")
        print("-" * 50)
    else:
        print(f"Chunk ID {chunk_id} is out of range.")
except Exception as e:
    print(f"Error querying the collection: {e}")
