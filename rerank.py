import os
import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "default_collection")
PDF_PATHS = [
    "/home/khushi/pincone/civics1.pdf",
    "/home/khushi/pincone/history1.pdf",
    "/home/khushi/pincone/eco1.pdf",
    "/home/khushi/pincone/geo2.pdf",
    "/home/khushi/pincone/history2.pdf",
    "/home/khushi/pincone/geo1.pdf",
]

chroma_client = chromadb.HttpClient(
    host=CHROMA_HOST,
    port=CHROMA_PORT,
    settings=Settings()
)

embedding_function = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

def store_chunks_in_chromadb(pdf_paths):
    chroma_client.delete_collection(name=COLLECTION_NAME)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' successfully initialized.")

    previous = 0  

    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            print(f"PDF file not found at {pdf_path}")
            continue

        pdf_name = os.path.basename(pdf_path).split(".")[0]
        print(f"Processing PDF: {pdf_name}")

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        chunks = text_splitter.split_documents(documents)
        print(f"PDF split into {len(chunks)} chunks.")

        chunk_ids = [f"{pdf_name}_chunk_{i + previous}" for i in range(len(chunks))]
        previous += len(chunks)

        chunk_texts = [chunk.page_content for chunk in chunks]
        embeddings = embedding_function.encode(chunk_texts, batch_size=16)
        metadata = [{"source": pdf_name} for _ in range(len(chunks))]
        
        try:
            collection.add(
                ids=chunk_ids,
                embeddings=embeddings,
                metadatas=metadata,
                documents=chunk_texts
            )
            print(f"Added {len(chunks)} chunks from {pdf_name} to the collection.")
        except Exception as e:
            print(f"Error adding chunks for {pdf_name}: {e}")

    print("All PDFs processed and chunks stored in ChromaDB.")
    return collection
    
collection = store_chunks_in_chromadb(PDF_PATHS)

try:
    query_text = "atal bhujal yojna"
    results = collection.query(
        query_texts=[query_text],
        n_results=5
    )

    if not results['ids'] or not results['ids'][0]:
        print("No results found for the query.")
    else:
        print("Query Results:")
        for idx, (chunk_id, document, metadata, score) in enumerate(
            zip(results['ids'][0], results['documents'][0], results['metadatas'][0], results['distances'][0]), 1
        ):
            print(f"Rank {idx}:")
            print(f"Chunk ID: {chunk_id}")
            print(f"Text: {document}")
            print("-" * 50)

except Exception as e:
    print(f"Error querying the collection: {e}")

