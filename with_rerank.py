import os
import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "default_collection")
PDF_PATHS = [
    "/home/khushi/pincone/Mobile.pdf",
    "/home/khushi/pincone/Mixer.pdf",
    "/home/khushi/pincone/laundry_center.pdf",
]

chroma_client = chromadb.HttpClient(
    host=CHROMA_HOST,
    port=CHROMA_PORT,
    settings=Settings()
)
embedding_function = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

def fce(PDF_PATHS):
    previous = 0 
    all_chunks = [] 
    chroma_client.delete_collection(name=COLLECTION_NAME)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' successfully initialized.")

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

        all_chunks.extend(chunks)

        embeddings = [embedding_function.encode(chunk.page_content) for chunk in chunks]
        print(f"Embedded {len(chunks)} chunks.")

        chunk_ids = [f"{pdf_name}_chunk_{i + previous}" for i in range(len(chunks))]
        previous += len(chunks)

        metadata = [{"source": pdf_name} for _ in range(len(chunks))]

        try:
            collection.add(
                ids=chunk_ids,
                embeddings=embeddings,
                metadatas=metadata
            )
            print(f"Added {len(chunks)} chunks from {pdf_name} to the collection.")
        except Exception as e:
            print(f"Error adding chunks for {pdf_name}: {e}")

    print("All PDFs processed and stored.")
    return collection, all_chunks 

collection, all_chunks = fce(PDF_PATHS)

try:
    query_text = "What are common washing problems"
    results = collection.query(
        query_texts=[query_text],
        n_results=5
    )

    print(f"Query Results: {results}")

    query_results_text = []
    chunk_ids = results['ids'][0]
    candidate_texts = [all_chunks[int(chunk_id.split('_')[-1]) - 1].page_content for chunk_id in chunk_ids]
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')
    scores = reranker.predict([(query_text, candidate) for candidate in candidate_texts])
    reranked_texts = [candidate_texts[i] for i in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)]

    print("Re-ranked Query Results:")
    for idx, text in enumerate(reranked_texts, 1):
        print(f"Rank {idx}:")
        print(f"Text: {text}")
        print("-" * 50)

except Exception as e:
    print(f"Error querying the collection: {e}")
 