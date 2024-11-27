import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import CrossEncoder
from transformers import pipeline
import cohere

co = cohere.Client('A30oAcq9woG1QXsN2ZWE5oNEUSPjOy3lLqXFK7bK')

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "default_collection")

chroma_client = chromadb.HttpClient(
    host=CHROMA_HOST,
    port=CHROMA_PORT,
    settings=Settings()
)

try:
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    print(f"Connected to collection: {COLLECTION_NAME}")
except Exception as e:
    print(f"Error connecting to ChromaDB collection: {e}")
    exit()

query_text = "What is atal bhujal yojna"
try:
    results = collection.query(
        query_texts=[query_text],
        n_results=3
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
        
        candidate_texts = results['documents'][0]
        candidate_ids = results['ids'][0]          
        reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')

        scores = reranker.predict([(query_text, candidate) for candidate in candidate_texts])

        reranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        print("Reranked Query Results:")
        for rank, idx in enumerate(reranked_indices, 1):
            print(f"Rank {rank}:")
            print(f"Chunk ID: {candidate_ids[idx]}")
            print(f"Score: {scores[idx]:.4f}")
            print(f"Text: {candidate_texts[idx]}")
            print("-" * 50)

        best_chunk_text = [candidate_texts[i] for i in reranked_indices]  

        response = co.generate(
            model='command-xlarge-nightly',  
            prompt=f"Question: {query_text}\n\nContext: {best_chunk_text}\n\nAnswer:",
            max_tokens=100,  
            temperature=0.7
        )

        print(f"Answer: {response.generations[0].text.strip()}")

    print(f"Before rerank: {results['ids'][0]}")
    reranked_ids = [candidate_ids[i] for i in reranked_indices]  
    print(f"After rerank: {reranked_ids}")

except Exception as e:
    print(f"Error querying the collection: {e}")
 
