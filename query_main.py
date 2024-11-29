import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import CrossEncoder
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
"""Function for giving related chunks"""
def query_chromadb(query_text): 
    try:
        collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
        print(f"Connected to collection: {COLLECTION_NAME}")
        results = collection.query(
            query_texts=[query_text],
            n_results=3
        )

        if not results['ids'] or not results['ids'][0]:
            print("No results found for the query.")
        else:
            print("Quering..............")
            return results
        
    except Exception as e:
        print(f"Error connecting to ChromaDB collection: {e}")

"""For reranking the chunks"""
def rerank_results(query_text, results):
    try:

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
        return best_chunk_text
    except Exception as e:
        print(f"Error during re-ranking: {e}")
        return None, None

"""For generating the answer"""
def generate_answer(query_text, context):
    try:
        response = co.generate(
            model='command-xlarge-nightly',
            prompt = f"Use the information provided below to answer the questions at the end. If the answer to the question is not contained in the provided information, say The answer is not in the context.Context information:{context} Question: {query_text}",
            #prompt=f"Question: {query_text}\n\nContext: {context}\n\nAnswer:",
            max_tokens=100,
            temperature=0.7
        )
        answer = response.generations[0].text.replace("\n", " ").strip()
        print(f"Answer: {answer}")
        return answer
    except Exception as e:
        print(f"Error generating answer: {e}")
        return None

while True:
    query_text = input("Enter your question here or print exit to Exit.... \n")

    if query_text.lower() == 'exit':
        print("Exiting.......!")
        break 

    results = query_chromadb(query_text)
    
    best_chunk_text = rerank_results(query_text, results)

    generate_answer(query_text, best_chunk_text)
