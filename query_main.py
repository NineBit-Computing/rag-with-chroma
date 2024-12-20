import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import CrossEncoder
import cohere
from sentence_transformers import SentenceTransformer

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
        embedding_function = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        my_query_embeddings = embedding_function.encode(query_text)
        # query_texts=[query_text],
        results = collection.query(
            query_embeddings=my_query_embeddings,
            n_results=3
        )
        print("hello")
        if not results['ids'] or not results['ids'][0]:
            print("No results found for the query.")
        else:
            print("Quering..............")
            return results
        
    except Exception as e:
        print(f"Error connecting to ChromaDB collection: {e}")


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
    
    generate_answer(query_text, results)
