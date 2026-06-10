import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

def load_retriever():
    embedder   = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")
    client     = chromadb.PersistentClient(path="data/chroma_db")
    collection = client.get_collection("medical_docs")
    return embedder, collection

def retrieve(query, collection, embedder, top_k=10):
    """Find the most similar document chunks to a user's question"""
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    return results['documents'][0]  # List of matching chunks

def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def retrieve_and_rerank(query, collection, embedder, reranker, top_k=10, final_k=3):
    # Step 1: Get top 10 candidates via vector search
    candidates = retrieve(query, collection, embedder, top_k=top_k)

    # Step 2: Score each candidate against the query
    pairs  = [(query, chunk) for chunk in candidates]
    scores = reranker.predict(pairs)

    # Step 3: Sort by score, keep best 3
    ranked     = sorted(zip(scores, candidates), reverse=True)
    top_chunks = [chunk for score, chunk in ranked[:final_k]]

    return top_chunks

if __name__ == "__main__":
    embedder, collection = load_retriever()
    reranker = load_reranker()
    test_questions = [
        "What are symptoms of type 2 diabetes?",
        "How is hypertension treated?",
        "What causes a migraine?",
        "What are the side effects of aspirin?",
        "How do you prevent heart disease?"
    ]
    for q in test_questions:
        print(f"\nQuestion: {q}")
        chunks = retrieve(q, collection, embedder)
        print(f"Top chunk (No Reranking): {chunks[0][:150]}...")
        
        reranked_chunks = retrieve_and_rerank(q, collection, embedder, reranker)
        print(f"Top chunk (With Reranking): {reranked_chunks[0][:150]}...")
