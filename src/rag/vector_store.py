import chromadb
from sentence_transformers import SentenceTransformer
import json

def build_vector_store(chunks_path="data/processed/chunks.json"):
    with open(chunks_path) as f:
        chunks = json.load(f)

    print(f"Loading embedding model...")
    # PubMedBERT is trained on medical text — better than a generic embedder
    embedder = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")

    print("Setting up ChromaDB...")
    client     = chromadb.PersistentClient(path="data/chroma_db")
    collection = client.get_or_create_collection("medical_docs")

    print(f"Embedding {len(chunks)} chunks...")
    batch_size = 100

    for i in range(0, len(chunks), batch_size):
        batch      = chunks[i:i + batch_size]
        texts      = [c['text'] for c in batch]
        embeddings = embedder.encode(texts).tolist()
        ids        = [f"chunk_{i + j}" for j in range(len(batch))]
        metadatas  = [{"topic": c['topic']} for c in batch]

        collection.add(embeddings=embeddings, documents=texts, ids=ids, metadatas=metadatas)

        if (i // batch_size) % 5 == 0:
            print(f"  Progress: {min(i + batch_size, len(chunks))}/{len(chunks)}")

    print("Vector store built and saved to data/chroma_db/")
    return collection

if __name__ == "__main__":
    build_vector_store()
