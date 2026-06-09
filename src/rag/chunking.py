import json
import re

def clean_text(text):
    """Remove XML/HTML tags and extra whitespace"""
    text = re.sub(r'<[^>]+>', ' ', text)     # Remove XML tags
    text = re.sub(r'\s+', ' ', text).strip() # Collapse whitespace
    return text

def chunk_text(text, chunk_size=400, overlap=50):
    """
    Split text into overlapping chunks.
    Overlap ensures a sentence isn't cut off at a boundary and lost.
    """
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk.strip()) > 100:  # Skip very short chunks
            chunks.append(chunk)

    return chunks

def process_all_articles():
    with open("data/raw/medlineplus/articles.json") as f:
        articles = json.load(f)

    all_chunks = []
    for article in articles:
        cleaned = clean_text(article['content'])
        chunks   = chunk_text(cleaned)
        for chunk in chunks:
            all_chunks.append({"topic": article['topic'], "text": chunk})

    with open("data/processed/chunks.json", "w") as f:
        json.dump(all_chunks, f)

    print(f"Total chunks created: {len(all_chunks)}")
    return all_chunks

if __name__ == "__main__":
    process_all_articles()
