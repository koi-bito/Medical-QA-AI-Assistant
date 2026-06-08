import requests
import json
import os

def fetch_medlineplus_articles():
    """Fetch health topic summaries from MedlinePlus API (public domain, free)"""
    os.makedirs("data/raw/medlineplus", exist_ok=True)
    articles = []

    topics = [
        "diabetes", "hypertension", "headache", "fever", "chest pain",
        "back pain", "depression", "anxiety", "asthma", "arthritis",
        "heart disease", "stroke", "kidney disease", "liver disease",
        "thyroid", "anemia", "pneumonia", "bronchitis", "sinusitis", "migraine"
    ]

    for topic in topics:
        url = "https://wsearch.nlm.nih.gov/ws/query"
        params = {"db": "healthTopics", "term": topic, "retmax": 25}
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                articles.append({"topic": topic, "content": response.text})
                print(f"Fetched: {topic}")
        except Exception as e:
            print(f"Failed: {topic} — {e}")

    with open("data/raw/medlineplus/articles.json", "w") as f:
        json.dump(articles, f)

    print(f"\nTotal articles fetched: {len(articles)}")
    return articles

if __name__ == "__main__":
    fetch_medlineplus_articles()
