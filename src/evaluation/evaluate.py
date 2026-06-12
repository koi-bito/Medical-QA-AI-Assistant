"""
Evaluate the QA system on a fixed set of test questions.
Checks whether the answer contains expected medical topics.
Uses multiple aliases per topic to handle medical synonyms
(e.g., "urination" also matches "polyuria", "frequent urination").
"""
import json
from src.rag.pipeline import load_all, answer_question

# Each topic has a list of aliases — if ANY alias matches, the topic counts as a hit.
# This handles cases where the model uses medical terminology instead of plain language.
TEST_QUESTIONS = [
    {"question": "What are symptoms of type 2 diabetes?",
     "expected_topics": [
         ["blood sugar", "glucose", "hyperglycemia"],
         ["thirst", "polydipsia"],
         ["urination", "polyuria", "frequent urination"],
         ["fatigue", "tired", "exhaustion"],
         ["blurred", "vision"]
     ]},
    {"question": "How is high blood pressure diagnosed?",
     "expected_topics": [
         ["blood pressure"],
         ["mmhg", "mm hg", "millimeters"],
         ["measurement", "monitor", "reading"],
         ["systolic", "diastolic", "120/80"]
     ]},
    {"question": "What is ibuprofen used for?",
     "expected_topics": [
         ["pain", "analgesic"],
         ["inflammation", "anti-inflammatory", "swelling"],
         ["fever", "antipyretic", "temperature"],
         ["nsaid", "nonsteroidal", "non-steroidal"]
     ]},
    {"question": "What causes asthma?",
     "expected_topics": [
         ["airway", "airways", "bronchial"],
         ["inflammation", "inflamed", "swelling"],
         ["trigger", "allergen", "irritant"],
         ["breathing", "breathe", "respiratory", "wheez"]
     ]},
    {"question": "What are symptoms of depression?",
     "expected_topics": [
         ["mood", "feeling"],
         ["sleep", "insomnia", "fatigue"],
         ["interest", "anhedonia", "pleasure"],
         ["energy", "tired", "exhaustion"],
         ["sad", "hopeless", "worthless"]
     ]},
]

def topic_hit(answer, topic_aliases):
    """Check if any alias for a topic appears in the answer"""
    return any(alias in answer for alias in topic_aliases)

def evaluate():
    print("Loading models...")
    model, tokenizer, embedder, collection, reranker = load_all()

    results = []
    for q in TEST_QUESTIONS:
        answer = answer_question(
            q['question'], model, tokenizer, embedder, collection, reranker
        )['answer'].lower()

        hits  = [aliases for aliases in q['expected_topics'] if topic_hit(answer, aliases)]
        score = len(hits) / len(q['expected_topics'])
        results.append({
            "question": q['question'],
            "score": score,
            "matched_topics": len(hits),
            "total_topics": len(q['expected_topics'])
        })
        print(f"Q: {q['question'][:50]}... | Score: {score:.0%}")

    avg_score = sum(r['score'] for r in results) / len(results)
    print(f"\nAverage score: {avg_score:.1%}")

    with open("evaluation_results.json", "w") as f:
        json.dump({"avg_score": avg_score, "results": results}, f, indent=2)

    return avg_score

if __name__ == "__main__":
    score = evaluate()
    if score < 0.5:
        print("WARNING: Score below 50%")
        exit(1)
