# test_app.py
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# -------------------
# CONFIG
# -------------------
BUGS_FILE = "data/dummy_bugs.jsonl"   # your bug dataset
MODEL_NAME = "all-MiniLM-L6-v2"       # same model used during indexing
FAISS_DIM = 384                       # embedding dimension
TOP_K = 10                            # number of results to retrieve

# -------------------
# Load dataset
# -------------------
bugs = []
with open(BUGS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        bugs.append(json.loads(line))

# -------------------
# Load model & FAISS index
# -------------------
print("📥 Loading model & building FAISS index...")
model = SentenceTransformer(MODEL_NAME)

bug_texts = [f"{b['summary']} - {b['description']} ({b['platform']})" for b in bugs]
embeddings = model.encode(bug_texts, convert_to_numpy=True)

index = faiss.IndexFlatL2(FAISS_DIM)
index.add(embeddings)
print(f"✅ FAISS index built with {len(bugs)} bug reports")

# -------------------
# Search function
# -------------------
def search_bugs(query, k=TOP_K, keyword_match=True):
    query_vec = model.encode([query], convert_to_numpy=True)
    D, I = index.search(np.array(query_vec), k)

    results = [(bugs[idx], D[0][rank]) for rank, idx in enumerate(I[0])]

    if keyword_match:
        keyword_hits = [
            (bug, 0.0) for bug in bugs
            if query.lower() in bug["summary"].lower() or query.lower() in bug["description"].lower()
        ]
        results_dict = {r[0]["summary"]: r for r in results}
        for bug, score in keyword_hits:
            results_dict[bug["summary"]] = (bug, score)
        results = list(results_dict.values())

    return results

# -------------------
# Severity & Priority heuristics
# -------------------
def suggest_severity_priority(bug):
    text = (bug["summary"] + " " + bug["description"]).lower()

    if any(word in text for word in ["crash", "data loss", "corruption", "unresponsive", "failover"]):
        return "Severity: Critical", "Priority: High"

    if any(word in text for word in ["latency", "delay", "slow", "high cpu", "memory leak", "timeout"]):
        return "Severity: Major", "Priority: Medium"

    if any(word in text for word in ["ui", "format", "glitch", "display", "alignment"]):
        return "Severity: Minor", "Priority: Low"

    return "Severity: Major", "Priority: Medium"  # default

# -------------------
# Interactive loop
# -------------------
print("\n💡 Type a query (or 'exit' to quit):")
while True:
    query = input("Query: ")
    if query.lower() in ["exit", "quit"]:
        break

    results = search_bugs(query, k=TOP_K, keyword_match=True)

    if not results:
        print("⚠️ No matches found.")
        continue

    print("\n🔎 Matches:")
    for i, (bug, score) in enumerate(results, 1):
        print(f"{i}. {bug['summary']} ({bug['platform']})")
        print(f"   {bug['description'][:70]}...")
        print(f"   Score: {score:.4f}\n")

    try:
        choice = int(input("Select issue number to view full defect: ")) - 1
        if choice < 0 or choice >= len(results):
            print("⚠️ Invalid choice.")
            continue
    except ValueError:
        print("⚠️ Please enter a valid number.")
        continue

    selected = results[choice][0]

    print("\n📌 Full Defect Description:")
    print(selected["description"])

    severity, priority = suggest_severity_priority(selected)
    print(severity)
    print(priority)
    print("-" * 80)
