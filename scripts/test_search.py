# scripts/test_search.py
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_PATH = "data/dummy_bugs.jsonl"   # or your LSE dataset
INDEX_PATH = "index/faiss_index.bin"  # built by build_index_dummy.py
EMB_PATH = "index/embeddings.npy"

# ---------------- Load dataset ----------------
with open(DATA_PATH, "r", encoding="utf-8") as f:
    bugs = [json.loads(line) for line in f if line.strip()]

# Load FAISS index + embeddings
index = faiss.read_index(INDEX_PATH)
embeddings = np.load(EMB_PATH)

# Model for embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")


# ---------------- Search ----------------
def search_bugs(query, k=10, keyword_match=True):
    """Search FAISS, optionally augment with keyword matches."""
    query_vec = model.encode([query], convert_to_numpy=True)
    D, I = index.search(np.array(query_vec), k)

    results = [(bugs[idx], D[0][rank]) for rank, idx in enumerate(I[0])]

    if keyword_match:
        keyword_hits = [
            (bug, 0.0) for bug in bugs
            if query.lower() in bug["summary"].lower() or query.lower() in bug["description"].lower()
        ]
        # Merge FAISS + keyword results (avoid duplicates)
        results_dict = {r[0]["summary"]: r for r in results}
        for bug, score in keyword_hits:
            results_dict[bug["summary"]] = (bug, score)
        results = list(results_dict.values())

    return results


# ---------------- Severity & Priority Heuristics ----------------
def suggest_severity_priority(bug):
    """Heuristic rules to guess severity and priority."""
    text = (bug["summary"] + " " + bug["description"]).lower()

    if any(word in text for word in ["crash", "data loss", "not triggered", "unresponsive", "corruption", "failover"]):
        return "Severity: Critical", "Priority: High"

    if any(word in text for word in ["latency", "delay", "slow", "high cpu", "memory leak", "timeout"]):
        return "Severity: Major", "Priority: Medium"

    if any(word in text for word in ["ui", "format", "glitch", "display", "alignment"]):
        return "Severity: Minor", "Priority: Low"

    return "Severity: Major", "Priority: Medium"  # default fallback


# ---------------- Main Loop ----------------
def main():
    print("💡 Type a query (or 'exit' to quit):")
    while True:
        query = input("Query: ")
        if query.lower() in ["exit", "quit"]:
            break

        results = search_bugs(query, k=10, keyword_match=True)

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
                print(" Invalid choice.")
                continue
        except ValueError:
            print(" Please enter a valid number.")
            continue

        selected = results[choice][0]

        print("\n📌 Full Defect Description:")
        print(selected["description"])

        severity, priority = suggest_severity_priority(selected)
        print(severity)
        print(priority)
        print("-" * 80)


if __name__ == "__main__":
    main()
