# test_app.py
import faiss
import json
from sentence_transformers import SentenceTransformer

# -------------------
# CONFIG
# -------------------
BUGS_FILE = "data/dummy_bugs.jsonl"   # your bug dataset
FAISS_DIM = 384                  # embedding dimension (all-MiniLM-L6-v2 = 384)
TOP_K = 3                        # how many results to return
MODEL_NAME = "all-MiniLM-L6-v2"  # same model used during indexing

# -------------------
# Load metadata
# -------------------
bugs = []
with open(BUGS_FILE, "r") as f:
    for line in f:
        bugs.append(json.loads(line))

# -------------------
# Load model & FAISS index
# -------------------
print(" Loading model & FAISS index...")
model = SentenceTransformer(MODEL_NAME)

# Rebuild index (for now we rebuild each time — later you can persist to disk)
bug_texts = [f"{b['summary']} - {b['description']} ({b['platform']})" for b in bugs]
embeddings = model.encode(bug_texts, convert_to_numpy=True)

index = faiss.IndexFlatL2(FAISS_DIM)
index.add(embeddings)
print(f"FAISS index built with {len(bugs)} bug reports")

# -------------------
# Interactive loop
# -------------------
print("\nType a query (or 'exit' to quit):")
while True:
    query = input("Query: ")
    if query.lower() in ["exit", "quit"]:
        break

    query_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_emb, TOP_K)

    print("\n🔎 Top matches:")
    for rank, idx in enumerate(I[0]):
        bug = bugs[idx]
        print(f"{rank+1}. {bug['summary']} ({bug['platform']})")
        print(f"   {bug['description']}")
        print(f"   Score: {D[0][rank]:.4f}\n")
