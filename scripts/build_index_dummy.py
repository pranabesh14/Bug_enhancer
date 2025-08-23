from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np
import os

DATA_PATH = "data/dummy_bugs.jsonl"
INDEX_PATH = "index/faiss_index.bin"
EMB_PATH = "index/embeddings.npy"

def main():
    # Load model (small + CPU friendly)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Load dataset
    bugs = []
    with open(DATA_PATH, "r") as f:
        for line in f:
            bugs.append(json.loads(line))

    # Build FAISS index
    bug_texts = [f"{b['summary']} - {b['description']} ({b['platform']})" for b in bugs]
    embeddings = model.encode(bug_texts, convert_to_numpy=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save index + embeddings
    os.makedirs("index", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    np.save(EMB_PATH, embeddings)

    print(f"FAISS index built with {len(bug_texts)} bug reports")

if __name__ == "__main__":
    main()