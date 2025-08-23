import os
import json
import argparse
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer




def iter_jsonl(path):
	with open(path, "r", encoding="utf-8") as f:
		for line in f:
			if line.strip():
				yield json.loads(line)




if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("--input", required=True)
	ap.add_argument("--outdir", required=True)
	ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
	args = ap.parse_args()

	os.makedirs(args.outdir, exist_ok=True)

	model = SentenceTransformer(args.model)
	texts, metas = [], []
	for rec in iter_jsonl(args.input):
		text = f"{rec.get('summary','')}\n\n{rec.get('description','')}"
		texts.append(text)
		metas.append({"key": rec.get("key"), "summary": rec.get("summary"), "description": rec.get("description")})

	embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=True, batch_size=64, normalize_embeddings=True)
	dim = embs.shape[1]
	index = faiss.IndexFlatIP(dim)
	index.add(embs)

	faiss.write_index(index, os.path.join(args.outdir, "faiss_index.bin"))
	np.save(os.path.join(args.outdir, "embeddings.npy"), embs)
	with open(os.path.join(args.outdir, "meta.json"), "w", encoding="utf-8") as f:
		json.dump(metas, f, ensure_ascii=False)

	print(f"Indexed {len(texts)} documents.")