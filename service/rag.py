import os, json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class Retriever:
	def __init__(self, cfg):
		self.model = SentenceTransformer(cfg.get("index", {}).get("embed_model", "sentence-transformers/all-MiniLM-L6-v2"))
		self.index = faiss.read_index(cfg["index"]["faiss_path"]) if os.path.exists(cfg["index"]["faiss_path"]) else None
		with open("index/meta.json", "r", encoding="utf-8") as f:
			self.meta = json.load(f)

	def search(self, query: str, k: int = 5):
		if self.index is None:
			return []
		q = self.model.encode([query], normalize_embeddings=True)
		D, I = self.index.search(q, k)
		results = []
		for score, idx in zip(D[0].tolist(), I[0].tolist()):
			if idx == -1:
				continue
			m = self.meta[idx]
			m["score"] = float(score)
			results.append(m)
		return results