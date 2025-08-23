import os
# return t.render(vague_text=vague_text, project_key=project_key, context_docs=context_docs)


# ---------- Providers ----------


def gen_openai(prompt: str):
	import requests, json
	base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
	model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
	headers = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}", "Content-Type": "application/json"}
	payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.2}
	r = requests.post(f"{base}/chat/completions", headers=headers, json=payload)
	r.raise_for_status()
	return r.json()["choices"][0]["message"]["content"].strip()




def gen_ollama(prompt: str):
	import requests
	base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
	model = os.getenv("OLLAMA_MODEL", "llama3")
	r = requests.post(f"{base}/api/generate", json={"model": model, "prompt": prompt, "options": {"temperature": 0.2}})
	r.raise_for_status()
	txt = ""
	for line in r.iter_lines():
		if not line:
			continue
		try:
			part = line.decode("utf-8")
		except Exception:
			part = line
		txt += part
	# Ollama returns JSONL; a simpler approach is to call without stream, but many builds stream by default
	# For POC simplicity, call non-stream:
	r = requests.post(f"{base}/api/generate", json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.2}})
	r.raise_for_status()
	return r.json().get("response", "").strip()




def gen_huggingface_t5(prompt: str):
	from transformers import T5ForConditionalGeneration, T5TokenizerFast
	path = os.getenv("HF_MODEL_PATH", "models/t5-lora-domain")
	tok = T5TokenizerFast.from_pretrained(path)
	mdl = T5ForConditionalGeneration.from_pretrained(path)
	ids = tok(prompt, return_tensors="pt").input_ids
	out = mdl.generate(ids, max_new_tokens=400)
	return tok.decode(out[0], skip_special_tokens=True).strip()




def generate(prompt: str) -> str:
	if PROVIDER == "openai":
		return gen_openai(prompt)
	elif PROVIDER == "huggingface":
		return gen_huggingface_t5(prompt)
	else:
		return gen_ollama(prompt)