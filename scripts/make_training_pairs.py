import json
import argparse


"""
Build supervised pairs from historical tickets. Heuristic:
- input = short/vague summary
- target = cleaned structured description synthesized from description + priority + environment
For better results, curate manually or use a templater.
"""


def iter_jsonl(path):
	with open(path, "r", encoding="utf-8") as f:
		for line in f:
			if line.strip():
				yield json.loads(line)




def to_structured(rec):
	desc = rec.get("description") or ""
	env = rec.get("environment") or ""
	priority = rec.get("priority") or "Unspecified"
	components = ", ".join(rec.get("components") or [])

	template = f"""
**Summary**: {rec.get('summary','')}


**Steps to Reproduce**:
1. [From historical context; may be incomplete]


**Expected Result**:
- [Fill based on module: {components or 'N/A'}]


**Actual Result**:
- {desc[:800]}


**Environment**:
- {env or 'Unknown'}


**Impact/Severity**:
- {priority}
"""
	return template.strip()




if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("--input", required=True)
	ap.add_argument("--output", required=True)
	ap.add_argument("--eval", required=True)
	args = ap.parse_args()

	train, eval_ = [], []
	for i, rec in enumerate(iter_jsonl(args.input)):
		pair = {
			"input": rec.get("summary") or rec.get("key"),
			"target": to_structured(rec)
		}
		(eval_ if i % 10 == 0 else train).append(pair)

	with open(args.output, "w", encoding="utf-8") as f:
		for p in train:
			f.write(json.dumps(p, ensure_ascii=False) + "\n")
	with open(args.eval, "w", encoding="utf-8") as f:
		for p in eval_:
			f.write(json.dumps(p, ensure_ascii=False) + "\n")
	print(f"Train: {len(train)}, Eval: {len(eval_)}")