import os
import json
import argparse
import requests
from dotenv import load_dotenv


load_dotenv()


BASE = os.getenv("JIRA_BASE_URL")
EMAIL = os.getenv("JIRA_EMAIL")
TOKEN = os.getenv("JIRA_API_TOKEN")


HEADERS = {"Accept": "application/json"}
AUTH = (EMAIL, TOKEN)




def fetch_issues(jql: str, limit: int = 500):
	start = 0
	all_issues = []
	while True:
		params = {
			"jql": jql,
			"startAt": start,
			"maxResults": min(100, limit - start),
			"fields": [
				"summary",
				"description",
				"issuetype",
				"created",
				"updated",
				"labels",
				"components",
				"priority",
				"environment"
			]
		}
		r = requests.get(f"{BASE}/rest/api/3/search", headers=HEADERS, params=params, auth=AUTH)
		r.raise_for_status()
		data = r.json()
		issues = data.get("issues", [])
		for it in issues:
			fields = it.get("fields", {})
			rec = {
				"key": it.get("key"),
				"summary": fields.get("summary") or "",
				"description": (fields.get("description") or ""),
				"labels": fields.get("labels") or [],
				"components": [c.get("name") for c in (fields.get("components") or [])],
				"priority": (fields.get("priority") or {}).get("name"),
				"environment": fields.get("environment") or "",
				"issuetype": (fields.get("issuetype") or {}).get("name"),
				"created": fields.get("created"),
				"updated": fields.get("updated"),
			}
			all_issues.append(rec)
		start += len(issues)
		if start >= limit or len(issues) == 0:
			break
	return all_issues




if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("--jql", required=True)
	ap.add_argument("--limit", type=int, default=500)
	ap.add_argument("--out", default="data/raw_jira_export.jsonl")
	args = ap.parse_args()

	issues = fetch_issues(args.jql, args.limit)
	with open(args.out, "w", encoding="utf-8") as f:
		for issue in issues:
			f.write(json.dumps(issue, ensure_ascii=False) + "\n")

	print(f"Saved {len(issues)} issues to {args.out}")