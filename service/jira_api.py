import os
import requests
from dotenv import load_dotenv


load_dotenv()


BASE = os.getenv("JIRA_BASE_URL")
EMAIL = os.getenv("JIRA_EMAIL")
TOKEN = os.getenv("JIRA_API_TOKEN")


HEADERS = {"Accept": "application/json", "Content-Type": "application/json"}
AUTH = (EMAIL, TOKEN)




def update_issue_description(issue_key: str, description: str):
	url = f"{BASE}/rest/api/3/issue/{issue_key}"
	payload = {"fields": {"description": description}}
	r = requests.put(url, headers=HEADERS, auth=AUTH, json=payload)
	r.raise_for_status()
	return True