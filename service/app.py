import os, json, tomli
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv


from .schemas import EnhanceRequest, EnhanceResponse
from .rag import Retriever
from .llm import render_prompt, generate
from .jira_api import update_issue_description


load_dotenv()


with open("config.toml", "rb") as f:
	CFG = tomli.load(f)


retriever = Retriever(CFG)
app = FastAPI(title="Defect Enhancer POC")




@app.post("/enhance", response_model=EnhanceResponse)
def enhance(req: EnhanceRequest):
	ctx = retriever.search(req.vague_text, k=req.top_k)
	prompt = render_prompt(
		CFG["prompt"]["template_path"],
		vague_text=req.vague_text,
		project_key=req.project_key,
		context_docs=ctx
	)
	enhanced = generate(prompt)

	updated_key = None
	if req.update_jira and req.issue_key:
		try:
			update_issue_description(req.issue_key, enhanced)
			updated_key = req.issue_key
		except Exception as e:
			enhanced += f"\n\n[Note: Failed to update Jira: {e}]"

	return EnhanceResponse(enhanced=enhanced, context=ctx, updated_issue_key=updated_key)