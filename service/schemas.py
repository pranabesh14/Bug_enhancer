from pydantic import BaseModel
from typing import List, Optional


class EnhanceRequest(BaseModel):
	project_key: str
	vague_text: str
	issue_key: Optional[str] = None
	update_jira: bool = False
	top_k: int = 5


class EnhanceResponse(BaseModel):
	enhanced: str
	context: List[dict]
	updated_issue_key: Optional[str] = None