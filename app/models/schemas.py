from pydantic import BaseModel
from typing import Optional, List

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5    

class QueryResponse(BaseModel):
    answer: str
    context: Optional[List[str]] = None