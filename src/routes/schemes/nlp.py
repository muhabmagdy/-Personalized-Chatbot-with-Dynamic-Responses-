from pydantic import BaseModel
from typing import Optional

class PushProjectRequest(BaseModel):
    do_reset: Optional[int] = 0
    
class PushAssetRequest(BaseModel):
    do_reset: Optional[int] = 0
    asset_id: int = 0

class SearchRequest(BaseModel):
    text: str
    limit: Optional[int] = 5
