# routes/schemes/nlp.py

from pydantic import BaseModel, Field
from typing import Optional
from models.enums.RAGTypeEnum import RAGTypeEnum

class PushProjectRequest(BaseModel):
    """Request model for project indexing."""
    do_reset: bool = Field(
        default=False,
        description="Whether to reset existing collection before indexing"
    )

class PushAssetRequest(BaseModel):
    """Request model for asset indexing."""
    asset_id: int = Field(
        ...,
        description="Asset ID to index"
    )
    do_reset: bool = Field(
        default=False,
        description="Whether to reset existing vectors for this asset"
    )

class SearchRequest(BaseModel):
    """Request model for vector search."""
    text: str = Field(
        ...,
        min_length=1,
        description="Query text for search"
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of results to return"
    )

class AnswerRAGRequest(BaseModel):
    """
    Extended request model for RAG answer endpoint.
    
    NEW: Supports chat memory and multiple RAG strategies.
    """
    text: str = Field(
        ...,
        min_length=1,
        description="User's question"
    )
    
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of documents to retrieve"
    )
    
    session_id: Optional[str] = Field(
        default=None,
        description="Chat session ID for conversation history. Auto-generated if not provided."
    )
    
    rag_type: str = Field(
        default=RAGTypeEnum.BASIC.value,
        description="RAG strategy to use: 'basic', 'fusion', or 'rerank'"
    )
    
    chat_history_limit: int = Field(
        default=10,
        ge=0,
        le=50,
        description="Maximum number of previous messages to include in context"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "text": "What are the main features of Python?",
                "limit": 10,
                "session_id": "user-123-conversation-1",
                "rag_type": "basic",
                "chat_history_limit": 10
            }
        }