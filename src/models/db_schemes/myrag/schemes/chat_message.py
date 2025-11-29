from .db_schema_base import SQLAlchemyBase
from sqlalchemy import Column, Integer, DateTime, func, String, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy import Index
import uuid

class ChatMessage(SQLAlchemyBase):
    """
    Stores chat messages for conversation history.
    Each message is linked to a project and has a session_id for grouping conversations.
    """
    __tablename__ = "chat_messages"

    message_id = Column(Integer, primary_key=True, autoincrement=True)
    message_uuid = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True, nullable=False)

    # Session identifier for grouping related messages in a conversation
    session_id = Column(String(255), nullable=False, index=True)
    
    # Message role: 'user', 'assistant', 'system'
    role = Column(String(50), nullable=False)
    
    # Message content (using Text for large messages)
    content = Column(Text, nullable=False)
    
    # Link to project
    project_id = Column(Integer, ForeignKey("projects.project_id"), nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), nullable=True)

    # Relationships
    project = relationship("Project", backref="chat_messages")

    __table_args__ = (
        Index('ix_chat_session_project', session_id, project_id),
        Index('ix_chat_created_at', created_at),
    )