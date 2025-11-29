from .ChatMemoryInterface import ChatMemoryInterface
from typing import List, Dict, Optional
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, delete, desc
from models.db_schemes.myrag.schemes.chat_message import ChatMessage
import logging

class DatabaseChatMemory(ChatMemoryInterface):
    """
    PostgreSQL-backed chat memory implementation.
    
    Advantages:
    - Persistent across restarts
    - Multi-user support
    - Queryable history
    - Production-ready
    
    This is the RECOMMENDED implementation for production use.
    """
    
    def __init__(self, db_client: sessionmaker):
        """
        Initialize with database client.
        
        Args:
            db_client: SQLAlchemy AsyncSession maker
        """
        self.db_client = db_client
        self.logger = logging.getLogger("uvicorn")
    
    async def add_message(
        self, 
        session_id: str, 
        role: str, 
        content: str, 
        project_id: int
    ) -> bool:
        """Add a message to database."""
        try:
            async with self.db_client() as session:
                message = ChatMessage(
                    session_id=session_id,
                    role=role,
                    content=content,
                    project_id=project_id
                )
                session.add(message)
                await session.commit()
                
                self.logger.debug(
                    f"Added message to session {session_id}: {role}"
                )
                return True
                
        except Exception as e:
            self.logger.error(f"Error adding message: {e}")
            return False
    
    async def get_messages(
        self, 
        session_id: str, 
        project_id: int,
        limit: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """
        Retrieve conversation history from database.
        Returns messages in chronological order (oldest first).
        """
        try:
            async with self.db_client() as session:
                # Build query
                query = select(ChatMessage).where(
                    ChatMessage.session_id == session_id,
                    ChatMessage.project_id == project_id
                ).order_by(ChatMessage.created_at.asc())
                
                # Apply limit if specified (get most recent N messages)
                if limit:
                    # Get total count first
                    result = await session.execute(query)
                    all_messages = result.scalars().all()
                    
                    # Take last N messages
                    messages = all_messages[-limit:] if len(all_messages) > limit else all_messages
                else:
                    result = await session.execute(query)
                    messages = result.scalars().all()
                
                # Convert to format expected by LLM
                return [
                    {
                        "role": msg.role,
                        "content": msg.content
                    }
                    for msg in messages
                ]
                
        except Exception as e:
            self.logger.error(f"Error retrieving messages: {e}")
            return []
    
    async def clear_session(self, session_id: str, project_id: int) -> bool:
        """Clear all messages for a session."""
        try:
            async with self.db_client() as session:
                stmt = delete(ChatMessage).where(
                    ChatMessage.session_id == session_id,
                    ChatMessage.project_id == project_id
                )
                await session.execute(stmt)
                await session.commit()
                
                self.logger.info(
                    f"Cleared session {session_id} for project {project_id}"
                )
                return True
                
        except Exception as e:
            self.logger.error(f"Error clearing session: {e}")
            return False
    
    async def get_recent_sessions(
        self, 
        project_id: int, 
        limit: int = 10
    ) -> List[str]:
        """Get list of recent session IDs."""
        try:
            async with self.db_client() as session:
                # Get distinct session IDs ordered by most recent message
                query = select(ChatMessage.session_id).where(
                    ChatMessage.project_id == project_id
                ).distinct().order_by(
                    desc(ChatMessage.created_at)
                ).limit(limit)
                
                result = await session.execute(query)
                session_ids = result.scalars().all()
                
                return list(session_ids)
                
        except Exception as e:
            self.logger.error(f"Error getting recent sessions: {e}")
            return []