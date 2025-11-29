from abc import ABC, abstractmethod
from typing import List, Dict, Optional

class ChatMemoryInterface(ABC):
    """
    Abstract interface for chat memory implementations.
    Follows Interface Segregation Principle (ISP) and Dependency Inversion Principle (DIP).
    """
    
    @abstractmethod
    async def add_message(
        self, 
        session_id: str, 
        role: str, 
        content: str, 
        project_id: int
    ) -> bool:
        """
        Add a message to the conversation history.
        
        Args:
            session_id: Unique identifier for the conversation session
            role: Message role ('user', 'assistant', 'system')
            content: Message content
            project_id: Project identifier
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_messages(
        self, 
        session_id: str, 
        project_id: int,
        limit: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """
        Retrieve conversation history for a session.
        
        Args:
            session_id: Session identifier
            project_id: Project identifier
            limit: Maximum number of recent messages to retrieve
            
        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        pass
    
    @abstractmethod
    async def clear_session(self, session_id: str, project_id: int) -> bool:
        """
        Clear all messages for a specific session.
        
        Args:
            session_id: Session identifier
            project_id: Project identifier
            
        Returns:
            bool: True if successful
        """
        pass
    
    @abstractmethod
    async def get_recent_sessions(
        self, 
        project_id: int, 
        limit: int = 10
    ) -> List[str]:
        """
        Get list of recent session IDs for a project.
        
        Args:
            project_id: Project identifier
            limit: Maximum number of sessions to return
            
        Returns:
            List of session IDs
        """
        pass