"""
API Client Service
Handles all communication with the RAG backend API
"""

import httpx
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class APIClient:
    """Client for communicating with RAG system API."""
    
    def __init__(self, base_url: str, timeout: float = 30.0):
        """
        Initialize API client.
        
        Args:
            base_url: Base URL of the API (e.g., "http://localhost:8000")
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        logger.info(f"API Client initialized with base URL: {self.base_url}")
    
    async def get_available_strategies(self) -> List[Dict[str, str]]:
        """
        Get list of available RAG strategies.
        
        Returns:
            List of strategy dictionaries with type, name, description
            
        Example:
            [
                {
                    "type": "basic",
                    "name": "Basic RAG",
                    "description": "Fast, simple retrieval"
                },
                ...
            ]
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.base_url}/api/v1/nlp/rag/strategies"
                )
                response.raise_for_status()
                data = response.json()
                
                strategies = data.get("strategies", [])
                logger.info(f"Fetched {len(strategies)} strategies from API")
                return strategies
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching strategies: {e.response.status_code}")
            return self._get_default_strategies()
        except httpx.RequestError as e:
            logger.error(f"Request error fetching strategies: {e}")
            return self._get_default_strategies()
        except Exception as e:
            logger.error(f"Unexpected error fetching strategies: {e}")
            return self._get_default_strategies()
    
    async def send_message(
        self,
        project_id: int,
        text: str,
        session_id: str,
        rag_type: str = "basic",
        limit: int = 10,
        chat_history_limit: int = 10
    ) -> Dict[str, Any]:
        """
        Send a message and get RAG-based response.
        
        Args:
            project_id: Project identifier
            text: User's message
            session_id: Chat session ID
            rag_type: RAG strategy to use (basic/fusion/rerank)
            limit: Number of documents to retrieve
            chat_history_limit: Max chat history messages to include
            
        Returns:
            Response dictionary with answer, strategy, session info, etc.
            
        Raises:
            APIError: If request fails
            
        Example Response:
            {
                "signal": "RAG_ANSWER_SUCCESS",
                "answer": "Python is a programming language...",
                "session_id": "abc-123",
                "rag_strategy": "Basic RAG",
                "rag_type": "basic",
                "chat_history_length": 2
            }
        """
        try:
            payload = {
                "text": text,
                "session_id": session_id,
                "rag_type": rag_type,
                "limit": limit,
                "chat_history_limit": chat_history_limit
            }
            
            logger.info(f"Sending message to API: project={project_id}, rag_type={rag_type}")
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/v1/nlp/index/answer/{project_id}",
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
                
                logger.info(f"Received response: {data.get('signal', 'Unknown')}")
                return data
                
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}: {e.response.text}"
            logger.error(f"HTTP error: {error_msg}")
            raise APIError(f"Request failed with status {e.response.status_code}")
        except httpx.RequestError as e:
            logger.error(f"Request error: {e}")
            raise APIError(f"Connection error: Cannot reach backend at {self.base_url}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise APIError(f"Unexpected error: {str(e)}")
    
    async def get_chat_sessions(
        self,
        project_id: int,
        limit: int = 10
    ) -> List[str]:
        """
        Get list of recent chat sessions.
        
        Args:
            project_id: Project identifier
            limit: Maximum sessions to return
            
        Returns:
            List of session IDs
            
        Example:
            ["session-abc-123", "session-def-456", ...]
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.base_url}/api/v1/nlp/chat/sessions/{project_id}",
                    params={"limit": limit}
                )
                response.raise_for_status()
                data = response.json()
                
                sessions = data.get("sessions", [])
                logger.info(f"Fetched {len(sessions)} sessions for project {project_id}")
                return sessions
                
        except Exception as e:
            logger.error(f"Error fetching sessions: {e}")
            return []
    
    async def clear_chat_session(
        self,
        project_id: int,
        session_id: str
    ) -> bool:
        """
        Clear a chat session history.
        
        Args:
            project_id: Project identifier
            session_id: Session to clear
            
        Returns:
            True if successful, False otherwise
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.delete(
                    f"{self.base_url}/api/v1/nlp/chat/session/{project_id}/{session_id}"
                )
                response.raise_for_status()
                
                logger.info(f"Cleared session {session_id} for project {project_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error clearing session: {e}")
            return False
    
    async def health_check(self) -> bool:
        """
        Check if the API is reachable.
        
        Returns:
            True if API is healthy, False otherwise
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/welcome")
                return response.status_code == 200
        except Exception:
            return False
    
    def _get_default_strategies(self) -> List[Dict[str, str]]:
        """
        Fallback strategies if API is unavailable.
        
        Returns:
            List of default strategy configurations
        """
        logger.warning("Using default strategies (API unavailable)")
        return [
            {
                "type": "basic",
                "name": "Basic RAG",
                "description": "Fast, simple retrieval - Best for straightforward questions"
            },
            {
                "type": "fusion",
                "name": "Fusion RAG",
                "description": "Query expansion with multiple searches - Better accuracy"
            },
            {
                "type": "rerank",
                "name": "Gemini ReRank",
                "description": "Two-stage retrieval with reranking - Highest precision (FREE)"
            }
        ]


class APIError(Exception):
    """Custom exception for API errors."""
    pass