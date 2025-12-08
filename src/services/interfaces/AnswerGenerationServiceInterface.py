# project/src/services/interfaces/AnswerGenerationServiceInterface.py

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Tuple
from models.db_schemes import Project


class AnswerGenerationServiceInterface(ABC):
    """
    Abstract interface for answer generation services.
    
    Separates answer generation from evaluation.
    Follows Single Responsibility Principle.
    """
    
    @abstractmethod
    async def generate_answer(
        self,
        project: Project,
        query: str,
        session_id: str,
        limit: int = 10,
        rag_type: str = "basic",
        chat_history_limit: int = 10
    ) -> Dict:
        """
        Generate answer using specified RAG strategy.
        
        Args:
            project: Project entity
            query: User's question
            session_id: Chat session identifier
            limit: Number of documents to retrieve
            rag_type: RAG strategy type
            chat_history_limit: Max chat history messages
            
        Returns:
            Dictionary with answer and metadata
        """
        pass
    
    @abstractmethod
    async def get_retrieved_contexts(
        self,
        project: Project,
        query: str,
        limit: int = 10,
        rag_type: str = "basic"
    ) -> List[str]:
        """
        Get retrieved document contexts without generating answer.
        
        Useful for evaluation preparation.
        
        Args:
            project: Project entity
            query: User's question
            limit: Number of documents to retrieve
            rag_type: RAG strategy type
            
        Returns:
            List of retrieved document texts
        """
        pass