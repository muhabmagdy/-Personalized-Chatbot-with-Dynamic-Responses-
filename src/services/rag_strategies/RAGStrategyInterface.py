from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
from models.db_schemes import Project, RetrievedDocument

class RAGStrategyInterface(ABC):
    """
    Abstract Strategy Pattern for RAG implementations.
    
    Each concrete strategy implements a different retrieval approach while
    maintaining a consistent interface. This follows:
    - Open/Closed Principle: Open for extension, closed for modification
    - Strategy Pattern: Encapsulate algorithms and make them interchangeable
    - Dependency Inversion: Depend on abstractions, not concretions
    """
    
    @abstractmethod
    async def retrieve_documents(
        self, 
        query: str, 
        project: Project,
        limit: int
    ) -> List[RetrievedDocument]:
        """
        Retrieve relevant documents using the strategy's approach.
        
        Args:
            query: User's question
            project: Project entity
            limit: Maximum number of documents to retrieve
            
        Returns:
            List of retrieved documents with scores
        """
        pass
    
    @abstractmethod
    async def generate_answer(
        self,
        query: str,
        retrieved_documents: List[RetrievedDocument],
        chat_history: List[Dict[str, str]]
    ) -> Tuple[Optional[str], str]:
        """
        Generate answer from retrieved documents and chat history.
        
        Args:
            query: User's question
            retrieved_documents: Documents retrieved by the strategy
            chat_history: Conversation history
            
        Returns:
            Tuple of (answer, full_prompt_used)
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return the name of this strategy for logging/debugging."""
        pass