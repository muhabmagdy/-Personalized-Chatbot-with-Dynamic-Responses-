# project/src/services/answer_generation/AnswerGenerationService.py

import time
import uuid
import typing
from typing import List, Optional, Dict
import logging

from services.interfaces.AnswerGenerationServiceInterface import AnswerGenerationServiceInterface
from services.rag_strategies.RAGStrategyFactory import RAGStrategyFactory
from services.chat_memory.ChatMemoryInterface import ChatMemoryInterface
from models.db_schemes import Project
from models.enums.RAGTypeEnum import RAGTypeEnum


class AnswerGenerationService(AnswerGenerationServiceInterface):
    """
    Answer Generation Service - Pure RAG Q&A without evaluation.
    
    Responsibilities:
    - Generate answers using RAG strategies
    - Manage chat memory
    - Handle document retrieval
    
    DOES NOT handle:
    - Evaluation (separate service)
    - MLflow tracking (separate service)
    
    SOLID Principles:
    - Single Responsibility: Only generates answers
    - Open/Closed: Extensible with new RAG strategies
    - Liskov Substitution: Implements AnswerGenerationServiceInterface
    - Dependency Inversion: Depends on abstractions
    """
    
    def __init__(
        self,
        vectordb_client,
        generation_client,
        embedding_client,
        template_parser,
        chat_memory: ChatMemoryInterface
    ):
        """
        Initialize answer generation service.
        
        Args:
            vectordb_client: Vector database client
            generation_client: LLM client for generation
            embedding_client: Embedding client
            template_parser: Template parser
            chat_memory: Chat memory implementation
        """
        self.vectordb_client = vectordb_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.template_parser = template_parser
        self.chat_memory = chat_memory
        self.logger = logging.getLogger("uvicorn")
        
        # Initialize RAG strategy factory
        self.rag_factory = RAGStrategyFactory(
            vectordb_client=vectordb_client,
            generation_client=generation_client,
            embedding_client=embedding_client,
            template_parser=template_parser
        )
        
        self.logger.info("Answer generation service initialized")
    
    async def generate_answer(
        self,
        project: Project,
        query: str,
        session_id: str,
        limit: int = 10,
        rag_type: str = RAGTypeEnum.BASIC.value,
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
            Dictionary with:
                - answer: Generated answer
                - session_id: Session identifier
                - strategy_name: RAG strategy used
                - retrieved_contexts: List of retrieved document texts
                - retrieved_documents_count: Number of documents retrieved
                - chat_history: Conversation history
                - full_prompt: Complete prompt used
                - execution_time_ms: Time taken to generate answer
        """
        start_time = time.time()
        project_id = typing.cast(int, project.project_id)
        
        self.logger.info(
            f"Generating answer for project {project_id}, "
            f"session {session_id}, strategy {rag_type}"
        )
        
        # Step 1: Retrieve chat history
        chat_history = await self.chat_memory.get_messages(
            session_id=session_id,
            project_id=project_id,
            limit=chat_history_limit
        )
        
        self.logger.debug(f"Retrieved {len(chat_history)} messages from chat history")
        
        # Step 2: Create RAG strategy
        strategy = self.rag_factory.create_strategy(rag_type=rag_type)
        
        if not strategy:
            self.logger.error(f"Failed to create RAG strategy: {rag_type}")
            raise ValueError(f"Unsupported RAG type: {rag_type}")
        
        strategy_name = strategy.get_strategy_name()
        self.logger.info(f"Using RAG strategy: {strategy_name}")
        
        # Step 3: Retrieve documents
        retrieved_documents = await strategy.retrieve_documents(
            query=query,
            project=project,
            limit=limit
        )
        
        if not retrieved_documents or len(retrieved_documents) == 0:
            self.logger.warning("No documents retrieved")
            raise ValueError("No relevant documents found for query")
        
        self.logger.debug(f"Retrieved {len(retrieved_documents)} documents")
        
        # Extract context texts
        retrieved_contexts = [doc.text for doc in retrieved_documents]
        
        # Step 4: Generate answer
        answer, full_prompt = await strategy.generate_answer(
            query=query,
            retrieved_documents=retrieved_documents,
            chat_history=chat_history
        )
        
        if not answer:
            self.logger.error("Failed to generate answer")
            raise ValueError("Failed to generate answer")
        
        # Step 5: Save conversation to chat memory
        await self._save_conversation_to_memory(
            session_id=session_id,
            project_id=project_id,
            user_query=query,
            assistant_answer=answer
        )
        
        # Calculate execution time
        execution_time_ms = (time.time() - start_time) * 1000
        
        self.logger.info(
            f"Answer generated successfully: {len(answer)} chars, "
            f"{len(retrieved_documents)} docs, {execution_time_ms:.0f}ms"
        )
        
        return {
            "answer": answer,
            "session_id": session_id,
            "strategy_name": strategy_name,
            "rag_type": rag_type,
            "retrieved_contexts": retrieved_contexts,
            "retrieved_documents_count": len(retrieved_documents),
            "chat_history": chat_history,
            "chat_history_length": len(chat_history),
            "full_prompt": full_prompt,
            "execution_time_ms": execution_time_ms
        }
    
    async def get_retrieved_contexts(
        self,
        project: Project,
        query: str,
        limit: int = 10,
        rag_type: str = RAGTypeEnum.BASIC.value
    ) -> List[str]:
        """
        Get retrieved document contexts without generating answer.
        
        Useful for evaluation preparation or inspection.
        
        Args:
            project: Project entity
            query: User's question
            limit: Number of documents to retrieve
            rag_type: RAG strategy type
            
        Returns:
            List of retrieved document texts
        """
        self.logger.info(
            f"Retrieving contexts for project {project.project_id}, "
            f"strategy {rag_type}"
        )
        
        # Create RAG strategy
        strategy = self.rag_factory.create_strategy(rag_type=rag_type)
        
        if not strategy:
            raise ValueError(f"Unsupported RAG type: {rag_type}")
        
        # Retrieve documents
        retrieved_documents = await strategy.retrieve_documents(
            query=query,
            project=project,
            limit=limit
        )
        
        # Extract texts
        contexts = [doc.text for doc in retrieved_documents]
        
        self.logger.info(f"Retrieved {len(contexts)} contexts")
        
        return contexts
    
    async def _save_conversation_to_memory(
        self,
        session_id: str,
        project_id: int,
        user_query: str,
        assistant_answer: str
    ) -> None:
        """
        Save user query and assistant answer to chat memory.
        
        Args:
            session_id: Session identifier
            project_id: Project identifier
            user_query: User's question
            assistant_answer: Generated answer
        """
        try:
            # Save user message
            await self.chat_memory.add_message(
                session_id=session_id,
                role="user",
                content=user_query,
                project_id=project_id
            )
            
            # Save assistant message
            await self.chat_memory.add_message(
                session_id=session_id,
                role="assistant",
                content=assistant_answer,
                project_id=project_id
            )
            
            self.logger.debug(f"Saved conversation to memory (session: {session_id})")
            
        except Exception as e:
            self.logger.error(f"Failed to save conversation to memory: {e}")
            # Don't fail the entire request if memory save fails
    
    async def clear_chat_session(
        self,
        session_id: str,
        project_id: int
    ) -> bool:
        """
        Clear chat history for a session.
        
        Args:
            session_id: Session identifier
            project_id: Project identifier
            
        Returns:
            True if successful
        """
        try:
            success = await self.chat_memory.clear_session(
                session_id=session_id,
                project_id=project_id
            )
            
            if success:
                self.logger.info(f"Cleared chat session: {session_id}")
            else:
                self.logger.warning(f"Failed to clear session: {session_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error clearing session: {e}")
            return False
    
    async def get_chat_sessions(
        self,
        project_id: int,
        limit: int = 10
    ) -> List[str]:
        """
        Get list of recent chat sessions for a project.
        
        Args:
            project_id: Project identifier
            limit: Maximum number of sessions to return
            
        Returns:
            List of session IDs
        """
        try:
            sessions = await self.chat_memory.get_recent_sessions(
                project_id=project_id,
                limit=limit
            )
            
            self.logger.info(
                f"Retrieved {len(sessions)} sessions for project {project_id}"
            )
            
            return sessions
            
        except Exception as e:
            self.logger.error(f"Error retrieving sessions: {e}")
            return []
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available RAG strategies."""
        return self.rag_factory.get_available_strategies()
    
    def get_strategy_info(self, rag_type: str) -> Dict:
        """Get detailed information about a RAG strategy."""
        return self.rag_factory.get_strategy_info(rag_type)