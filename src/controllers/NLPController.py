from controllers.BaseController import BaseController
from models.db_schemes import Project
from stores.llm.LLMEnums import DocumentTypeEnum
from typing import List, Optional, Dict, Tuple
from services.rag_strategies.RAGStrategyFactory import RAGStrategyFactory
from services.chat_memory.ChatMemoryInterface import ChatMemoryInterface
from models.enums.RAGTypeEnum import RAGTypeEnum
import json
import typing
import gc
import logging

# Embedding batch size for memory optimization
EMBEDDING_BATCH_SIZE = 500 

class NLPController(BaseController):
    """
    Refactored NLP Controller using Strategy Pattern for RAG.
    
    SOLID Principles Applied:
    - Single Responsibility: Each method has one clear purpose
    - Open/Closed: Open for extension (new RAG strategies), closed for modification
    - Liskov Substitution: All RAG strategies are interchangeable
    - Interface Segregation: Clean interfaces for strategies and chat memory
    - Dependency Inversion: Depends on abstractions (interfaces), not concrete implementations
    """

    def __init__(
        self, 
        vectordb_client, 
        generation_client, 
        embedding_client, 
        template_parser,
        chat_memory: ChatMemoryInterface
    ):
        super().__init__()

        self.vectordb_client = vectordb_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.template_parser = template_parser
        self.chat_memory = chat_memory
        self.logger = logging.getLogger("uvicorn")
        
        # Initialize RAG Strategy Factory
        self.rag_factory = RAGStrategyFactory(
            vectordb_client=vectordb_client,
            generation_client=generation_client,
            embedding_client=embedding_client,
            template_parser=template_parser
        )

    def create_collection_name(self, project_id: int):
        return f"collection_{self.vectordb_client.default_vector_size}_{project_id}".strip()
    
    async def reset_vector_db_collection(self, project: Project):
        collection_name = self.create_collection_name(
            project_id=typing.cast(int, project.project_id)
        )
        return await self.vectordb_client.delete_collection(
            collection_name=collection_name
        )
    
    async def get_vector_db_collection_info(self, project: Project):
        collection_name = self.create_collection_name(
            project_id=typing.cast(int, project.project_id)
        )
        collection_info = await self.vectordb_client.get_collection_info(
            collection_name=collection_name
        )

        return json.loads(
            json.dumps(collection_info, default=lambda x: x.__dict__)
        )
    
    # ==========================================
    # OPTIMIZED INDEXING (UNCHANGED)
    # ==========================================

    async def index_into_vector_db(
        self, 
        project: Project, 
        chunks_data: List[dict],
        collection_name: str
    ):
        """
        Memory-optimized indexing with explicit cleanup.
        """
        num_chunks = len(chunks_data)
        EMBEDDING_BATCH_SIZE = 10
        
        for i in range(0, num_chunks, EMBEDDING_BATCH_SIZE):
            batch_data = chunks_data[i:i + EMBEDDING_BATCH_SIZE]
            
            texts = [c["chunk_text"] for c in batch_data]
            metadata = [c["chunk_metadata"] for c in batch_data]
            chunk_ids = [c["chunk_id"] for c in batch_data]
            
            vectors = self.embedding_client.embed_text(
                text=texts, 
                document_type=DocumentTypeEnum.DOCUMENT.value
            )
            
            _ = await self.vectordb_client.insert_many(
                collection_name=collection_name,
                texts=texts,
                metadata=metadata,
                vectors=vectors,
                record_ids=chunk_ids,
            )
            
            del texts, metadata, chunk_ids, vectors, batch_data
            
            if i % 50 == 0:
                gc.collect()
        
        return True

    async def search_vector_db_collection(
        self, 
        project: Project, 
        text: str, 
        limit: Optional[int] = 10
    ):
        """
        Simple vector search (used for testing/debugging).
        """
        final_limit = limit if limit is not None else 10

        query_vector = None
        collection_name = self.create_collection_name(
            project_id=typing.cast(int, project.project_id)
        )

        vectors = self.embedding_client.embed_text(
            text=text, 
            document_type=DocumentTypeEnum.QUERY.value
        )

        if not vectors or len(vectors) == 0:
            return False
        
        if isinstance(vectors, list) and len(vectors) > 0:
            query_vector = vectors[0]

        if not query_vector:
            return False    

        results = await self.vectordb_client.search_by_vector(
            collection_name=collection_name,
            vector=query_vector,
            limit=final_limit
        )

        if not results:
            return False

        return results
    
    # ==========================================
    # NEW: STRATEGY-BASED RAG WITH CHAT MEMORY
    # ==========================================
    
    async def answer_rag_question(
        self, 
        project: Project, 
        query: str, 
        session_id: str,
        limit: Optional[int] = 10,
        rag_type: str = RAGTypeEnum.BASIC.value,
        chat_history_limit: Optional[int] = 10
    ) -> Tuple[Optional[str], str, List[Dict[str, str]], str]:
        """
        Answer RAG question using specified strategy and chat memory.
        
        This method now:
        1. Retrieves chat history from database
        2. Uses Strategy Pattern to select RAG approach
        3. Saves conversation to chat memory
        4. Returns comprehensive results
        
        Args:
            project: Project entity
            query: User's question
            session_id: Chat session identifier for memory
            limit: Number of documents to retrieve
            rag_type: Type of RAG strategy to use (basic/fusion/rerank)
            chat_history_limit: Maximum chat history messages to include
            
        Returns:
            Tuple of (answer, full_prompt, chat_history, strategy_name)
        """
        final_limit = limit if limit is not None else 10
        project_id = typing.cast(int, project.project_id)
        
        # Step 1: Retrieve chat history from memory
        chat_history = await self.chat_memory.get_messages(
            session_id=session_id,
            project_id=project_id,
            limit=chat_history_limit
        )
        
        self.logger.info(
            f"Retrieved {len(chat_history)} messages from chat history "
            f"(session: {session_id})"
        )
        
        # Step 2: Create RAG strategy
        strategy = self.rag_factory.create_strategy(rag_type=rag_type)
        
        if not strategy:
            self.logger.error(f"Failed to create RAG strategy: {rag_type}")
            return None, "", chat_history, "unknown"
        
        strategy_name = strategy.get_strategy_name()
        self.logger.info(f"Using RAG strategy: {strategy_name}")
        
        # Step 3: Retrieve documents using strategy
        retrieved_documents = await strategy.retrieve_documents(
            query=query,
            project=project,
            limit=final_limit
        )
        
        if not retrieved_documents or len(retrieved_documents) == 0:
            self.logger.warning("No documents retrieved")
            return None, "", chat_history, strategy_name
        
        # Step 4: Generate answer using strategy
        answer, full_prompt = await strategy.generate_answer(
            query=query,
            retrieved_documents=retrieved_documents,
            chat_history=chat_history
        )
        
        if not answer:
            self.logger.error("Failed to generate answer")
            return None, full_prompt, chat_history, strategy_name
        
        # Step 5: Save conversation to chat memory
        await self._save_conversation_to_memory(
            session_id=session_id,
            project_id=project_id,
            user_query=query,
            assistant_answer=answer
        )
        
        self.logger.info(
            f"Successfully answered question using {strategy_name} "
            f"(retrieved {len(retrieved_documents)} docs)"
        )
        
        return answer, full_prompt, chat_history, strategy_name
    
    async def _save_conversation_to_memory(
        self,
        session_id: str,
        project_id: int,
        user_query: str,
        assistant_answer: str
    ) -> None:
        """
        Save user query and assistant answer to chat memory.
        """
        # Save user message
        await self.chat_memory.add_message(
            session_id=session_id,
            role=self.generation_client.enums.USER.value ,
            content=user_query,
            project_id=project_id
        )
        
        # Save assistant response
        await self.chat_memory.add_message(
            session_id=session_id,
            role=self.generation_client.enums.ASSISTANT.value,
            content=assistant_answer,
            project_id=project_id
        )
        
        self.logger.debug(
            f"Saved conversation to memory (session: {session_id})"
        )
    
    async def clear_chat_session(
        self, 
        session_id: str, 
        project_id: int
    ) -> bool:
        """
        Clear chat history for a specific session.
        
        Args:
            session_id: Session identifier
            project_id: Project identifier
            
        Returns:
            bool: True if successful
        """
        return await self.chat_memory.clear_session(
            session_id=session_id,
            project_id=project_id
        )
    
    async def get_chat_sessions(self, project_id: int, limit: int = 10) -> List[str]:
        """
        Get list of recent chat sessions for a project.
        
        Args:
            project_id: Project identifier
            limit: Maximum number of sessions to return
            
        Returns:
            List of session IDs
        """
        return await self.chat_memory.get_recent_sessions(
            project_id=project_id,
            limit=limit
        )