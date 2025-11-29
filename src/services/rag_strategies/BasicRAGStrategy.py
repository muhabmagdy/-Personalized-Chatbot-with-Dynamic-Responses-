from .RAGStrategyInterface import RAGStrategyInterface
from typing import List, Dict, Optional, Tuple
from models.db_schemes import Project, RetrievedDocument
from stores.llm.LLMEnums import DocumentTypeEnum
import typing
import logging

class BasicRAGStrategy(RAGStrategyInterface):
    """
    Basic RAG Strategy: Single query → Vector search → Generate answer
    
    This is the standard RAG approach:
    1. Embed user query
    2. Search vector database for similar chunks
    3. Construct prompt with retrieved context
    4. Generate answer with LLM
    
    Best for: Simple Q&A, straightforward queries
    """
    
    def __init__(
        self, 
        vectordb_client,
        generation_client,
        embedding_client,
        template_parser
    ):
        """
        Initialize with required dependencies (Dependency Injection).
        
        Args:
            vectordb_client: Vector database client
            generation_client: LLM client for generation
            embedding_client: Embedding model client
            template_parser: Template parser for prompts
        """
        self.vectordb_client = vectordb_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.template_parser = template_parser
        self.logger = logging.getLogger("uvicorn")
    
    def get_strategy_name(self) -> str:
        """Return strategy name."""
        return "Basic RAG"
    
    async def retrieve_documents(
        self, 
        query: str, 
        project: Project,
        limit: int
    ) -> List[RetrievedDocument]:
        """
        Retrieve documents using simple vector similarity search.
        """
        # Step 1: Create collection name
        collection_name = self._create_collection_name(
            project_id=typing.cast(int, project.project_id)
        )
        
        # Step 2: Embed the query
        query_vectors = self.embedding_client.embed_text(
            text=query,
            document_type=DocumentTypeEnum.QUERY.value
        )
        
        if not query_vectors or len(query_vectors) == 0:
            self.logger.error("Failed to embed query")
            return []
        
        query_vector = query_vectors[0]
        
        # Step 3: Search vector database
        results = await self.vectordb_client.search_by_vector(
            collection_name=collection_name,
            vector=query_vector,
            limit=limit
        )
        
        if not results:
            self.logger.warning(f"No documents found for query: {query[:50]}...")
            return []
        
        self.logger.info(
            f"Retrieved {len(results)} documents using {self.get_strategy_name()}"
        )
        
        return results
    
    async def generate_answer(
        self,
        query: str,
        retrieved_documents: List[RetrievedDocument],
        chat_history: List[Dict[str, str]]
    ) -> Tuple[Optional[str], str]:
        """
        Generate answer using retrieved documents and chat history.
        """
        if not retrieved_documents or len(retrieved_documents) == 0:
            return None, ""
        
        # Step 1: Construct system prompt
        system_prompt = self.template_parser.get("rag", "system_prompt")
        
        # Step 2: Construct document context
        documents_prompts = "\n".join([
            self.template_parser.get("rag", "document_prompt", {
                "doc_num": idx + 1,
                "chunk_text": self.generation_client.process_text(doc.text),
            })
            for idx, doc in enumerate(retrieved_documents)
        ])
        
        # Step 3: Construct footer with query
        footer_prompt = self.template_parser.get("rag", "footer_prompt", {
            "query": query
        })
        
        # Step 4: Build full prompt
        full_prompt = "\n\n".join([documents_prompts, footer_prompt])
        
        # Step 5: Prepare chat history with system prompt
        final_chat_history = [
            self.generation_client.construct_prompt(
                prompt=system_prompt,
                role=self.generation_client.enums.SYSTEM.value,
            )
        ]
        
        # Add existing chat history (if any)
        final_chat_history.extend(chat_history)
        
        # Step 6: Generate answer
        answer = self.generation_client.generate_text(
            prompt=full_prompt,
            chat_history=final_chat_history
        )
        
        return answer, full_prompt
    
    def _create_collection_name(self, project_id: int) -> str:
        """Helper to create collection name."""
        return f"collection_{self.vectordb_client.default_vector_size}_{project_id}".strip()