from .RAGStrategyInterface import RAGStrategyInterface
from typing import List, Dict, Optional, Tuple
from models.db_schemes import Project, RetrievedDocument
from stores.llm.LLMEnums import DocumentTypeEnum
import typing
import logging

class ReRankRAGStrategy(RAGStrategyInterface):
    """
    ReRank RAG Strategy: Retrieve many → Cross-encoder reranking → Generate
    
    Two-stage retrieval approach:
    1. First stage: Fast bi-encoder retrieval (many candidates)
    2. Second stage: Precise cross-encoder reranking (top candidates)
    3. Generate answer with best-reranked documents
    
    Best for: High precision requirements, complex documents
    
    Note: This implementation uses LLM-based reranking as a fallback.
    For production, consider using a dedicated cross-encoder model like:
    - sentence-transformers/ms-marco-MiniLM-L-12-v2
    - BAAI/bge-reranker-large
    """
    
    def __init__(
        self, 
        vectordb_client,
        generation_client,
        embedding_client,
        template_parser,
        initial_retrieve_multiplier: int = 3
    ):
        """
        Initialize ReRank RAG strategy.
        
        Args:
            initial_retrieve_multiplier: Retrieve N×limit documents for reranking
                                        (default: 3, so retrieve 30 for top 10)
        """
        self.vectordb_client = vectordb_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.template_parser = template_parser
        self.initial_retrieve_multiplier = initial_retrieve_multiplier
        self.logger = logging.getLogger("uvicorn")
    
    def get_strategy_name(self) -> str:
        return "ReRank RAG (Two-Stage Retrieval)"
    
    async def retrieve_documents(
        self, 
        query: str, 
        project: Project,
        limit: int
    ) -> List[RetrievedDocument]:
        """
        Retrieve documents using two-stage reranking.
        """
        # Step 1: First-stage retrieval (retrieve more candidates)
        initial_limit = limit * self.initial_retrieve_multiplier
        
        collection_name = self._create_collection_name(
            project_id=typing.cast(int, project.project_id)
        )
        
        # Embed query
        query_vectors = self.embedding_client.embed_text(
            text=query,
            document_type=DocumentTypeEnum.QUERY.value
        )
        
        if not query_vectors or len(query_vectors) == 0:
            self.logger.error("Failed to embed query")
            return []
        
        # Initial retrieval
        initial_results = await self.vectordb_client.search_by_vector(
            collection_name=collection_name,
            vector=query_vectors[0],
            limit=initial_limit
        )
        
        if not initial_results or len(initial_results) == 0:
            self.logger.warning("No documents found in initial retrieval")
            return []
        
        self.logger.info(
            f"Initial retrieval: {len(initial_results)} candidates"
        )
        
        # Step 2: Rerank using LLM (or cross-encoder if available)
        reranked_results = await self._rerank_documents(
            query=query,
            documents=initial_results,
            top_k=limit
        )
        
        self.logger.info(
            f"Reranked to top {len(reranked_results)} documents"
        )
        
        return reranked_results
    
    async def generate_answer(
        self,
        query: str,
        retrieved_documents: List[RetrievedDocument],
        chat_history: List[Dict[str, str]]
    ) -> Tuple[Optional[str], str]:
        """
        Generate answer (same as Basic RAG).
        """
        if not retrieved_documents or len(retrieved_documents) == 0:
            return None, ""
        
        system_prompt = self.template_parser.get("rag", "system_prompt")
        
        documents_prompts = "\n".join([
            self.template_parser.get("rag", "document_prompt", {
                "doc_num": idx + 1,
                "chunk_text": self.generation_client.process_text(doc.text),
            })
            for idx, doc in enumerate(retrieved_documents)
        ])
        
        footer_prompt = self.template_parser.get("rag", "footer_prompt", {
            "query": query
        })
        
        full_prompt = "\n\n".join([documents_prompts, footer_prompt])
        
        final_chat_history = [
            self.generation_client.construct_prompt(
                prompt=system_prompt,
                role=self.generation_client.enums.SYSTEM.value,
            )
        ]
        final_chat_history.extend(chat_history)
        
        answer = self.generation_client.generate_text(
            prompt=full_prompt,
            chat_history=final_chat_history
        )
        
        return answer, full_prompt
    
    async def _rerank_documents(
        self,
        query: str,
        documents: List[RetrievedDocument],
        top_k: int
    ) -> List[RetrievedDocument]:
        """
        Rerank documents using LLM-based relevance scoring.
        
        PRODUCTION NOTE:
        For better performance and cost-efficiency, replace this with a 
        dedicated cross-encoder model such as:
        - sentence-transformers/ms-marco-MiniLM-L-12-v2
        - BAAI/bge-reranker-large
        
        These models are specifically trained for reranking and are much
        faster and cheaper than using an LLM.
        """
        try:
            # Create reranking prompt
            docs_text = "\n\n".join([
                f"[Document {idx + 1}]\n{doc.text[:500]}"  # Truncate for context
                for idx, doc in enumerate(documents)
            ])
            
            rerank_prompt = f"""Given the query and documents below, rank the documents by relevance to the query.
Return ONLY a comma-separated list of document numbers in order of relevance (most relevant first).

Query: {query}

Documents:
{docs_text}

Ranking (comma-separated numbers):"""
            
            response = self.generation_client.generate_text(
                prompt=rerank_prompt,
                chat_history=[],
                temperature=0.1,  # Low temperature for consistent ranking
                max_output_tokens=100
            )
            
            if not response:
                # Fallback: return original order
                return documents[:top_k]
            
            # Parse ranking
            ranking = self._parse_ranking(response, len(documents))
            
            # Reorder documents
            reranked = []
            for rank_idx in ranking[:top_k]:
                if 0 <= rank_idx < len(documents):
                    # Update score based on new rank
                    doc = documents[rank_idx]
                    new_score = 1.0 - (len(reranked) / top_k)  # Decreasing score
                    reranked.append(
                        RetrievedDocument(text=doc.text, score=new_score)
                    )
            
            return reranked if reranked else documents[:top_k]
            
        except Exception as e:
            self.logger.error(f"Error during reranking: {e}")
            # Fallback: return original order
            return documents[:top_k]
    
    def _parse_ranking(self, response: str, num_docs: int) -> List[int]:
        """
        Parse LLM response to extract document ranking.
        """
        try:
            # Extract numbers from response
            import re
            numbers = re.findall(r'\d+', response)
            
            # Convert to 0-indexed
            ranking = [int(n) - 1 for n in numbers if n.isdigit()]
            
            # Filter valid indices
            ranking = [idx for idx in ranking if 0 <= idx < num_docs]
            
            # Add missing documents at the end
            remaining = [i for i in range(num_docs) if i not in ranking]
            ranking.extend(remaining)
            
            return ranking
            
        except Exception as e:
            self.logger.error(f"Error parsing ranking: {e}")
            return list(range(num_docs))
    
    def _create_collection_name(self, project_id: int) -> str:
        return f"collection_{self.vectordb_client.default_vector_size}_{project_id}".strip()