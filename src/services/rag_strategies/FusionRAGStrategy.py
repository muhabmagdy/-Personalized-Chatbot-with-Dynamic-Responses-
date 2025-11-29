from .RAGStrategyInterface import RAGStrategyInterface
from typing import List, Dict, Optional, Tuple
from models.db_schemes import Project, RetrievedDocument
from stores.llm.LLMEnums import DocumentTypeEnum
import typing
import logging
from collections import defaultdict

class FusionRAGStrategy(RAGStrategyInterface):
    """
    Fusion RAG Strategy: Query expansion → Multiple searches → Reciprocal Rank Fusion
    
    Advanced RAG approach:
    1. Generate multiple query variations using LLM
    2. Search vector DB with each variation
    3. Fuse results using Reciprocal Rank Fusion (RRF)
    4. Generate answer with best-ranked documents
    
    Best for: Complex queries, multi-perspective information needs
    
    Reference: "Forget RAG, the Future is RAG-Fusion" - Advanced RAG Techniques
    """
    
    def __init__(
        self, 
        vectordb_client,
        generation_client,
        embedding_client,
        template_parser,
        num_queries: int = 3
    ):
        """
        Initialize Fusion RAG strategy.
        
        Args:
            num_queries: Number of query variations to generate (default: 3)
        """
        self.vectordb_client = vectordb_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.template_parser = template_parser
        self.num_queries = num_queries
        self.logger = logging.getLogger("uvicorn")
    
    def get_strategy_name(self) -> str:
        return "Fusion RAG (Query Expansion + RRF)"
    
    async def retrieve_documents(
        self, 
        query: str, 
        project: Project,
        limit: int
    ) -> List[RetrievedDocument]:
        """
        Retrieve documents using query fusion approach.
        """
        # Step 1: Generate query variations
        query_variations = await self._generate_query_variations(query)
        self.logger.info(
            f"Generated {len(query_variations)} query variations"
        )
        
        # Step 2: Search with each query variation
        all_results = {}  # doc_text -> list of (rank, score)
        
        collection_name = self._create_collection_name(
            project_id=typing.cast(int, project.project_id)
        )
        
        for idx, q in enumerate(query_variations):
            # Embed query
            query_vectors = self.embedding_client.embed_text(
                text=q,
                document_type=DocumentTypeEnum.QUERY.value
            )
            
            if not query_vectors or len(query_vectors) == 0:
                continue
            
            # Search
            results = await self.vectordb_client.search_by_vector(
                collection_name=collection_name,
                vector=query_vectors[0],
                limit=limit * 2  # Retrieve more for fusion
            )
            
            # Store results with rank
            for rank, doc in enumerate(results):
                if doc.text not in all_results:
                    all_results[doc.text] = []
                all_results[doc.text].append((rank + 1, doc.score))
        
        # Step 3: Apply Reciprocal Rank Fusion
        fused_results = self._reciprocal_rank_fusion(all_results, limit)
        
        self.logger.info(
            f"Fused {len(all_results)} unique documents → top {len(fused_results)}"
        )
        
        return fused_results
    
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
    
    async def _generate_query_variations(self, original_query: str) -> List[str]:
        """
        Generate multiple query variations using LLM.
        """
        prompt = f"""Given the following question, generate {self.num_queries - 1} alternative phrasings that capture the same intent but use different wording.

Original question: {original_query}

Generate {self.num_queries - 1} variations (one per line):"""
        
        try:
            response = self.generation_client.generate_text(
                prompt=prompt,
                chat_history=[],
                temperature=0.7
            )
            
            if not response:
                return [original_query]
            
            # Parse variations
            variations = [original_query]  # Always include original
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            
            for line in lines[:self.num_queries - 1]:
                # Remove numbering if present
                cleaned = line.lstrip('0123456789.-) ').strip()
                if cleaned and len(cleaned) > 10:
                    variations.append(cleaned)
            
            return variations
            
        except Exception as e:
            self.logger.error(f"Error generating query variations: {e}")
            return [original_query]
    
    def _reciprocal_rank_fusion(
        self, 
        all_results: Dict[str, List[Tuple[int, float]]], 
        limit: int,
        k: int = 60
    ) -> List[RetrievedDocument]:
        """
        Apply Reciprocal Rank Fusion to combine multiple ranking lists.
        
        RRF Score = Σ(1 / (k + rank))
        
        Args:
            all_results: Dict mapping doc_text to list of (rank, score) tuples
            limit: Number of top documents to return
            k: RRF constant (default: 60)
        """
        fusion_scores = {}
        
        for doc_text, rankings in all_results.items():
            rrf_score = sum(1.0 / (k + rank) for rank, _ in rankings)
            # Also consider original similarity scores
            avg_similarity = sum(score for _, score in rankings) / len(rankings)
            
            # Combine RRF with average similarity
            fusion_scores[doc_text] = (rrf_score + avg_similarity) / 2
        
        # Sort by fusion score
        sorted_docs = sorted(
            fusion_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:limit]
        
        # Convert to RetrievedDocument
        return [
            RetrievedDocument(text=doc_text, score=score)
            for doc_text, score in sorted_docs
        ]
    
    def _create_collection_name(self, project_id: int) -> str:
        return f"collection_{self.vectordb_client.default_vector_size}_{project_id}".strip()