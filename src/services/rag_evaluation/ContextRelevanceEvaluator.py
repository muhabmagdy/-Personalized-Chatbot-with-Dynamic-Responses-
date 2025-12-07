from .RAGEvaluatorInterface import RAGEvaluatorInterface
from models.dtos.EvaluationResult import EvaluationResult
from models.enums.EvaluationMethodEnum import EvaluationMethodEnum
from typing import List, Optional
import logging


class ContextRelevanceEvaluator(RAGEvaluatorInterface):
    """
    Context Relevance Evaluator: Assesses retriever quality.
    
    Measures: How relevant are the retrieved documents to the query?
    
    This metric specifically evaluates the retrieval component,
    independent of the generation quality.
    
    Evaluation Methods:
    1. Similarity Comparison: Query-document embeddings (primary)
    2. LLM Feedback: Ask LLM to rate document relevance (optional)
    
    Score Interpretation:
    - 1.0: All retrieved docs highly relevant
    - 0.7-0.9: Most docs relevant
    - 0.4-0.6: Mixed relevance
    - 0.0-0.3: Poor retrieval quality
    
    Use Cases:
    - Evaluate retrieval strategies
    - Optimize chunk size and embeddings
    - Compare RAG strategies
    """
    
    def __init__(
        self,
        embedding_client,
        llm_client=None,
        use_llm_verification: bool = False
    ):
        """
        Initialize evaluator.
        
        Args:
            embedding_client: For similarity computation
            llm_client: Optional LLM for verification
            use_llm_verification: Use LLM to verify relevance (slower but more accurate)
        """
        self.embedding_client = embedding_client
        self.llm_client = llm_client
        self.use_llm_verification = use_llm_verification
        self.logger = logging.getLogger("uvicorn")
    
    def get_metric_name(self) -> str:
        return "context_relevance"
    
    def get_evaluation_method(self) -> EvaluationMethodEnum:
        return EvaluationMethodEnum.HYBRID if self.use_llm_verification else EvaluationMethodEnum.SIMILARITY_COMPARISON
    
    async def evaluate(
        self,
        query: str,
        answer: str,
        retrieved_documents: List[str],
        ground_truth: Optional[str] = None,
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate context relevance using similarity and optional LLM verification.
        """
        try:
            if not retrieved_documents:
                return EvaluationResult(
                    metric_name=self.get_metric_name(),
                    score=0.0,
                    method=self.get_evaluation_method(),
                    explanation="No documents retrieved",
                    metadata={"document_count": 0}
                )
            
            # Method 1: Similarity-based evaluation
            similarity_scores = await self._evaluate_with_similarity(
                query, retrieved_documents
            )
            
            avg_similarity = sum(similarity_scores) / len(similarity_scores)
            
            # Method 2: Optional LLM verification
            llm_score = None
            if self.use_llm_verification and self.llm_client:
                llm_score = await self._evaluate_with_llm(query, retrieved_documents)
            
            # Combine scores
            if llm_score is not None:
                final_score = (avg_similarity * 0.6) + (llm_score * 0.4)
                explanation = f"Average similarity: {avg_similarity:.2f}, LLM verification: {llm_score:.2f}"
            else:
                final_score = avg_similarity
                explanation = f"Average query-document similarity: {avg_similarity:.2f}"
            
            # Add relevance distribution
            relevant_count = sum(1 for score in similarity_scores if score >= 0.7)
            explanation += f" ({relevant_count}/{len(similarity_scores)} docs highly relevant)"
            
            return EvaluationResult(
                metric_name=self.get_metric_name(),
                score=self.normalize_score(final_score),
                method=self.get_evaluation_method(),
                explanation=explanation,
                metadata={
                    "document_count": len(retrieved_documents),
                    "similarity_scores": similarity_scores,
                    "avg_similarity": avg_similarity,
                    "llm_score": llm_score,
                    "highly_relevant_count": relevant_count
                }
            )
            
        except Exception as e:
            self.logger.error(f"Context relevance evaluation error: {e}")
            return EvaluationResult(
                metric_name=self.get_metric_name(),
                score=0.0,
                method=self.get_evaluation_method(),
                explanation=f"Evaluation failed: {str(e)}",
                metadata={"error": str(e)}
            )
    
    async def _evaluate_with_similarity(
        self,
        query: str,
        documents: List[str]
    ) -> List[float]:
        """
        Calculate similarity between query and each document.
        """
        try:
            from stores.llm.LLMEnums import DocumentTypeEnum
            
            # Embed query
            query_embedding = self.embedding_client.embed_text(
                text=query,
                document_type=DocumentTypeEnum.QUERY.value
            )
            
            if not query_embedding:
                return [0.5] * len(documents)
            
            query_vec = query_embedding[0]
            
            # Embed and compare each document
            similarities = []
            for doc in documents:
                doc_embedding = self.embedding_client.embed_text(
                    text=doc[:1000],  # Truncate for efficiency
                    document_type=DocumentTypeEnum.DOCUMENT.value
                )
                
                if doc_embedding:
                    similarity = self._cosine_similarity(query_vec, doc_embedding[0])
                    similarities.append(similarity)
                else:
                    similarities.append(0.5)
            
            return similarities
            
        except Exception as e:
            self.logger.error(f"Similarity calculation error: {e}")
            return [0.5] * len(documents)
    
    async def _evaluate_with_llm(
        self,
        query: str,
        documents: List[str]
    ) -> float:
        """
        Use LLM to verify document relevance.
        """
        try:
            docs_preview = "\n\n".join([
                f"Document {idx+1}: {doc[:200]}..."
                for idx, doc in enumerate(documents[:5])  # Limit for context
            ])
            
            prompt = f"""Evaluate how relevant these retrieved documents are to the query.

                    Query: {query}

                    Retrieved Documents:
                    {docs_preview}

                    Rate the overall relevance on a scale of 0.0 to 1.0.
                    Consider:
                    - Do the documents contain information to answer the query?
                    - Are the documents on-topic?
                    - Is there sufficient relevant content?

                    Provide only the numerical score (e.g., 0.85):"""
            
            response = self.llm_client.generate_text(
                prompt=prompt,
                chat_history=[],
                temperature=0.3,
                max_output_tokens=100
            )
            
            if not response:
                return 0.5
            
            # Extract score
            import re
            score_match = re.search(r'(\d+\.?\d*)', response)
            if score_match:
                return float(score_match.group(1))
            
            return 0.5
            
        except Exception as e:
            self.logger.error(f"LLM verification error: {e}")
            return 0.5
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity."""
        import math
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)