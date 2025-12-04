from .RAGEvaluatorInterface import RAGEvaluatorInterface
from models.dtos.EvaluationResult import EvaluationResult
from models.enums.EvaluationMethodEnum import EvaluationMethodEnum

from typing import List, Optional
import logging

#Considered as Advanced Rag Evaluations
class ContextPrecisionEvaluator(RAGEvaluatorInterface):
    """
    Context Precision: Measures relevance ranking quality.
    
    Evaluates: Are the most relevant documents ranked highest?
    
    This metric assesses whether the retrieval system properly ranks
    documents by relevance, placing the most useful documents first.
    
    Formula: Precision@K for top-K documents
    """
    
    def __init__(self, llm_client, embedding_client):
        self.llm_client = llm_client
        self.embedding_client = embedding_client
        self.logger = logging.getLogger("uvicorn")
    
    def get_metric_name(self) -> str:
        return "context_precision"
    
    def get_evaluation_method(self) -> EvaluationMethodEnum:
        return EvaluationMethodEnum.HYBRID
    
    async def evaluate(
        self,
        query: str,
        answer: str,
        retrieved_documents: List[str],
        ground_truth: Optional[str] = None,
        **kwargs
    ) -> EvaluationResult:
        """Evaluate context precision."""
        try:
            if not retrieved_documents:
                return EvaluationResult(
                    metric_name=self.get_metric_name(),
                    score=0.0,
                    method=self.get_evaluation_method(),
                    explanation="No documents retrieved",
                    metadata={}
                )
            
            # Calculate relevance for each position
            relevance_scores = []
            for idx, doc in enumerate(retrieved_documents):
                relevance = await self._calculate_document_relevance(query, doc)
                relevance_scores.append(relevance)
            
            # Calculate precision: Are top docs most relevant?
            precision_at_k = []
            for k in range(1, len(relevance_scores) + 1):
                relevant_in_top_k = sum(1 for r in relevance_scores[:k] if r >= 0.7)
                precision_at_k.append(relevant_in_top_k / k)
            
            # Average precision
            avg_precision = sum(precision_at_k) / len(precision_at_k) if precision_at_k else 0.0
            
            explanation = f"Avg precision: {avg_precision:.2f}. Top-3 precision: {precision_at_k[2]:.2f if len(precision_at_k) > 2 else 'N/A'}"
            
            return EvaluationResult(
                metric_name=self.get_metric_name(),
                score=self.normalize_score(avg_precision),
                method=self.get_evaluation_method(),
                explanation=explanation,
                metadata={
                    "precision_at_k": precision_at_k,
                    "relevance_scores": relevance_scores
                }
            )
            
        except Exception as e:
            self.logger.error(f"Context precision evaluation error: {e}")
            return EvaluationResult(
                metric_name=self.get_metric_name(),
                score=0.0,
                method=self.get_evaluation_method(),
                explanation=f"Evaluation failed: {str(e)}",
                metadata={"error": str(e)}
            )
    
    async def _calculate_document_relevance(self, query: str, document: str) -> float:
        """Calculate relevance of a single document."""
        try:
            from stores.llm.LLMEnums import DocumentTypeEnum
            
            query_emb = self.embedding_client.embed_text(
                text=query,
                document_type=DocumentTypeEnum.QUERY.value
            )
            
            doc_emb = self.embedding_client.embed_text(
                text=document[:1000],
                document_type=DocumentTypeEnum.DOCUMENT.value
            )
            
            if not query_emb or not doc_emb:
                return 0.5
            
            return self._cosine_similarity(query_emb[0], doc_emb[0])
            
        except Exception:
            return 0.5
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        import math
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        mag1 = math.sqrt(sum(a * a for a in vec1))
        mag2 = math.sqrt(sum(b * b for b in vec2))
        return dot_product / (mag1 * mag2) if mag1 and mag2 else 0.0

