from .RAGEvaluatorInterface import RAGEvaluatorInterface
from models.dtos.EvaluationResult import EvaluationResult
from models.enums.EvaluationMethodEnum import EvaluationMethodEnum

from typing import List, Optional
import logging
import re


class AnswerRelevanceEvaluator(RAGEvaluatorInterface):
    """
    Answer Relevance Evaluator: Assesses if answer addresses the query.
    
    Measures: How well does the answer respond to the user's question?
    
    Evaluation Methods:
    1. LLM Feedback: Ask LLM to rate relevance (primary)
    2. Similarity: Compare query-answer embeddings (fallback)
    
    Score Interpretation:
    - 1.0: Perfectly answers the question
    - 0.7-0.9: Mostly relevant, minor gaps
    - 0.4-0.6: Partially relevant
    - 0.0-0.3: Not relevant or off-topic
    
    Use Cases:
    - Evaluate LLM output quality
    - Compare different answer strategies
    - Monitor system performance
    """
    
    def __init__(
        self,
        llm_client,
        embedding_client,
        use_hybrid: bool = True
    ):
        """
        Initialize evaluator.
        
        Args:
            llm_client: LLM for feedback evaluation
            embedding_client: For similarity comparison
            use_hybrid: Combine LLM and similarity (default: True)
        """
        self.llm_client = llm_client
        self.embedding_client = embedding_client
        self.use_hybrid = use_hybrid
        self.logger = logging.getLogger("uvicorn")
    
    def get_metric_name(self) -> str:
        return "answer_relevance"
    
    def get_evaluation_method(self) -> EvaluationMethodEnum:
        return EvaluationMethodEnum.HYBRID if self.use_hybrid else EvaluationMethodEnum.LLM_FEEDBACK
    
    async def evaluate(
        self,
        query: str,
        answer: str,
        retrieved_documents: List[str],
        ground_truth: Optional[str] = None,
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate answer relevance using LLM feedback and/or similarity.
        """
        try:
            # Method 1: LLM Feedback
            llm_score, llm_explanation = await self._evaluate_with_llm(query, answer)
            
            # Method 2: Similarity (if hybrid)
            similarity_score = 0.0
            if self.use_hybrid:
                similarity_score = await self._evaluate_with_similarity(query, answer)
            
            # Combine scores
            if self.use_hybrid:
                final_score = (llm_score * 0.7) + (similarity_score * 0.3)
                explanation = f"{llm_explanation} (LLM: {llm_score:.2f}, Similarity: {similarity_score:.2f})"
            else:
                final_score = llm_score
                explanation = llm_explanation
            
            return EvaluationResult(
                metric_name=self.get_metric_name(),
                score=self.normalize_score(final_score),
                method=self.get_evaluation_method(),
                explanation=explanation,
                metadata={
                    "llm_score": llm_score,
                    "similarity_score": similarity_score if self.use_hybrid else None,
                    "query_length": len(query),
                    "answer_length": len(answer)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Answer relevance evaluation error: {e}")
            return EvaluationResult(
                metric_name=self.get_metric_name(),
                score=0.0,
                method=self.get_evaluation_method(),
                explanation=f"Evaluation failed: {str(e)}",
                metadata={"error": str(e)}
            )
    
    async def _evaluate_with_llm(
        self,
        query: str,
        answer: str
    ) -> tuple[float, str]:
        """
        Use LLM to evaluate answer relevance.
        """
        prompt = f"""Evaluate how well the answer addresses the question.

                Question: {query}

                Answer: {answer}

                Rate the relevance on a scale of 0.0 to 1.0:
                - 1.0: Perfectly answers the question
                - 0.7-0.9: Mostly relevant
                - 0.4-0.6: Partially relevant
                - 0.0-0.3: Not relevant

                Provide your rating and brief explanation in this format:
                SCORE: <number>
                EXPLANATION: <brief explanation>"""
        
        try:
            response = self.llm_client.generate_text(
                prompt=prompt,
                chat_history=[],
                temperature=0.3,
                max_output_tokens=1000
            )
            
            if not response:
                return 0.5, "LLM evaluation unavailable"
            
            # Parse response
            score_match = re.search(r'SCORE:\s*([\d.]+)', response)
            explanation_match = re.search(r'EXPLANATION:\s*(.+)', response, re.DOTALL)
            
            score = float(score_match.group(1)) if score_match else 0.5
            explanation = explanation_match.group(1).strip() if explanation_match else "No explanation provided"
            
            return score, explanation
            
        except Exception as e:
            self.logger.error(f"LLM evaluation error: {e}")
            return 0.5, f"LLM evaluation error: {str(e)}"
    
    async def _evaluate_with_similarity(
        self,
        query: str,
        answer: str
    ) -> float:
        """
        Use embedding similarity to evaluate relevance.
        """
        try:
            from stores.llm.LLMEnums import DocumentTypeEnum
            
            # Embed query and answer
            query_embedding = self.embedding_client.embed_text(
                text=query,
                document_type=DocumentTypeEnum.QUERY.value
            )
            
            answer_embedding = self.embedding_client.embed_text(
                text=answer,
                document_type=DocumentTypeEnum.DOCUMENT.value
            )
            
            if not query_embedding or not answer_embedding:
                return 0.5
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding[0], answer_embedding[0])
            
            return similarity
            
        except Exception as e:
            self.logger.error(f"Similarity evaluation error: {e}")
            return 0.5
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import math
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)