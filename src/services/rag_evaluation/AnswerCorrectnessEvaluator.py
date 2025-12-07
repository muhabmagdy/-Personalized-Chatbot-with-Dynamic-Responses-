from .RAGEvaluatorInterface import RAGEvaluatorInterface
from models.dtos.EvaluationResult import EvaluationResult
from models.enums.EvaluationMethodEnum import EvaluationMethodEnum

from typing import List, Optional
import logging
import re

#Considered as Advanced Rag Evaluations
class AnswerCorrectnessEvaluator(RAGEvaluatorInterface):
    """
    Answer Correctness: Compares answer to ground truth.
    
    Evaluates: Is the answer factually correct?
    
    Requires ground truth answer for comparison.
    """
    
    def __init__(self, llm_client, embedding_client):
        self.llm_client = llm_client
        self.embedding_client = embedding_client
        self.logger = logging.getLogger("uvicorn")
    
    def get_metric_name(self) -> str:
        return "answer_correctness"
    
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
        """Evaluate answer correctness against ground truth."""
        try:
            if not ground_truth:
                return EvaluationResult(
                    metric_name=self.get_metric_name(),
                    score=0.5,
                    method=self.get_evaluation_method(),
                    explanation="Ground truth required for correctness evaluation",
                    metadata={}
                )
            
            # Semantic similarity
            similarity = await self._calculate_similarity(answer, ground_truth)
            
            # LLM-based correctness check
            llm_score = await self._llm_correctness_check(answer, ground_truth)
            
            # Combine scores
            final_score = (similarity * 0.5) + (llm_score * 0.5)
            
            explanation = f"Similarity: {similarity:.2f}, LLM correctness: {llm_score:.2f}"
            
            return EvaluationResult(
                metric_name=self.get_metric_name(),
                score=self.normalize_score(final_score),
                method=self.get_evaluation_method(),
                explanation=explanation,
                metadata={
                    "similarity_score": similarity,
                    "llm_score": llm_score
                }
            )
            
        except Exception as e:
            self.logger.error(f"Answer correctness evaluation error: {e}")
            return EvaluationResult(
                metric_name=self.get_metric_name(),
                score=0.0,
                method=self.get_evaluation_method(),
                explanation=f"Evaluation failed: {str(e)}",
                metadata={"error": str(e)}
            )
    
    async def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity."""
        try:
            from stores.llm.LLMEnums import DocumentTypeEnum
            
            emb1 = self.embedding_client.embed_text(
                text=text1,
                document_type=DocumentTypeEnum.DOCUMENT.value
            )
            emb2 = self.embedding_client.embed_text(
                text=text2,
                document_type=DocumentTypeEnum.DOCUMENT.value
            )
            
            if not emb1 or not emb2:
                return 0.5
            
            return self._cosine_similarity(emb1[0], emb2[0])
            
        except Exception:
            return 0.5
    
    async def _llm_correctness_check(self, answer: str, ground_truth: str) -> float:
        """LLM-based correctness verification."""
        try:
            prompt = f"""Compare the answer to the ground truth. Rate correctness 0.0 to 1.0.

                    Ground Truth: {ground_truth}

                    Answer: {answer}

                    Correctness score (number only):"""
            
            response = self.llm_client.generate_text(
                prompt=prompt,
                chat_history=[],
                temperature=0.3,
                max_output_tokens=1000
            )
            
            if not response:
                return 0.5
            
            match = re.search(r'(\d+\.?\d*)', response)
            return float(match.group(1)) if match else 0.5
            
        except Exception:
            return 0.5
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        import math
        dot = sum(a * b for a, b in zip(vec1, vec2))
        mag1 = math.sqrt(sum(a * a for a in vec1))
        mag2 = math.sqrt(sum(b * b for b in vec2))
        return dot / (mag1 * mag2) if mag1 and mag2 else 0.0