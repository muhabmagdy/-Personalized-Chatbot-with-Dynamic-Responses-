from .RAGEvaluatorInterface import RAGEvaluatorInterface
from models.dtos.EvaluationResult import EvaluationResult
from models.enums.EvaluationMethodEnum import EvaluationMethodEnum

from typing import List, Optional
import logging

#Considered as Advanced Rag Evaluations
class ContextRecallEvaluator(RAGEvaluatorInterface):
    """
    Context Recall: Measures retrieval completeness.
    
    Evaluates: Did we retrieve all relevant information?
    
    Requires ground truth to identify what information should be retrieved.
    """
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.logger = logging.getLogger("uvicorn")
    
    def get_metric_name(self) -> str:
        return "context_recall"
    
    def get_evaluation_method(self) -> EvaluationMethodEnum:
        return EvaluationMethodEnum.LLM_FEEDBACK
    
    async def evaluate(
        self,
        query: str,
        answer: str,
        retrieved_documents: List[str],
        ground_truth: Optional[str] = None,
        **kwargs
    ) -> EvaluationResult:
        """Evaluate context recall."""
        try:
            if not ground_truth:
                return EvaluationResult(
                    metric_name=self.get_metric_name(),
                    score=0.5,
                    method=self.get_evaluation_method(),
                    explanation="Ground truth required for recall evaluation",
                    metadata={}
                )
            
            # Extract key facts from ground truth
            ground_truth_facts = await self._extract_facts(ground_truth)
            
            if not ground_truth_facts:
                return EvaluationResult(
                    metric_name=self.get_metric_name(),
                    score=1.0,
                    method=self.get_evaluation_method(),
                    explanation="No facts to verify in ground truth",
                    metadata={}
                )
            
            # Check which facts are covered in retrieved docs
            combined_docs = "\n\n".join(retrieved_documents)
            covered_facts = 0
            
            for fact in ground_truth_facts:
                if await self._is_fact_covered(fact, combined_docs):
                    covered_facts += 1
            
            recall = covered_facts / len(ground_truth_facts)
            
            explanation = f"{covered_facts}/{len(ground_truth_facts)} required facts retrieved"
            
            return EvaluationResult(
                metric_name=self.get_metric_name(),
                score=self.normalize_score(recall),
                method=self.get_evaluation_method(),
                explanation=explanation,
                metadata={
                    "total_facts": len(ground_truth_facts),
                    "covered_facts": covered_facts,
                    "missing_facts": len(ground_truth_facts) - covered_facts
                }
            )
            
        except Exception as e:
            self.logger.error(f"Context recall evaluation error: {e}")
            return EvaluationResult(
                metric_name=self.get_metric_name(),
                score=0.0,
                method=self.get_evaluation_method(),
                explanation=f"Evaluation failed: {str(e)}",
                metadata={"error": str(e)}
            )
    
    async def _extract_facts(self, text: str) -> List[str]:
        """Extract key facts from text."""
        try:
            prompt = f"""Extract the key facts from this text (one per line):

                    {text}

                    Facts:"""
            
            response = self.llm_client.generate_text(
                prompt=prompt,
                chat_history=[],
                temperature=0.3,
                max_output_tokens=1000
            )
            
            if not response:
                return []
            
            facts = [line.strip().lstrip('-•') for line in response.split('\n') if line.strip()]
            return facts[:10]
            
        except Exception:
            return []
    
    async def _is_fact_covered(self, fact: str, documents: str) -> bool:
        """Check if fact is covered in documents."""
        try:
            prompt = f"""Is this fact mentioned or covered in the documents?

                    Fact: {fact}

                    Documents:
                    {documents[:2000]}

                    Answer YES or NO:"""
            
            response = self.llm_client.generate_text(
                prompt=prompt,
                chat_history=[],
                temperature=0.1,
                max_output_tokens=1000
            )
            
            return 'yes' in response.lower() if response else False
            
        except Exception:
            return False