from .RAGEvaluatorInterface import RAGEvaluatorInterface
from models.dtos.EvaluationResult import EvaluationResult
from models.enums.EvaluationMethodEnum import EvaluationMethodEnum
from typing import List, Optional, Dict
import logging

class GroundednessEvaluator(RAGEvaluatorInterface):
    """
    Groundedness (Faithfulness) Evaluator: Detects hallucinations.
    
    Measures: Is the answer based solely on retrieved documents?
    
    This is critical for RAG systems to ensure factual accuracy
    and prevent the LLM from generating unsupported information.
    
    Evaluation Method:
    - LLM Feedback: Ask LLM to verify each claim against documents
    
    Score Interpretation:
    - 1.0: All claims supported by documents
    - 0.7-0.9: Mostly grounded, minor unsupported details
    - 0.4-0.6: Some hallucinations
    - 0.0-0.3: Significant hallucinations
    
    Use Cases:
    - Detect and prevent hallucinations
    - Ensure factual accuracy
    - Compare LLM prompting strategies
    - Monitor system reliability
    """
    
    def __init__(self, llm_client):
        """
        Initialize evaluator.
        
        Args:
            llm_client: LLM for claim verification
        """
        self.llm_client = llm_client
        self.logger = logging.getLogger("uvicorn")
    
    def get_metric_name(self) -> str:
        return "groundedness"
    
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
        """
        Evaluate groundedness by checking claims against documents.
        """
        try:
            if not retrieved_documents:
                return EvaluationResult(
                    metric_name=self.get_metric_name(),
                    score=0.0,
                    method=self.get_evaluation_method(),
                    explanation="No documents to verify against",
                    metadata={"document_count": 0}
                )
            
            # Extract claims from answer
            claims = await self._extract_claims(answer)
            
            if not claims:
                return EvaluationResult(
                    metric_name=self.get_metric_name(),
                    score=1.0,
                    method=self.get_evaluation_method(),
                    explanation="No factual claims to verify",
                    metadata={"claim_count": 0}
                )
            
            # Verify each claim
            verification_results = await self._verify_claims(
                claims, retrieved_documents
            )
            
            # Calculate groundedness score
            supported_claims = sum(1 for v in verification_results if v['supported'])
            total_claims = len(verification_results)
            
            groundedness_score = supported_claims / total_claims if total_claims > 0 else 1.0
            
            # Build explanation
            explanation = f"{supported_claims}/{total_claims} claims supported by retrieved documents"
            
            if groundedness_score < 0.7:
                unsupported = [v['claim'] for v in verification_results if not v['supported']]
                explanation += f". Unsupported: {', '.join(unsupported[:2])}"
                if len(unsupported) > 2:
                    explanation += f" and {len(unsupported) - 2} more"
            
            return EvaluationResult(
                metric_name=self.get_metric_name(),
                score=self.normalize_score(groundedness_score),
                method=self.get_evaluation_method(),
                explanation=explanation,
                metadata={
                    "total_claims": total_claims,
                    "supported_claims": supported_claims,
                    "unsupported_claims": total_claims - supported_claims,
                    "verification_details": verification_results
                }
            )
            
        except Exception as e:
            self.logger.error(f"Groundedness evaluation error: {e}")
            return EvaluationResult(
                metric_name=self.get_metric_name(),
                score=0.0,
                method=self.get_evaluation_method(),
                explanation=f"Evaluation failed: {str(e)}",
                metadata={"error": str(e)}
            )
    
    async def _extract_claims(self, answer: str) -> List[str]:
        """
        Extract factual claims from the answer.
        """
        try:
            prompt = f"""Extract the main factual claims from this answer.
                    List each claim as a separate bullet point.

                    Answer: {answer}

                    Factual claims (one per line, starting with "-"):"""
            
            response = self.llm_client.generate_text(
                prompt=prompt,
                chat_history=[],
                temperature=0.3,
                max_output_tokens=1000
            )
            
            if not response:
                return []
            
            # Parse claims
            claims = []
            for line in response.split('\n'):
                line = line.strip()
                if line.startswith('-') or line.startswith('•'):
                    claim = line.lstrip('-•').strip()
                    if claim and len(claim) > 10:
                        claims.append(claim)
            self.logger.info(f"Extracted claims: {claims}")
            return claims[:10]  # Limit to top 10 claims
            
        except Exception as e:
            self.logger.error(f"Claim extraction error: {e}")
            return []
    
    async def _verify_claims(
        self,
        claims: List[str],
        documents: List[str]
    ) -> List[Dict]:
        """
        Verify each claim against the retrieved documents.
        """
        verification_results = []
        
        # Combine documents for verification
        combined_docs = "\n\n".join([
            f"[Document {idx+1}]\n{doc}"
            for idx, doc in enumerate(documents)
        ])
        
        for claim in claims:
            try:
                is_supported = await self._verify_single_claim(claim, combined_docs)
                
                verification_results.append({
                    'claim': claim,
                    'supported': is_supported
                })
                
            except Exception as e:
                self.logger.error(f"Error verifying claim '{claim}': {e}")
                verification_results.append({
                    'claim': claim,
                    'supported': False
                })
        
        return verification_results
    
    async def _verify_single_claim(
        self,
        claim: str,
        documents: str
    ) -> bool:
        """
        Verify a single claim against documents.
        """
        try:
            prompt = f"""Is the following claim supported by the documents?

                    Claim: {claim}

                    Documents:
                    {documents[:2000]}

                    Answer with only "YES" or "NO":"""
            
            response = self.llm_client.generate_text(
                prompt=prompt,
                chat_history=[],
                temperature=0.1,
                max_output_tokens=1000
            )
            
            if not response:
                return False
            
            # Check response
            response_lower = response.lower().strip()
            return 'yes' in response_lower and 'no' not in response_lower
            
        except Exception as e:
            self.logger.error(f"Claim verification error: {e}")
            return False