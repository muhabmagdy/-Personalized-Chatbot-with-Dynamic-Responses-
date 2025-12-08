# project/src/services/evaluation/metrics/FaithfulnessMetric.py

from typing import List
import logging
import json
import re


class FaithfulnessMetric:
    """
    Faithfulness metric for RAG evaluation.
    
    Measures how well the generated answer is grounded in the retrieved contexts.
    A high faithfulness score means the answer contains only information that can
    be verified from the provided contexts.
    
    Algorithm:
    1. Extract claims from the answer
    2. For each claim, verify if it's supported by the contexts
    3. Score = (number of supported claims) / (total claims)
    
    This implementation does NOT require ground truth.
    """
    
    def __init__(self, llm_client):
        """
        Initialize faithfulness metric.
        
        Args:
            llm_client: LLM client for claim extraction and verification
        """
        self.llm_client = llm_client
        self.logger = logging.getLogger("uvicorn")
    
    async def compute(
        self,
        query: str,
        answer: str,
        contexts: List[str]
    ) -> float:
        """
        Compute faithfulness score.
        
        Args:
            query: Original query (for context)
            answer: Generated answer to evaluate
            contexts: Retrieved document texts
            
        Returns:
            Faithfulness score between 0.0 and 1.0
        """
        if not answer or not contexts:
            return 0.0
        
        # Step 1: Extract claims from answer
        claims = self._extract_claims(answer)
        
        if not claims:
            self.logger.warning("No claims extracted from answer")
            return 1.0  # No claims = nothing to contradict
        
        self.logger.debug(f"Extracted {len(claims)} claims from answer")
        
        # Step 2: Verify each claim against contexts
        context_text = "\n\n".join(contexts)
        supported_count = 0
        
        for claim in claims:
            is_supported = self._verify_claim(claim, context_text)
            if is_supported:
                supported_count += 1
        
        # Step 3: Calculate score
        score = supported_count / len(claims) if claims else 0.0
        
        self.logger.debug(
            f"Faithfulness: {supported_count}/{len(claims)} claims supported = {score:.3f}"
        )
        
        return score
    
    def _extract_claims(self, answer: str) -> List[str]:
        """
        Extract atomic claims from the answer.
        
        Uses LLM to break down the answer into individual factual statements.
        
        Args:
            answer: Generated answer
            
        Returns:
            List of atomic claims
        """
        prompt = f"""Break down the following answer into a list of atomic, factual claims.
                Each claim should be a single statement that can be independently verified as true or false.

                Answer:
                {answer}

                Return ONLY a JSON array of claims, like this:
                ["claim 1", "claim 2", "claim 3"]

                Do not include any other text or explanation. Just the JSON array.
                """
        
        try:
            response = self.llm_client.generate_text(
                prompt=prompt,
                max_output_tokens=500,
                temperature=0.0
            )
            
            # Parse JSON response
            claims = self._parse_json_response(response)
            
            return claims if isinstance(claims, list) else []
            
        except Exception as e:
            self.logger.error(f"Failed to extract claims: {e}")
            return []
    
    def _verify_claim(self, claim: str, context: str) -> bool:
        """
        Verify if a claim is supported by the context.
        
        Uses LLM to determine if the claim can be inferred from the context.
        
        Args:
            claim: Atomic claim to verify
            context: All retrieved contexts combined
            
        Returns:
            True if claim is supported, False otherwise
        """
        prompt = f"""Given the following context, determine if the claim is supported by the information provided.

                Context:
                {context}

                Claim:
                {claim}

                Is this claim supported by the context? Answer with ONLY "yes" or "no".
                - Answer "yes" if the claim can be directly inferred or verified from the context
                - Answer "no" if the claim contradicts the context or cannot be verified from it

                Answer (yes/no):"""
        
        try:
            response = self.llm_client.generate_text(
                prompt=prompt,
                max_output_tokens=10,
                temperature=0.0
            )
            
            # Parse yes/no response
            response_lower = response.strip().lower()
            
            # Check for positive indicators
            if any(word in response_lower for word in ["yes", "true", "supported"]):
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to verify claim: {e}")
            return False
    
    def _parse_json_response(self, response: str) -> List[str]:
        """
        Parse JSON array from LLM response.
        
        Handles cases where LLM adds extra text or formatting.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Parsed list of claims
        """
        try:
            # Try direct JSON parsing
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON array from text
        try:
            # Look for JSON array pattern
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception:
            pass
        
        # Fallback: split by newlines and clean
        lines = response.strip().split('\n')
        claims = []
        
        for line in lines:
            # Remove common prefixes and clean
            line = line.strip()
            line = re.sub(r'^[-*•\d.)\]]+\s*', '', line)  # Remove bullets, numbers
            line = re.sub(r'^["\']|["\']$', '', line)  # Remove quotes
            
            if line and len(line) > 10:  # Minimum claim length
                claims.append(line)
        
        return claims