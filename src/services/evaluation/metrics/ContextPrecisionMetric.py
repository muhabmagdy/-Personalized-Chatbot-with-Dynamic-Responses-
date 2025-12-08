# project/src/services/evaluation/metrics/ContextPrecisionMetric.py

from typing import List
import logging


class ContextPrecisionMetric:
    """
    Context Precision metric for RAG evaluation.
    
    Measures how precise the retrieved contexts are - whether the useful
    information appears early in the ranked list of contexts.
    
    Algorithm:
    1. For each context, determine if it's relevant to answering the query
    2. Calculate precision at each position (P@k)
    3. Score = average precision (rewards ranking relevant contexts higher)
    
    This implementation does NOT require ground truth.
    """
    
    def __init__(self, llm_client):
        """
        Initialize context precision metric.
        
        Args:
            llm_client: LLM client for relevance assessment
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
        Compute context precision score.
        
        Args:
            query: Original user query
            answer: Generated answer (used to verify context usefulness)
            contexts: Retrieved document texts in ranked order
            
        Returns:
            Context precision score between 0.0 and 1.0
        """
        if not contexts or not query:
            return 0.0
        
        # Step 1: Assess relevance of each context
        relevance_scores = []
        
        for i, context in enumerate(contexts, 1):
            is_relevant = self._is_context_relevant(
                query=query,
                context=context,
                answer=answer
            )
            relevance_scores.append(1 if is_relevant else 0)
            
            self.logger.debug(
                f"Context {i}/{len(contexts)}: "
                f"{'relevant' if is_relevant else 'not relevant'}"
            )
        
        # Step 2: Calculate average precision
        # AP = sum of (precision at k * relevance at k) / total relevant
        total_relevant = sum(relevance_scores)
        
        if total_relevant == 0:
            self.logger.warning("No relevant contexts found")
            return 0.0
        
        precision_sum = 0.0
        relevant_count = 0
        
        for k, is_relevant in enumerate(relevance_scores, 1):
            if is_relevant:
                relevant_count += 1
                precision_at_k = relevant_count / k
                precision_sum += precision_at_k
        
        average_precision = precision_sum / total_relevant
        
        self.logger.debug(
            f"Context precision: {relevant_count}/{len(contexts)} relevant, "
            f"AP={average_precision:.3f}"
        )
        
        return average_precision
    
    def _is_context_relevant(
        self,
        query: str,
        context: str,
        answer: str
    ) -> bool:
        """
        Determine if a context is relevant for answering the query.
        
        A context is relevant if it contains information that helps answer
        the query or is used in generating the answer.
        
        Args:
            query: Original query
            context: Single retrieved context to evaluate
            answer: Generated answer
            
        Returns:
            True if context is relevant, False otherwise
        """
        prompt = f"""Determine if the given context is relevant for answering the query.

                Query:
                {query}

                Context:
                {context}

                Generated Answer:
                {answer}

                Is this context relevant for answering the query? Consider:
                - Does it contain information that helps answer the query?
                - Is the information from this context used in the generated answer?
                - Would removing this context hurt the answer quality?

                Answer with ONLY "yes" or "no".

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
            if any(word in response_lower for word in ["yes", "true", "relevant"]):
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to assess context relevance: {e}")
            return False