# project/src/services/evaluation/metrics/AnswerRelevancyMetric.py

from typing import List
import logging
import json
import re
import numpy as np


class AnswerRelevancyMetric:
    """
    Answer Relevancy metric for RAG evaluation.
    
    Measures how relevant the generated answer is to the original query.
    Uses question generation and semantic similarity to assess relevancy.
    
    Algorithm:
    1. Generate possible questions that the answer could address
    2. Calculate semantic similarity between generated questions and original query
    3. Score = average similarity (higher = more relevant)
    
    This implementation does NOT require ground truth.
    """
    
    def __init__(self, llm_client, embedding_client):
        """
        Initialize answer relevancy metric.
        
        Args:
            llm_client: LLM client for question generation
            embedding_client: Embedding client for similarity calculation
        """
        self.llm_client = llm_client
        self.embedding_client = embedding_client
        self.logger = logging.getLogger("uvicorn")
        self.num_questions = 3  # Number of questions to generate
    
    async def compute(
        self,
        query: str,
        answer: str,
        contexts: List[str]
    ) -> float:
        """
        Compute answer relevancy score.
        
        Args:
            query: Original user query
            answer: Generated answer to evaluate
            contexts: Retrieved contexts (not used but kept for interface consistency)
            
        Returns:
            Relevancy score between 0.0 and 1.0
        """
        if not answer or not query:
            return 0.0
        
        # Step 1: Generate questions from the answer
        generated_questions = self._generate_questions(answer)
        
        if not generated_questions:
            self.logger.warning("Failed to generate questions from answer")
            return 0.0
        
        self.logger.debug(f"Generated {len(generated_questions)} questions from answer")
        
        # Step 2: Calculate semantic similarity
        similarity_score = await self._calculate_similarity(
            query, 
            generated_questions
        )
        
        self.logger.debug(f"Answer relevancy: {similarity_score:.3f}")
        
        return similarity_score
    
    def _generate_questions(self, answer: str) -> List[str]:
        """
        Generate questions that the answer could address.
        
        Uses LLM to create plausible questions based on the answer content.
        
        Args:
            answer: Generated answer
            
        Returns:
            List of generated questions
        """
        prompt = f"""Given the following answer, generate {self.num_questions} different questions that this answer could be addressing.
                The questions should be specific and directly related to the information in the answer.

                Answer:
                {answer}

                Return ONLY a JSON array of questions, like this:
                ["question 1?", "question 2?", "question 3?"]

                Do not include any other text or explanation. Just the JSON array.
                """
        
        try:
            response = self.llm_client.generate_text(
                prompt=prompt,
                max_output_tokens=300,
                temperature=0.3 # Slight temperature for diversity
            )

            self.logger.info(f"Promopt sent for question generation. {prompt}")

            self.logger.info(f"LLM response for question generation: {response}")
            
            # Parse JSON response
            questions = self._parse_json_response(response)

            self.logger.info(f"Parsed questions: {questions}")

            # Filter and validate questions
            valid_questions = [
                q for q in questions 
                if isinstance(q, str) and len(q) > 10 and '?' in q
            ]

            self.logger.info(f"Valid questions: {valid_questions}")
            
            return valid_questions[:self.num_questions]
            
        except Exception as e:
            self.logger.error(f"Failed to generate questions: {e}")
            return []
    
    async def _calculate_similarity(
        self, 
        original_query: str, 
        generated_questions: List[str]
    ) -> float:
        """
        Calculate semantic similarity between original query and generated questions.
        
        Uses cosine similarity of embeddings.
        
        Args:
            original_query: The original user query
            generated_questions: Questions generated from the answer
            
        Returns:
            Average similarity score (0.0 to 1.0)
        """
        try:
            # Get embedding for original query
            query_embedding = self.embedding_client.embed_text(
                text=[original_query],
                document_type = 'query'
            )
            query_vector = np.array(query_embedding[0])
            
            # Get embeddings for generated questions
            question_embeddings = self.embedding_client.embed_text(
                text=generated_questions,
                document_type = 'query'
            )
            
            # Calculate cosine similarities
            similarities = []
            for q_embedding in question_embeddings:
                q_vector = np.array(q_embedding)
                
                # Cosine similarity
                similarity = self._cosine_similarity(query_vector, q_vector)
                similarities.append(similarity)
            
            # Return average similarity
            avg_similarity = np.mean(similarities) if similarities else 0.0
            
            return float(avg_similarity)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate similarity: {e}")
            return 0.0
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity (-1.0 to 1.0, normalized to 0.0 to 1.0)
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cosine_sim = dot_product / (norm1 * norm2)
        
        # Normalize from [-1, 1] to [0, 1]
        normalized_sim = (cosine_sim + 1) / 2
        
        return float(normalized_sim)
    
    def _parse_json_response(self, response: str) -> List[str]:
        """
        Parse JSON array from LLM response.
        
        Handles cases where LLM adds extra text or formatting.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Parsed list of questions
        """
        try:
            # Try direct JSON parsing
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON array from text
        try:
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception:
            pass
        
        # Fallback: extract questions by looking for question marks
        lines = response.strip().split('\n')
        questions = []
        
        for line in lines:
            line = line.strip()
            
            # Remove common prefixes
            line = re.sub(r'^[-*•\d.)\]]+\s*', '', line)
            line = re.sub(r'^["\']|["\']$', '', line)
            
            # Check if it looks like a question
            if '?' in line and len(line) > 10:
                questions.append(line)
        
        return questions