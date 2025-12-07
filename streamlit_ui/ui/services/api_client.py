"""
API Client Service
Handles all communication with the RAG backend API
Supports all 6 RAG strategies and comprehensive evaluation
"""

import httpx
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class APIClient:
    """Client for communicating with RAG system API."""
    
    def __init__(self, base_url: str, timeout: float = 30.0):
        """
        Initialize API client.
        
        Args:
            base_url: Base URL of the API (e.g., "http://localhost:8000")
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        logger.info(f"API Client initialized with base URL: {self.base_url}")
    
    async def get_available_strategies(self) -> List[Dict[str, Any]]:
        """
        Get list of available RAG strategies with full information.
        
        Returns:
            List of strategy dictionaries with type, name, description, use_cases, requirements
            
        Example:
            [
                {
                    "type": "basic",
                    "name": "Basic RAG",
                    "description": "Fast, simple retrieval",
                    "use_cases": ["Simple Q&A", "Fast responses"],
                    "requirements": {"api_keys": [], "complexity": "low"}
                },
                ...
            ]
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.base_url}/api/v1/nlp/strategies/info"
                )
                response.raise_for_status()
                data = response.json()
                
                strategies = data.get("strategies", [])
                logger.info(f"Fetched {len(strategies)} strategies from API")
                return strategies
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching strategies: {e.response.status_code}")
            return self._get_default_strategies()
        except httpx.RequestError as e:
            logger.error(f"Request error fetching strategies: {e}")
            return self._get_default_strategies()
        except Exception as e:
            logger.error(f"Unexpected error fetching strategies: {e}")
            return self._get_default_strategies()
    
    async def get_strategy_recommendations(
        self,
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Get AI-powered strategy recommendations for a query.
        
        Args:
            query: User's question
            
        Returns:
            List of recommended strategies with scores and reasoning
            
        Example:
            [
                {
                    "strategy": "web_search",
                    "score": 0.95,
                    "reasoning": "Recent information needed",
                    "info": {...}
                },
                ...
            ]
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/v1/nlp/strategies/recommend",
                    params={"query": query}
                )
                response.raise_for_status()
                data = response.json()
                
                recommendations = data.get("recommendations", [])
                logger.info(f"Got {len(recommendations)} strategy recommendations")
                return recommendations
                
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return []
    
    async def send_message(
        self,
        project_id: int,
        text: str,
        session_id: str,
        rag_type: str = "basic",
        limit: int = 3,
        chat_history_limit: int = 3,
        evaluate: bool = False,
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send a message and get RAG-based response with optional evaluation.
        
        Args:
            project_id: Project identifier
            text: User's message
            session_id: Chat session ID
            rag_type: RAG strategy (basic/fusion/rerank/sentence_window/auto_merging/web_search)
            limit: Number of documents to retrieve
            chat_history_limit: Max chat history messages to include
            evaluate: Enable comprehensive RAG evaluation
            ground_truth: Optional reference answer for evaluation
            
        Returns:
            Response dictionary with answer, strategy, evaluation (if enabled)
            
        Raises:
            APIError: If request fails
            
        Example Response:
            {
                "signal": "RAG_ANSWER_SUCCESS",
                "answer": "Python is a programming language...",
                "session_id": "abc-123",
                "rag_strategy": "Sentence Window RAG",
                "rag_type": "sentence_window",
                "chat_history_length": 2,
                "evaluation": {  # if evaluate=true
                    "overall_score": 0.85,
                    "metrics": {...},
                    "summary": "..."
                }
            }
        """
        try:
            payload = {
                "text": text,
                "session_id": session_id,
                "rag_type": rag_type,
                "limit": limit,
                "chat_history_limit": chat_history_limit,
                "evaluate": evaluate
            }
            
            if ground_truth:
                payload["ground_truth"] = ground_truth
            
            logger.info(
                f"Sending message to API: project={project_id}, "
                f"rag_type={rag_type}, evaluate={evaluate}"
            )
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/v1/nlp/index/answer/{project_id}",
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
                
                logger.info(f"Received response: {data.get('signal', 'Unknown')}")
                return data
                
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}: {e.response.text}"
            logger.error(f"HTTP error: {error_msg}")
            raise APIError(f"Request failed with status {e.response.status_code}")
        except httpx.RequestError as e:
            logger.error(f"Request error: {e}")
            raise APIError(f"Connection error: Cannot reach backend at {self.base_url}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise APIError(f"Unexpected error: {str(e)}")
    
    async def evaluate_response(
        self,
        project_id: int,
        query: str,
        answer: str,
        retrieved_documents: List[str],
        strategy_name: Optional[str] = None,
        ground_truth: Optional[str] = None,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate an existing RAG response.
        
        Args:
            project_id: Project identifier
            query: Original query
            answer: Generated answer
            retrieved_documents: Retrieved document texts
            strategy_name: Strategy used
            ground_truth: Optional reference answer
            metrics: Specific metrics to evaluate (None = all)
            
        Returns:
            Evaluation report with all metrics
            
        Example Response:
            {
                "signal": "EVALUATION_SUCCESS",
                "evaluation": {
                    "overall_score": 0.85,
                    "metrics": {
                        "answer_relevance": {...},
                        "context_relevance": {...},
                        "groundedness": {...},
                        ...
                    },
                    "summary": "..."
                }
            }
        """
        try:
            payload = {
                "query": query,
                "answer": answer,
                "retrieved_documents": retrieved_documents,
                "strategy_name": strategy_name,
                "ground_truth": ground_truth,
                "metrics": metrics
            }
            
            logger.info(f"Evaluating response for project {project_id}")
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/v1/nlp/evaluate/response/{project_id}",
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
                
                logger.info("Evaluation completed successfully")
                return data
                
        except Exception as e:
            logger.error(f"Error evaluating response: {e}")
            raise APIError(f"Evaluation failed: {str(e)}")
    
    async def batch_evaluate(
        self,
        project_id: int,
        evaluation_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Batch evaluate multiple RAG responses.
        
        Args:
            project_id: Project identifier
            evaluation_data: List of evaluation data dictionaries
            
        Returns:
            Batch evaluation results with aggregate metrics
        """
        try:
            payload = {
                "evaluation_data": evaluation_data
            }
            
            logger.info(f"Batch evaluating {len(evaluation_data)} responses")
            
            async with httpx.AsyncClient(timeout=self.timeout * 2) as client:
                response = await client.post(
                    f"{self.base_url}/api/v1/nlp/evaluate/batch/{project_id}",
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
                
                logger.info("Batch evaluation completed")
                return data
                
        except Exception as e:
            logger.error(f"Error in batch evaluation: {e}")
            raise APIError(f"Batch evaluation failed: {str(e)}")
    
    async def get_chat_sessions(
        self,
        project_id: int,
        limit: int = 10
    ) -> List[str]:
        """
        Get list of recent chat sessions.
        
        Args:
            project_id: Project identifier
            limit: Maximum sessions to return
            
        Returns:
            List of session IDs
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.base_url}/api/v1/nlp/chat/sessions/{project_id}",
                    params={"limit": limit}
                )
                response.raise_for_status()
                data = response.json()
                
                sessions = data.get("sessions", [])
                logger.info(f"Fetched {len(sessions)} sessions for project {project_id}")
                return sessions
                
        except Exception as e:
            logger.error(f"Error fetching sessions: {e}")
            return []
    
    async def clear_chat_session(
        self,
        project_id: int,
        session_id: str
    ) -> bool:
        """
        Clear a chat session history.
        
        Args:
            project_id: Project identifier
            session_id: Session to clear
            
        Returns:
            True if successful, False otherwise
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.delete(
                    f"{self.base_url}/api/v1/nlp/chat/session/{project_id}/{session_id}"
                )
                response.raise_for_status()
                
                logger.info(f"Cleared session {session_id} for project {project_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error clearing session: {e}")
            return False
    
    async def health_check(self) -> bool:
        """
        Check if the API is reachable.
        
        Returns:
            True if API is healthy, False otherwise
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/welcome")
                return response.status_code == 200
        except Exception:
            return False
    
    def _get_default_strategies(self) -> List[Dict[str, Any]]:
        """
        Fallback strategies if API is unavailable.
        
        Returns:
            List of default strategy configurations
        """
        logger.warning("Using default strategies (API unavailable)")
        return [
            {
                "type": "basic",
                "description": "Simple vector similarity search - Fast and reliable",
                "use_cases": ["Simple Q&A", "Well-defined queries", "Fast responses needed"],
                "requirements": {"api_keys": [], "complexity": "low", "avg_latency_ms": 200}
            },
            {
                "type": "fusion",
                "description": "Query expansion with result fusion - Better for complex queries",
                "use_cases": ["Complex queries", "Multi-perspective questions"],
                "requirements": {"api_keys": [], "complexity": "medium", "avg_latency_ms": 500}
            },
            {
                "type": "rerank",
                "description": "Two-stage retrieval with reranking - High precision",
                "use_cases": ["High precision required", "Complex technical documentation"],
                "requirements": {"api_keys": [], "complexity": "medium", "avg_latency_ms": 600}
            },
            {
                "type": "sentence_window",
                "description": "Sentence-level retrieval with context window - Focused and precise",
                "use_cases": ["Precise factual queries", "Detailed documentation"],
                "requirements": {"api_keys": [], "complexity": "medium", "avg_latency_ms": 400}
            },
            {
                "type": "auto_merging",
                "description": "Hierarchical retrieval with smart merging - Maintains document structure",
                "use_cases": ["Structured content", "Technical documentation"],
                "requirements": {"api_keys": [], "complexity": "high", "avg_latency_ms": 450}
            },
            {
                "type": "web_search",
                "description": "Hybrid vector + web search - Access to real-time information",
                "use_cases": ["Current events", "Recent information", "Fact verification"],
                "requirements": {
                    "api_keys": ["tavily_api_key"],
                    "complexity": "medium",
                    "avg_latency_ms": 800,
                    "external_dependencies": ["Tavily API"]
                }
            }
        ]


class APIError(Exception):
    """Custom exception for API errors."""
    pass