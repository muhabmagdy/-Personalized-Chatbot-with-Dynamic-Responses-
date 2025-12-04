from .RAGStrategyInterface import RAGStrategyInterface
from typing import List, Dict, Optional, Tuple
from models.db_schemes import Project, RetrievedDocument
from stores.llm.LLMEnums import DocumentTypeEnum
import typing
import logging
import httpx
from datetime import datetime


class WebSearchRAGStrategy(RAGStrategyInterface):
    """
    Web Search RAG Strategy: Hybrid vector search + real-time web search
    
    Advanced hybrid approach:
    1. Search vector database for internal knowledge
    2. Search web (SerpAPI free tier) for recent/external information
    3. Combine and rank results from both sources
    4. Generate answer with enriched context
    
    Benefits:
    - Access to real-time information
    - Combines internal knowledge with external data
    - Handles queries about recent events
    - Verifies and enriches existing knowledge
    
    Best for: Current events, evolving topics, fact verification
    
    Free Tier Options (250 calls/month):
    - SerpAPI: https://serpapi.com (100 searches/month free)
    - Tavily AI: https://tavily.com (1000 searches/month free) ⭐ Recommended
    - SearchAPI: https://www.searchapi.io (100 searches/month free)
    
    This implementation uses Tavily AI (most generous free tier)
    """
    
    def __init__(
        self, 
        vectordb_client,
        generation_client,
        embedding_client,
        template_parser,
        tavily_api_key: Optional[str] = None,
        web_search_count: int = 3,
        vector_search_weight: float = 0.6,
        web_search_weight: float = 0.4
    ):
        """
        Initialize Web Search RAG strategy.
        
        Args:
            tavily_api_key: Tavily API key (get free at https://tavily.com)
            web_search_count: Number of web results to fetch (default: 3)
            vector_search_weight: Weight for vector search results (default: 0.6)
            web_search_weight: Weight for web search results (default: 0.4)
        """
        self.vectordb_client = vectordb_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.template_parser = template_parser
        self.tavily_api_key = tavily_api_key
        self.web_search_count = web_search_count
        self.vector_search_weight = vector_search_weight
        self.web_search_weight = web_search_weight
        self.logger = logging.getLogger("uvicorn")
        
        # Tavily API endpoint
        self.tavily_api_url = "https://api.tavily.com/search"
    
    def get_strategy_name(self) -> str:
        return "Web Search RAG (Hybrid Vector + Web)"
    
    async def retrieve_documents(
        self, 
        query: str, 
        project: Project,
        limit: int
    ) -> List[RetrievedDocument]:
        """
        Retrieve documents from both vector DB and web search.
        """
        # Step 1: Vector database search
        vector_results = await self._vector_search(query, project, limit)
        
        # Step 2: Web search (if API key provided)
        web_results = []
        if self.tavily_api_key:
            web_results = await self._web_search(query)
        else:
            self.logger.warning(
                "Tavily API key not provided. Skipping web search. "
                "Get free key at https://tavily.com"
            )
        
        # Step 3: Combine and rank results
        combined_results = self._combine_results(
            vector_results=vector_results,
            web_results=web_results,
            limit=limit
        )
        
        self.logger.info(
            f"Retrieved {len(vector_results)} vector + {len(web_results)} web results "
            f"→ Combined to {len(combined_results)} total"
        )
        
        return combined_results
    
    async def generate_answer(
        self,
        query: str,
        retrieved_documents: List[RetrievedDocument],
        chat_history: List[Dict[str, str]]
    ) -> Tuple[Optional[str], str]:
        """
        Generate answer with source attribution (internal vs web).
        """
        if not retrieved_documents:
            return None, ""
        
        system_prompt = self.template_parser.get("rag", "system_prompt")
        
        # Enhanced document prompt with source type
        documents_prompts = "\n".join([
            self._format_document_with_source(idx + 1, doc)
            for idx, doc in enumerate(retrieved_documents)
        ])
        
        footer_prompt = self.template_parser.get("rag", "footer_prompt", {
            "query": query
        })
        
        full_prompt = "\n\n".join([documents_prompts, footer_prompt])
        
        final_chat_history = [
            self.generation_client.construct_prompt(
                prompt=system_prompt,
                role=self.generation_client.enums.SYSTEM.value,
            )
        ]
        final_chat_history.extend(chat_history)
        
        answer = self.generation_client.generate_text(
            prompt=full_prompt,
            chat_history=final_chat_history
        )
        
        return answer, full_prompt
    
    async def _vector_search(
        self,
        query: str,
        project: Project,
        limit: int
    ) -> List[RetrievedDocument]:
        """Perform vector database search."""
        try:
            collection_name = self._create_collection_name(
                project_id=typing.cast(int, project.project_id)
            )
            
            query_vectors = self.embedding_client.embed_text(
                text=query,
                document_type=DocumentTypeEnum.QUERY.value
            )
            
            if not query_vectors or len(query_vectors) == 0:
                return []
            
            results = await self.vectordb_client.search_by_vector(
                collection_name=collection_name,
                vector=query_vectors[0],
                limit=limit
            )
            
            return results or []
            
        except Exception as e:
            self.logger.error(f"Vector search error: {e}")
            return []
    
    async def _web_search(self, query: str) -> List[RetrievedDocument]:
        """
        Perform web search using Tavily AI.
        
        Tavily Free Tier: 1000 searches/month
        API Docs: https://docs.tavily.com/docs/tavily-api/introduction
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.tavily_api_url,
                    json={
                        "api_key": self.tavily_api_key,
                        "query": query,
                        "search_depth": "basic",  # Use "basic" for free tier
                        "max_results": self.web_search_count,
                        "include_answer": False,
                        "include_raw_content": False
                    },
                    timeout=10.0
                )
                
                if response.status_code != 200:
                    self.logger.error(
                        f"Tavily API error: {response.status_code} - {response.text}"
                    )
                    return []
                
                data = response.json()
                
                # Parse Tavily response
                web_results = []
                for result in data.get("results", []):
                    text = f"{result.get('title', '')}\n\n{result.get('content', '')}"
                    url = result.get('url', '')
                    
                    # Add source metadata
                    text_with_source = f"[Web Source: {url}]\n{text}"
                    
                    web_results.append(
                        RetrievedDocument(
                            text=text_with_source,
                            score=result.get('score', 0.5)  # Tavily provides relevance score
                        )
                    )
                
                return web_results
                
        except httpx.TimeoutException:
            self.logger.error("Web search timeout")
            return []
        except Exception as e:
            self.logger.error(f"Web search error: {e}")
            return []
    
    def _combine_results(
        self,
        vector_results: List[RetrievedDocument],
        web_results: List[RetrievedDocument],
        limit: int
    ) -> List[RetrievedDocument]:
        """
        Combine vector and web results with weighted scoring.
        """
        # Apply weights to scores
        weighted_vector = [
            RetrievedDocument(
                text=doc.text,
                score=doc.score * self.vector_search_weight
            )
            for doc in vector_results
        ]
        
        weighted_web = [
            RetrievedDocument(
                text=doc.text,
                score=doc.score * self.web_search_weight
            )
            for doc in web_results
        ]
        
        # Combine and sort by weighted score
        combined = weighted_vector + weighted_web
        combined.sort(key=lambda x: x.score, reverse=True)
        
        return combined[:limit]
    
    def _format_document_with_source(
        self,
        doc_num: int,
        doc: RetrievedDocument
    ) -> str:
        """Format document with clear source indication."""
        source_type = "Web" if "[Web Source:" in doc.text else "Internal Knowledge Base"
        
        return f"""Document {doc_num} (Source: {source_type}):
{self.generation_client.process_text(doc.text)}"""
    
    def _create_collection_name(self, project_id: int) -> str:
        return f"collection_{self.vectordb_client.default_vector_size}_{project_id}".strip()


# Alternative: SerpAPI implementation (100 searches/month free)
class SerpAPIWebSearchRAGStrategy(WebSearchRAGStrategy):
    """
    Web Search RAG using SerpAPI (alternative implementation)
    
    SerpAPI Free Tier: 100 searches/month
    Get API key: https://serpapi.com
    """
    
    def __init__(self, serpapi_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.serpapi_key = serpapi_key
        self.serpapi_url = "https://serpapi.com/search"
    
    async def _web_search(self, query: str) -> List[RetrievedDocument]:
        """Web search using SerpAPI."""
        if not self.serpapi_key:
            return []
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    self.serpapi_url,
                    params={
                        "q": query,
                        "api_key": self.serpapi_key,
                        "num": self.web_search_count
                    },
                    timeout=10.0
                )
                
                if response.status_code != 200:
                    return []
                
                data = response.json()
                
                web_results = []
                for result in data.get("organic_results", [])[:self.web_search_count]:
                    text = f"{result.get('title', '')}\n\n{result.get('snippet', '')}"
                    url = result.get('link', '')
                    
                    text_with_source = f"[Web Source: {url}]\n{text}"
                    
                    web_results.append(
                        RetrievedDocument(text=text_with_source, score=0.8)
                    )
                
                return web_results
                
        except Exception as e:
            self.logger.error(f"SerpAPI search error: {e}")
            return []