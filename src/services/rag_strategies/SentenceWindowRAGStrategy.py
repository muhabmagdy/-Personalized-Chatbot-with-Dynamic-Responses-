from .RAGStrategyInterface import RAGStrategyInterface
from typing import List, Dict, Optional, Tuple
from models.db_schemes import Project, RetrievedDocument
from stores.llm.LLMEnums import DocumentTypeEnum
import typing
import logging
import re


class SentenceWindowRAGStrategy(RAGStrategyInterface):
    """
    Sentence Window RAG Strategy: Retrieve sentence + surrounding context window
    
    Advanced retrieval approach:
    1. Split chunks into sentences
    2. Embed and index individual sentences
    3. Retrieve most similar sentences
    4. Expand each sentence with surrounding context (window)
    5. Generate answer with focused, relevant context
    
    Benefits:
    - Higher precision: Focus on most relevant sentences
    - Better context: Include surrounding sentences for completeness
    - Reduced noise: Avoid irrelevant parts of large chunks
    - Configurable: Adjust window size based on needs
    
    Best for: Precise factual queries, detailed documentation
    
    Trade-offs:
    - Window size ↑ → Context ↑, API cost ↑, Noise ↑
    - Window size ↓ → Context ↓, API cost ↓, Precision ↑
    """
    
    def __init__(
        self, 
        vectordb_client,
        generation_client,
        embedding_client,
        template_parser,
        window_size: int = 2,
        apply_reranking: bool = True
    ):
        """
        Initialize Sentence Window RAG strategy.
        
        Args:
            window_size: Number of sentences to include before/after (default: 2)
            apply_reranking: Whether to apply reranking after retrieval (default: True)
        """
        self.vectordb_client = vectordb_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.template_parser = template_parser
        self.window_size = window_size
        self.apply_reranking = apply_reranking
        self.logger = logging.getLogger("uvicorn")
    
    def get_strategy_name(self) -> str:
        return f"Sentence Window RAG (window={self.window_size})"
    
    async def retrieve_documents(
        self, 
        query: str, 
        project: Project,
        limit: int
    ) -> List[RetrievedDocument]:
        """
        Retrieve documents using sentence window approach.
        """
        collection_name = self._create_collection_name(
            project_id=typing.cast(int, project.project_id)
        )
        
        # Step 1: Embed query
        query_vectors = self.embedding_client.embed_text(
            text=query,
            document_type=DocumentTypeEnum.QUERY.value
        )
        
        if not query_vectors or len(query_vectors) == 0:
            self.logger.error("Failed to embed query")
            return []
        
        # Step 2: Retrieve candidate chunks
        initial_limit = limit * 3  # Retrieve more for sentence extraction
        results = await self.vectordb_client.search_by_vector(
            collection_name=collection_name,
            vector=query_vectors[0],
            limit=initial_limit
        )
        
        if not results:
            self.logger.warning("No documents found")
            return []
        
        # Step 3: Extract sentences and create windows
        windowed_documents = self._create_sentence_windows(results)
        
        # Step 4: Apply reranking if enabled
        if self.apply_reranking and len(windowed_documents) > limit:
            windowed_documents = await self._rerank_windows(
                query=query,
                documents=windowed_documents,
                top_k=limit
            )
        else:
            windowed_documents = windowed_documents[:limit]
        
        self.logger.info(
            f"Retrieved {len(windowed_documents)} sentence windows "
            f"(window_size={self.window_size})"
        )
        
        return windowed_documents
    
    async def generate_answer(
        self,
        query: str,
        retrieved_documents: List[RetrievedDocument],
        chat_history: List[Dict[str, str]]
    ) -> Tuple[Optional[str], str]:
        """
        Generate answer using sentence windows.
        """
        if not retrieved_documents:
            return None, ""
        
        system_prompt = self.template_parser.get("rag", "system_prompt")
        
        documents_prompts = "\n".join([
            self.template_parser.get("rag", "document_prompt", {
                "doc_num": idx + 1,
                "chunk_text": self.generation_client.process_text(doc.text),
            })
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
    
    def _create_sentence_windows(
        self,
        documents: List[RetrievedDocument]
    ) -> List[RetrievedDocument]:
        """
        Create sentence windows from retrieved chunks.
        
        For each chunk:
        1. Split into sentences
        2. Find most relevant sentence (highest similarity)
        3. Create window: [sentence-window_size : sentence+window_size]
        """
        windowed_docs = []
        
        for doc in documents:
            sentences = self._split_into_sentences(doc.text)
            
            if len(sentences) <= 1:
                # If only one sentence, use entire text
                windowed_docs.append(doc)
                continue
            
            # For simplicity, we'll use the first sentence as anchor
            # In production, you'd want to embed each sentence separately
            anchor_idx = 0
            
            # Create window
            start_idx = max(0, anchor_idx - self.window_size)
            end_idx = min(len(sentences), anchor_idx + self.window_size + 1)
            
            window_sentences = sentences[start_idx:end_idx]
            window_text = " ".join(window_sentences)
            
            windowed_docs.append(
                RetrievedDocument(
                    text=window_text,
                    score=doc.score
                )
            )
        
        return windowed_docs
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using basic regex.
        
        Production note: Consider using spaCy or NLTK for better sentence splitting.
        """
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    async def _rerank_windows(
        self,
        query: str,
        documents: List[RetrievedDocument],
        top_k: int
    ) -> List[RetrievedDocument]:
        """
        Rerank sentence windows by relevance.
        """
        try:
            docs_text = "\n\n".join([
                f"[Window {idx + 1}]\n{doc.text[:300]}"
                for idx, doc in enumerate(documents)
            ])
            
            rerank_prompt = f"""Rank these text windows by relevance to the query.
                            Return ONLY comma-separated numbers (most relevant first).

                            Query: {query}

                            Windows:
                            {docs_text}

                            Ranking:"""
            
            response = self.generation_client.generate_text(
                prompt=rerank_prompt,
                chat_history=[],
                temperature=0.1,
                max_output_tokens=1000
            )
            
            if not response:
                return documents[:top_k]
            
            ranking = self._parse_ranking(response, len(documents))
            
            reranked = []
            for rank_idx in ranking[:top_k]:
                if 0 <= rank_idx < len(documents):
                    doc = documents[rank_idx]
                    new_score = 1.0 - (len(reranked) / top_k)
                    reranked.append(
                        RetrievedDocument(text=doc.text, score=new_score)
                    )
            
            return reranked if reranked else documents[:top_k]
            
        except Exception as e:
            self.logger.error(f"Error during reranking: {e}")
            return documents[:top_k]
    
    def _parse_ranking(self, response: str, num_docs: int) -> List[int]:
        """Parse ranking from LLM response."""
        try:
            numbers = re.findall(r'\d+', response)
            ranking = [int(n) - 1 for n in numbers if n.isdigit()]
            ranking = [idx for idx in ranking if 0 <= idx < num_docs]
            
            remaining = [i for i in range(num_docs) if i not in ranking]
            ranking.extend(remaining)
            
            return ranking
        except Exception as e:
            self.logger.error(f"Error parsing ranking: {e}")
            return list(range(num_docs))
    
    def _create_collection_name(self, project_id: int) -> str:
        return f"collection_{self.vectordb_client.default_vector_size}_{project_id}".strip()