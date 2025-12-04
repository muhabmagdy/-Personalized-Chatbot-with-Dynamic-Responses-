from .RAGStrategyInterface import RAGStrategyInterface
from typing import List, Dict, Optional, Tuple, Set
from models.db_schemes import Project, RetrievedDocument
from stores.llm.LLMEnums import DocumentTypeEnum
import typing
import logging
from dataclasses import dataclass


@dataclass
class HierarchicalNode:
    """Represents a node in the document hierarchy."""
    node_id: str
    parent_id: Optional[str]
    text: str
    level: int  # 0 = root, 1 = parent, 2 = child
    children_ids: List[str]


class AutoMergingRAGStrategy(RAGStrategyInterface):
    """
    Auto-Merging RAG Strategy: Hierarchical retrieval with automatic merging
    
    Advanced retrieval approach:
    1. Documents are structured hierarchically (Parent → Children nodes)
    2. Retrieve child nodes (small chunks) for precision
    3. If multiple children from same parent are retrieved → auto-merge to parent
    4. Return complete, structured context to LLM
    
    Benefits:
    - Maintains document structure and context
    - Combines precision (child retrieval) with completeness (parent merging)
    - Avoids fragmented information
    - Automatically determines optimal context size
    
    Best for: Technical documentation, structured content, complex queries
    
    Example:
    - Document: "Python Functions Guide"
      - Parent: "Function Definition Section"
        - Child 1: "Basic syntax: def func():"
        - Child 2: "Parameters and arguments"
        - Child 3: "Return values"
    
    If Child 1 and Child 3 are retrieved → Merge to Parent for complete context
    """
    
    def __init__(
        self, 
        vectordb_client,
        generation_client,
        embedding_client,
        template_parser,
        merge_threshold: int = 2,
        similarity_threshold: float = 0.7
    ):
        """
        Initialize Auto-Merging RAG strategy.
        
        Args:
            merge_threshold: Min number of children to trigger merge (default: 2)
            similarity_threshold: Min similarity score to consider (default: 0.7)
        """
        self.vectordb_client = vectordb_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.template_parser = template_parser
        self.merge_threshold = merge_threshold
        self.similarity_threshold = similarity_threshold
        self.logger = logging.getLogger("uvicorn")
        
        # In-memory hierarchy (in production, store in database)
        self.hierarchy: Dict[str, HierarchicalNode] = {}
    
    def get_strategy_name(self) -> str:
        return f"Auto-Merging RAG (threshold={self.merge_threshold})"
    
    async def retrieve_documents(
        self, 
        query: str, 
        project: Project,
        limit: int
    ) -> List[RetrievedDocument]:
        """
        Retrieve documents with auto-merging.
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
        
        # Step 2: Retrieve child nodes (more granular chunks)
        # Retrieve more than needed for merging analysis
        initial_limit = limit * 2
        results = await self.vectordb_client.search_by_vector(
            collection_name=collection_name,
            vector=query_vectors[0],
            limit=initial_limit
        )
        
        if not results:
            self.logger.warning("No documents found")
            return []
        
        # Step 3: Filter by similarity threshold
        filtered_results = [
            doc for doc in results 
            if doc.score >= self.similarity_threshold
        ]
        
        if not filtered_results:
            self.logger.warning("No documents above similarity threshold")
            return results[:limit]  # Fallback to original results
        
        # Step 4: Build hierarchy from retrieved documents
        self._build_hierarchy_from_documents(filtered_results)
        
        # Step 5: Apply auto-merging logic
        merged_documents = self._apply_auto_merging(filtered_results)
        
        # Step 6: Return top K after merging
        final_results = merged_documents[:limit]
        
        self.logger.info(
            f"Retrieved {len(filtered_results)} child nodes → "
            f"Merged to {len(merged_documents)} nodes → "
            f"Returned top {len(final_results)}"
        )
        
        return final_results
    
    async def generate_answer(
        self,
        query: str,
        retrieved_documents: List[RetrievedDocument],
        chat_history: List[Dict[str, str]]
    ) -> Tuple[Optional[str], str]:
        """
        Generate answer using merged documents.
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
    
    def _build_hierarchy_from_documents(
        self,
        documents: List[RetrievedDocument]
    ) -> None:
        """
        Build hierarchical structure from flat documents.
        
        In production, this would be done during indexing.
        Here we simulate by analyzing document structure.
        """
        # Simple heuristic: Group documents by semantic similarity
        # In production, use explicit parent-child relationships
        
        for idx, doc in enumerate(documents):
            node_id = f"child_{idx}"
            parent_id = f"parent_{idx // 3}"  # Simple grouping
            
            # Create child node
            if node_id not in self.hierarchy:
                self.hierarchy[node_id] = HierarchicalNode(
                    node_id=node_id,
                    parent_id=parent_id,
                    text=doc.text,
                    level=2,  # Child level
                    children_ids=[]
                )
            
            # Create/update parent node
            if parent_id not in self.hierarchy:
                self.hierarchy[parent_id] = HierarchicalNode(
                    node_id=parent_id,
                    parent_id=None,
                    text="",  # Will be merged from children
                    level=1,  # Parent level
                    children_ids=[]
                )
            
            # Link parent to child
            if node_id not in self.hierarchy[parent_id].children_ids:
                self.hierarchy[parent_id].children_ids.append(node_id)
    
    def _apply_auto_merging(
        self,
        documents: List[RetrievedDocument]
    ) -> List[RetrievedDocument]:
        """
        Apply auto-merging: Replace multiple children with parent if threshold met.
        """
        merged_docs = []
        processed_parents: Set[str] = set()
        
        # Group documents by parent
        parent_to_children: Dict[str, List[RetrievedDocument]] = {}
        doc_to_node_id: Dict[str, str] = {}
        
        for idx, doc in enumerate(documents):
            node_id = f"child_{idx}"
            doc_to_node_id[doc.text] = node_id
            
            if node_id in self.hierarchy:
                node = self.hierarchy[node_id]
                parent_id = node.parent_id
                
                if parent_id:
                    if parent_id not in parent_to_children:
                        parent_to_children[parent_id] = []
                    parent_to_children[parent_id].append(doc)
        
        # Check merge threshold for each parent
        for parent_id, children_docs in parent_to_children.items():
            if len(children_docs) >= self.merge_threshold:
                # Merge: Create parent document
                if parent_id not in processed_parents:
                    parent_text = self._merge_children_text(children_docs)
                    avg_score = sum(d.score for d in children_docs) / len(children_docs)
                    
                    merged_docs.append(
                        RetrievedDocument(
                            text=parent_text,
                            score=avg_score * 1.1  # Boost merged documents
                        )
                    )
                    processed_parents.add(parent_id)
            else:
                # Don't merge: Keep individual children
                merged_docs.extend(children_docs)
        
        # Sort by score
        merged_docs.sort(key=lambda x: x.score, reverse=True)
        
        return merged_docs
    
    def _merge_children_text(
        self,
        children_docs: List[RetrievedDocument]
    ) -> str:
        """
        Merge children text into cohesive parent text.
        """
        # Sort by score to prioritize most relevant
        sorted_children = sorted(
            children_docs,
            key=lambda x: x.score,
            reverse=True
        )
        
        merged_text = "\n\n".join([doc.text for doc in sorted_children])
        
        # Add header for clarity
        header = f"[Merged context from {len(children_docs)} related sections]\n\n"
        
        return header + merged_text
    
    def _create_collection_name(self, project_id: int) -> str:
        return f"collection_{self.vectordb_client.default_vector_size}_{project_id}".strip()