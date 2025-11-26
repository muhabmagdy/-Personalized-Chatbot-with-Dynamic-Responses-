from typing import List, Dict, Any
from langchain_core.documents import Document

LC_Document = Document


class AutoMergingRetriever:
    """
    Wraps a LangChain retriever and merges related chunks into larger documents.
    Compatible with VectorStoreRetriever from recent LangChain versions.

    Args:
        base_retriever: LangChain retriever (VectorStoreRetriever or custom)
        merge_char_limit: max characters for a merged output unit (approx)
        max_chunks_per_merge: fallback limit for number of chunks to merge
        top_k: how many candidate chunks to fetch
    """

    def __init__(
        self,
        base_retriever,
        merge_char_limit: int = 1500,
        max_chunks_per_merge: int = 8,
        top_k: int = 50,
    ):
        self.base_retriever = base_retriever
        self.merge_char_limit = merge_char_limit
        self.max_chunks_per_merge = max_chunks_per_merge
        self.top_k = top_k

    def _group_by_source(self, docs: List[Document]) -> Dict[str, List[Document]]:
        groups: Dict[str, List[Document]] = {}
        for doc in docs:
            src = None
            if doc.metadata:
                src = doc.metadata.get("source") or doc.metadata.get("file_name") or doc.metadata.get("path")
            if not src:
                src = "unknown_source"
            groups.setdefault(src, []).append(doc)
        return groups

    def _order_docs(self, docs: List[Document]) -> List[Document]:
        def sort_key(doc: Document) -> int:
            md = doc.metadata or {}
            for k in ("chunk", "chunk_index", "page", "page_number", "part"):
                if k in md:
                    try:
                        return int(md[k])
                    except Exception:
                        pass
            return 0

        return sorted(docs, key=sort_key)

    def _fetch_candidates(self, query: str) -> List[Document]:
        """
        Fetch documents from base retriever.
        Supports both old and new LangChain API.
        """
        if hasattr(self.base_retriever, 'invoke'):
            return self.base_retriever.invoke(query)
        
        if hasattr(self.base_retriever, "get_relevant_documents"):
            return self.base_retriever.get_relevant_documents(query)
        
        if hasattr(self.base_retriever, "as_retriever"):
            retriever = self.base_retriever.as_retriever(search_kwargs={"k": self.top_k})
            
            if hasattr(retriever, 'invoke'):
                return retriever.invoke(query)
            else:
                return retriever.get_relevant_documents(query)
        
        raise AttributeError(
            f"The base_retriever {type(self.base_retriever)} does not support "
            "invoke() or get_relevant_documents()"
        )

    def retrieve(self, query: str) -> List[Document]:
        candidates = self._fetch_candidates(query)
        if not isinstance(candidates, list):
            candidates = list(candidates)

        groups = self._group_by_source(candidates)
        merged_documents: List[Document] = []

        for src, docs in groups.items():
            ordered_docs = self._order_docs(docs)
            cur_text_parts: List[str] = []
            cur_members: List[Dict[str, Any]] = []
            cur_len = 0
            cur_chunks_count = 0

            for doc in ordered_docs:
                txt = doc.page_content or ""

                if cur_text_parts and (
                    cur_len + len(txt) > self.merge_char_limit
                    or cur_chunks_count + 1 > self.max_chunks_per_merge
                ):
                    merged_text = "\n\n".join(cur_text_parts).strip()
                    meta = {
                        "source": src,
                        "merged_from_count": len(cur_members),
                        "members": cur_members.copy(),
                    }
                    merged_documents.append(LC_Document(page_content=merged_text, metadata=meta))
                    cur_text_parts = []
                    cur_members = []
                    cur_len = 0
                    cur_chunks_count = 0

                cur_text_parts.append(txt)
                cur_members.append({"id": doc.metadata.get("id") if doc.metadata else None, "meta": doc.metadata})
                cur_len += len(txt)
                cur_chunks_count += 1

            if cur_text_parts:
                merged_text = "\n\n".join(cur_text_parts).strip()
                meta = {
                    "source": src,
                    "merged_from_count": len(cur_members),
                    "members": cur_members.copy(),
                }
                merged_documents.append(LC_Document(page_content=merged_text, metadata=meta))

        return merged_documents