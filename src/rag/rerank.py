from sentence_transformers import CrossEncoder
from typing import List

class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu", max_length=512):
        try:
            self.model = CrossEncoder(
                model_name_or_path=model_name,
                device=device,
                max_length=max_length
            )
            self.enabled = True
        except Exception as e:
            print(f"Failed to load reranker: {e}")
            self.model = None
            self.enabled = False

    def rerank(self, query: str, docs: List[str], top_k: int = 5):
        if not self.enabled or not self.model:
            return [(doc, 1.0) for doc in docs[:top_k]]
        
        pairs = [(query, doc) for doc in docs]
        scores = self.model.predict(pairs, batch_size=16)
        scored = list(zip(docs, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]