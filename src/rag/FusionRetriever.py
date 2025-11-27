from typing import List, Dict
from collections import defaultdict

class FusionRetriever:
    def __init__(self, k: int = 60):
        self.k = k  

    def fuse(self, results: List[List[str]], top_k: int = 5):
        """
        results = [
            [doc1, doc2, doc3],      # retriever A
            [doc3, doc1, doc5],      # retriever B
            ...
        ]
        """
        scores = defaultdict(float)

        for retriever_output in results:
            for rank, doc in enumerate(retriever_output):
                scores[doc] += 1 / (self.k + rank + 1)

        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in fused[:top_k]]