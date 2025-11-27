class RetrieverAdapter:
    def __init__(self, retriever, top_k: int = 5):
        self.retriever = retriever
        self.top_k = top_k

    def get_relevant_documents(self, query, k=None):
        if hasattr(self.retriever, 'invoke'):
            try:
                return self.retriever.invoke(query)
            except Exception as e:
                print(f"invoke() failed: {e}")
        
       
        if hasattr(self.retriever, 'get_relevant_documents'):
            try:
                return self.retriever.get_relevant_documents(query)
            except Exception as e:
                print(f"get_relevant_documents() failed: {e}")
        
      
        if callable(self.retriever):
            try:
                return self.retriever(query)
            except Exception as e:
                print(f"callable retriever failed: {e}")
        
        raise AttributeError(
            f"Retriever {type(self.retriever)} does not support invoke(), "
            )