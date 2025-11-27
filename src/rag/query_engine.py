class QueryEngine:
    def __init__(self, retriever):
        self.retriever = retriever

    def query(self, query_text):
        
        if hasattr(self.retriever, "retrieve"):
            results = self.retriever.retrieve(query_text)
        else:
            
            try:
                results = self.retriever.get_relevant_documents(query_text)
            except TypeError:
                results = self.retriever.get_relevant_documents(query=query_text)

        
        print("\nTop retrieved (merged) documents:\n")
        for i, doc in enumerate(results[:3]):
            preview = doc.page_content[:500].replace("\n", " ")
            print(f"--- Document {i+1} (source={doc.metadata.get('source') if doc.metadata else 'N/A'}) ---")
            print(preview, "\n")

        return results