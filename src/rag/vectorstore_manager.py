from langchain_community.vectorstores import Qdrant

class VectorStoreManager:
    def __init__(self, embedding_model, location=":memory:", collection_name="books_db"):
        self.embedding_model = embedding_model
        self.location = location
        self.collection_name = collection_name
        self.qdrant = None

    def create_store(self, chunks):
        self.qdrant = Qdrant.from_documents(
            documents=chunks,
            embedding=self.embedding_model,
            location=self.location,
            collection_name=self.collection_name,
        )
        print(f"Qdrant vector store '{self.collection_name}' created successfully.")
        return self.qdrant

    def get_retriever(self, top_k=5):
        if self.qdrant is None:
            raise ValueError("Qdrant store not initialized.")
        return self.qdrant.as_retriever(search_kwargs={"k": top_k})