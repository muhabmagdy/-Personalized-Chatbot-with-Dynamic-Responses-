from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


class IntelligentMemory:
    def __init__(
        self,
        collection_name="long_term_memory",
        path="./memory_qdrant_db",
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        top_k=5
    ):
        self.collection_name = collection_name
        self.top_k = top_k

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": "cpu"}
        )

        self.client = QdrantClient(path=path)

        # Create collection if it doesn't exist
        self._ensure_collection_exists()

        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings
        )

        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.top_k}
        )

    def _ensure_collection_exists(self):
        """Create the collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=384,
                    distance=Distance.COSINE
                )
            )
            print(f"Created new collection: {self.collection_name}")

    def add_memory(self, text: str):
        self.vectorstore.add_texts([text])
        return f"Memory stored: {text}"

    def get_relevant_history(self, query: str):
        docs = self.retriever.invoke(query)
        return "\n".join([doc.page_content for doc in docs])

    def get_retriever(self):
        return self.retriever
    
    # Alias for backwards compatibility
    def get_memory_retriever(self):
        return self.retriever