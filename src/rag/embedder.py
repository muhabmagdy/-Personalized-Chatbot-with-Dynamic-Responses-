from langchain_community.embeddings import HuggingFaceEmbeddings

class Embedder:
    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self.model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"}  
        )
        print(f"Embedding model '{model_name}' loaded on CPU.")

    def embed_query(self, text):
        return self.model.embed_query(text)

    def get_model(self):
        return self.model