from langchain_community.document_loaders import TextLoader

class DataLoader:
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def load(self):
        documents = []
        for file in self.file_paths:
            loader = TextLoader(file, encoding="utf-8")
            docs = loader.load()  
            documents.extend(docs)
        print(f"Loaded {len(documents)} text documents from {len(self.file_paths)} files.")
        return documents