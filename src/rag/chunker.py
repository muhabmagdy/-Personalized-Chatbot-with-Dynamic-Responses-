from langchain_text_splitters import RecursiveCharacterTextSplitter

class TextChunker:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, documents):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunks = splitter.split_documents(documents)  
        print(f"Created {len(chunks)} chunks.")
        return chunks