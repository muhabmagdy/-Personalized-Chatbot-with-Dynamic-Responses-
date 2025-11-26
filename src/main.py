import os
from dotenv import load_dotenv
from rag.data_loader import DataLoader
from rag.chunker import TextChunker
from rag.embedder import Embedder
from rag.vectorstore_manager import VectorStoreManager
from rag.query_engine import QueryEngine
from rag.auto_merging_retriever import AutoMergingRetriever
from rag.rerank import Reranker
from rag.FusionRetriever import FusionRetriever
from llm.llm_generator import LLMGenerator
from rag.intelligent_memory import IntelligentMemory


load_dotenv()


def main():
    try:
      
        text_files = [
            "-Personalized-Chatbot-with-Dynamic-Responses-/src/data/cleaned_texts/automate-the-boring-stuff-with-python-3rd-edition-early-access-3nbsped-9781718503403-9781718503410_com.txt",
            "-Personalized-Chatbot-with-Dynamic-Responses-/src/data/cleaned_texts/fluent-python-2nbsped-9781492056348-9781492056287.txt",
            "-Personalized-Chatbot-with-Dynamic-Responses-/src/data/cleaned_texts/learning-python-powerful-object-oriented-programming-6nbsped-1098171306-9781098171308.txt",
            "-Personalized-Chatbot-with-Dynamic-Responses-/src/data/cleaned_texts/leetcode-python.txt",
            "-Personalized-Chatbot-with-Dynamic-Responses-/src/data/cleaned_texts/python-crash-course-3nbsped-1718502702-9781718502703_compress.txt"
        ]

        print("Loading documents...")
        loader = DataLoader(text_files)
        documents = loader.load()

        print("Chunking documents...")
        chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
        chunks = chunker.split(documents)

        print("Creating embeddings...")
        embedder = Embedder()
        embedding_model = embedder.get_model()

        print("Building vector store...")
        vector_manager = VectorStoreManager(embedding_model)
        qdrant_store = vector_manager.create_store(chunks)


        base_retriever = vector_manager.get_retriever(top_k=50)
        auto_retriever = AutoMergingRetriever(
            base_retriever=base_retriever,
            merge_char_limit=1500,
            max_chunks_per_merge=6,
            top_k=50
        )
        query_engine = QueryEngine(auto_retriever)

        print("Initializing Reranker...")
        try:

            reranker = Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
            print("✓ Reranker loaded successfully")
        except Exception as e:
            print(f"⚠ Reranker failed to load: {e}")
            print("⚠ Continuing without reranking...")
            reranker = None
        
        print("Initializing Fusion Retriever...")
        fusion_retriever = FusionRetriever(k=60)

        print("Initializing LLM...")
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_key:
            raise RuntimeError("GEMINI_API_KEY not found in environment variables. Please add it to your .env file.")
        generator = LLMGenerator(api_key=gemini_key)

        print("Initializing Intelligent Memory...")
        memory_system = IntelligentMemory(
            collection_name="user_memory",
            path="./user_memory_qdrant_db"
        )
        memory = memory_system.get_retriever()

        print("\n=== Python RAG + Memory Chatbot ===")
        print("Type 'exit' or 'quit' to stop, or press Ctrl+C")
        print("Features: Auto-Merging | Reranking | Fusion | Memory\n")

        while True:
            user_question = input("You: ").strip()
            
            if user_question.lower() in ["exit", "quit"]:
                print("Exiting...")
                break
                
            if not user_question:
                continue

            print("\nRetrieving long-term memory...")
            try:
                memory_docs = memory.invoke(user_question)
                memory_texts = [doc.page_content for doc in memory_docs]
            except Exception as mem_error:
                print(f"Memory retrieval warning: {mem_error}")
                memory_texts = []

            print("Retrieving RAG documents...")
            merged_docs = query_engine.query(user_question)
            rag_texts = [d.page_content for d in merged_docs[:10]]
            
            print("Applying Fusion...")
            retrieval_results = [memory_texts, rag_texts]
            fused_texts = fusion_retriever.fuse(retrieval_results, top_k=10)
            
            if reranker:
                print("Reranking results...")
                reranked = reranker.rerank(user_question, fused_texts, top_k=5)
                retrieved_texts = [doc for doc, score in reranked]
            else:
                print("Using top results (reranker unavailable)...")
                retrieved_texts = fused_texts[:5]

            print("Generating answer...")
            answer = generator.answer_generation(user_question, retrieved_texts)
            
            print("\n" + "-"*80)
            print("AI:", answer)
            print("-"*80 + "\n")


            try:
                memory_system.add_memory(f"Q: {user_question}\nA: {answer[:500]}")
            except Exception as mem_error:
                print(f"Failed to save to memory: {mem_error}")

    except KeyboardInterrupt:
        print("\n\nExiting gracefully... Goodbye!")
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()