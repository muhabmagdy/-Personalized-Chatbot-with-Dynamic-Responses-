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
from rag.google_search_retriever import GoogleSearchRetriever
from rag.conversation_memory import ConversationBufferMemory
from llm.llm_generator import LLMGenerator

load_dotenv()


def should_use_web_search(query: str) -> bool:
    """Determine if query needs web search based on keywords."""
    web_search_indicators = [
        "latest", "recent", "current", "news", "today",
        "2024", "2025", "update", "new", "now",
        "what's happening", "trending"
    ]
    query_lower = query.lower()
    return any(indicator in query_lower for indicator in web_search_indicators)


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
            reranker = Reranker(
                model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                device="cpu"
            )
            print("✓ Reranker loaded successfully")
        except Exception as e:
            print(f"⚠ Reranker failed to load: {e}")
            print("⚠ Continuing without reranking...")
            reranker = None

        print("Initializing Fusion Retriever...")
        fusion_retriever = FusionRetriever(k=60)

        print("Initializing Google Search...")
        try:
            google_retriever = GoogleSearchRetriever(num_results=3)
            print("✓ Google Search enabled")
        except ValueError as e:
            google_retriever = None
            print(f"⚠ {e}")
            print("⚠ Web search disabled")

        print("Initializing LLM...")
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_key:
            raise RuntimeError(
                "GEMINI_API_KEY not found in environment variables. "
                "Please add it to your .env file."
            )
        generator = LLMGenerator(api_key=gemini_key)

        print("Initializing Conversation Memory...")
        try:
            memory = ConversationBufferMemory(window_size=3)
            print("✓ Conversation memory initialized")
        except Exception as e:
            print(f"⚠ Memory initialization failed: {e}")
            memory = None

        print("\n" + "="*80)
        print("=== Python RAG + Web Search Chatbot ===")
        print("="*80)
        print("Type 'exit' or 'quit' to stop, or press Ctrl+C")
        print("Features: Auto-Merging | Reranking | Fusion | Memory | Web Search")
        print("="*80 + "\n")

        while True:
            user_question = input("You: ").strip()
            
            if user_question.lower() in ["exit", "quit"]:
                print("Exiting...")
                break
            
            if not user_question:
                continue

            print("\n[1/5] Retrieving conversation memory...")
            memory_context = ""
            if memory:
                try:
                    memory_context = memory.get_memory()
                    if memory_context:
                        print(f"    ✓ Retrieved conversation history ({memory.get_exchange_count()} exchanges)")
                except Exception as mem_error:
                    print(f"    ⚠ Memory retrieval warning: {mem_error}")

            print("[2/5] Retrieving RAG documents...")
            merged_docs = query_engine.query(user_question)
            rag_texts = [d.page_content for d in merged_docs[:10]]
            use_web_search = should_use_web_search(user_question)
            web_texts = []
            
            if use_web_search and google_retriever:
                print("[3/5] Searching the web...")
                try:
                    web_docs = google_retriever.retrieve(user_question)
                    web_texts = [doc.page_content for doc in web_docs]
                    print(f"    ✓ Found {len(web_texts)} web results")
                except Exception as e:
                    print(f"    ⚠ Web search failed: {e}")
            else:
                print("[3/5] Skipping web search (not needed)")

            print("[4/5] Applying Fusion...")
            retrieval_results = [rag_texts, web_texts]
            if memory_context:
                retrieval_results.insert(0, [memory_context])
            retrieval_results = [r for r in retrieval_results if r]
            fused_texts = fusion_retriever.fuse(retrieval_results, top_k=10)

            if reranker:
                print("[5/5] Reranking results...")
                reranked = reranker.rerank(user_question, fused_texts, top_k=5)
                retrieved_texts = [doc for doc, score in reranked]
            else:
                print("[5/5] Using top results (reranker unavailable)...")
                retrieved_texts = fused_texts[:5]

            print("Generating answer...\n")

            if memory_context:
                context_with_memory = [f"[Conversation History]\n{memory_context}"] + retrieved_texts
                answer = generator.answer_generation(user_question, context_with_memory)
            else:
                answer = generator.answer_generation(user_question, retrieved_texts)

            print("\n" + "="*80)
            print("AI:", answer)
            print("="*80 + "\n")

            if memory:
                try:
                    memory.add_exchange(user_question, answer)
                    print("✓ Conversation saved to memory")
                except Exception as mem_error:
                    print(f"⚠ Failed to save to memory: {mem_error}")

    except KeyboardInterrupt:
        print("\n\nExiting gracefully... Goodbye!")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()