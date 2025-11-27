import os
import streamlit as st
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
from evaluation.evaluator import evaluate_answer

load_dotenv()

st.set_page_config(
    page_title="Python RAG + Web Search Chatbot",
    page_icon="🐍",
    layout="wide"
)

st.markdown("""
<style>
.stChat message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    text-align: center;
}
.web-search-indicator {
    background-color: #e3f2fd;
    padding: 0.5rem;
    border-radius: 0.3rem;
    border-left: 3px solid #2196f3;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)


def should_use_web_search(query: str) -> bool:
    """Determine if query needs web search."""
    web_search_indicators = [
        "latest", "recent", "current", "news", "today",
        "2024", "2025", "update", "new", "now",
        "what's happening", "trending"
    ]
    query_lower = query.lower()
    return any(indicator in query_lower for indicator in web_search_indicators)


@st.cache_resource
def initialize_rag_system():
    """Initialize RAG components (cached to avoid reloading)."""
    import os
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    possible_data_dirs = [
        os.path.join(parent_dir, "data", "cleaned_texts"),
        os.path.join(parent_dir, "data"),
        os.path.join(current_dir, "data", "cleaned_texts"),
        os.path.join(current_dir, "data"),
    ]
    
    file_names = [
        "automate-the-boring-stuff-with-python-3rd-edition-early-access-3nbsped-9781718503403-9781718503410_com.txt",
        "fluent-python-2nbsped-9781492056348-9781492056287.txt",
        "learning-python-powerful-object-oriented-programming-6nbsped-1098171306-9781098171308.txt",
        "leetcode-python.txt",
        "python-crash-course-3nbsped-1718502702-9781718502703_compress.txt"
    ]
    
    data_dir = None
    for possible_dir in possible_data_dirs:
        if os.path.exists(possible_dir):
            test_file = os.path.join(possible_dir, file_names[0])
            if os.path.exists(test_file):
                data_dir = possible_dir
                break
    
    if not data_dir:
        raise FileNotFoundError(
            f"Could not find data files. Searched in: {possible_data_dirs}"
        )
    
    text_files = [os.path.join(data_dir, fname) for fname in file_names]
    missing_files = [f for f in text_files if not os.path.exists(f)]
    if missing_files:
        raise FileNotFoundError(f"Missing files: {missing_files}")
    
    loader = DataLoader(text_files)
    documents = loader.load()
    
    chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
    chunks = chunker.split(documents)
    
    embedder = Embedder()
    embedding_model = embedder.get_model()
    
    vector_manager = VectorStoreManager(embedding_model)
    vector_manager.create_store(chunks)
    base_retriever = vector_manager.get_retriever(top_k=50)
    
    auto_retriever = AutoMergingRetriever(
        base_retriever=base_retriever,
        merge_char_limit=1500,
        max_chunks_per_merge=6,
        top_k=50
    )
    query_engine = QueryEngine(auto_retriever)
    
    try:
        reranker = Reranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            device="cpu"
        )
    except Exception as e:
        print(f"Warning: Reranker failed to load: {e}")
        reranker = None
    
    fusion_retriever = FusionRetriever(k=60)
    
    return query_engine, len(chunks), reranker, fusion_retriever


@st.cache_resource
def initialize_llm():
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        return None
    return LLMGenerator(api_key=gemini_key)


@st.cache_resource
def initialize_google_search():
    try:
        return GoogleSearchRetriever(num_results=3)
    except ValueError as e:
        print(f"Failed to initialize Google Search: {e}")
        return None


def main():
    st.title("🐍 Python RAG + Web Search Chatbot")
    st.markdown("*Your AI-powered Python learning assistant with real-time web search*")

    with st.sidebar:
        st.header("⚙️ Settings")

        gemini_key = os.environ.get("GEMINI_API_KEY")
        serpapi_key = os.environ.get("SERPAPI_API_KEY")
        
        if gemini_key:
            st.success("✅ Gemini API Key Loaded")
        else:
            st.error("❌ Gemini API Key Missing")
            st.info("Add GEMINI_API_KEY to your .env file")
        
        if serpapi_key:
            st.success("✅ SerpAPI Key Loaded")
        else:
            st.warning("⚠️ SerpAPI Key Missing")
            st.info("Add SERPAPI_API_KEY to enable web search")
        
        st.divider()
        
        show_sources = st.checkbox("Show retrieved sources", value=False)
        show_metrics = st.checkbox("Show evaluation metrics", value=False)
        
        if 'reranker' in st.session_state and st.session_state.get('reranker') is not None:
            use_reranking = st.checkbox(
                "Enable Reranking",
                value=True,
                help="Use cross-encoder reranking for better results"
            )
        else:
            use_reranking = False
            st.info("ℹ️ Reranker unavailable (model not loaded)")
        
        use_fusion = st.checkbox(
            "Enable Fusion",
            value=True,
            help="Combine multiple retrieval strategies"
        )
        
        auto_web_search = st.checkbox(
            "Auto Web Search",
            value=True,
            help="Automatically search web for recent/current queries"
        )
        
        force_web_search = st.checkbox(
            "Always Use Web Search",
            value=False,
            help="Force web search for every query"
        )
        
        st.divider()

        st.header("🧠 Memory")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Clear Chat"):
                st.session_state.messages = []
                st.rerun()
        
        with col2:
            if st.button("Clear Memory"):
                if 'memory' in st.session_state and st.session_state.memory:
                    st.session_state.memory.clear()
                    st.success("Memory cleared!")
                    st.rerun()

        if 'memory' in st.session_state and st.session_state.memory:
            st.info(f"💬 {st.session_state.memory.get_exchange_count()} exchanges in memory")
        
        st.divider()

        st.header("📊 System Status")

    with st.spinner("Loading RAG system... This may take a moment."):
        try:
            query_engine, num_chunks, reranker, fusion_retriever = initialize_rag_system()
            st.sidebar.success(f"✅ RAG System Ready")
            st.sidebar.info(f"📚 {num_chunks} chunks indexed")
            st.session_state['reranker'] = reranker
            
            if reranker:
                st.sidebar.success("✅ Reranker Ready")
            else:
                st.sidebar.warning("⚠️ Reranker Unavailable")
        except Exception as e:
            st.sidebar.error(f"❌ RAG Error: {str(e)[:50]}")
            query_engine = None
            reranker = None
            fusion_retriever = None
    
    generator = initialize_llm()
    if generator:
        st.sidebar.success("✅ LLM Connected")
    else:
        st.sidebar.warning("⚠️ No API Key - Enter in sidebar")

    if 'memory' not in st.session_state:
        try:
            st.session_state.memory = ConversationBufferMemory(window_size=3)
            st.sidebar.success("✅ Memory System Ready")
        except Exception as e:
            st.sidebar.error(f"❌ Memory Error: {str(e)[:50]}")
            st.session_state.memory = None
    
    memory = st.session_state.memory
    
    google_retriever = initialize_google_search()
    if google_retriever:
        st.sidebar.success("✅ Web Search Ready")
    else:
        st.sidebar.warning("⚠️ Web Search Unavailable")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message.get("used_web_search"):
                st.markdown(
                    '<div class="web-search-indicator">🌐 Used Web Search</div>',
                    unsafe_allow_html=True
                )
            
            if show_sources and "sources" in message:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(message["sources"][:3]):
                        st.markdown(f"**Source {i+1}:**")
                        st.text(source[:500] + "..." if len(source) > 500 else source)
            
            if show_metrics and "metrics" in message:
                with st.expander("📊 Evaluation Metrics"):
                    cols = st.columns(3)
                    metrics = message["metrics"]
                    cols[0].metric("Answer Relevance", f"{metrics.get('answer_relevance', 0):.2f}")
                    cols[1].metric("Context Relevance", f"{metrics.get('context_relevance', 0):.2f}")
                    cols[2].metric("Groundedness", f"{metrics.get('groundedness', 0):.2f}")

    if prompt := st.chat_input("Ask me anything about Python..."):
        if not generator:
            st.error("Please enter your Gemini API key in the sidebar.")
            return
        
        if not query_engine:
            st.error("RAG system failed to initialize. Check your data files.")
            return
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    memory_context = ""
                    if memory:
                        try:
                            memory_context = memory.get_memory()
                        except:
                            pass

                    merged_docs = query_engine.query(prompt)
                    rag_texts = [d.page_content for d in merged_docs[:10]]
 
                    web_texts = []
                    used_web_search = False
                    
                    if google_retriever and (force_web_search or 
                                            (auto_web_search and should_use_web_search(prompt))):
                        with st.status("🌐 Searching the web...", expanded=False) as status:
                            try:
                                web_docs = google_retriever.retrieve(prompt)
                                web_texts = [doc.page_content for doc in web_docs]
                                used_web_search = True
                                status.update(
                                    label=f"✅ Found {len(web_texts)} web results",
                                    state="complete"
                                )
                            except Exception as e:
                                status.update(
                                    label=f"⚠️ Web search failed: {str(e)}",
                                    state="error"
                                )

                    if use_fusion and fusion_retriever:
                        retrieval_results = [rag_texts, web_texts]
                        if memory_context:
                            retrieval_results.insert(0, [memory_context])
                        retrieval_results = [r for r in retrieval_results if r]
                        fused_texts = fusion_retriever.fuse(retrieval_results, top_k=10)
                        retrieved_texts = fused_texts
                    else:
                        retrieved_texts = rag_texts + web_texts
                        if memory_context:
                            retrieved_texts.insert(0, memory_context)

                    if use_reranking and reranker and len(retrieved_texts) > 0:
                        reranked = reranker.rerank(prompt, retrieved_texts, top_k=5)
                        retrieved_texts = [doc for doc, score in reranked]
                    else:
                        retrieved_texts = retrieved_texts[:5]

                    if memory_context:
                        context_with_memory = [f"[Conversation History]\n{memory_context}"] + retrieved_texts
                        answer = generator.answer_generation(prompt, context_with_memory)
                    else:
                        answer = generator.answer_generation(prompt, retrieved_texts)

                    metrics = evaluate_answer(
                        question=prompt,
                        answer=answer,
                        contexts=retrieved_texts,
                        sources=retrieved_texts
                    )
                    
                    st.markdown(answer)
                    
                    if used_web_search:
                        st.markdown(
                            '<div class="web-search-indicator">🌐 Used Web Search</div>',
                            unsafe_allow_html=True
                        )
                    
                    message_data = {
                        "role": "assistant",
                        "content": answer,
                        "sources": retrieved_texts,
                        "metrics": metrics,
                        "used_web_search": used_web_search
                    }
                    st.session_state.messages.append(message_data)
                    
                    if show_sources:
                        with st.expander("📚 Sources"):
                            for i, source in enumerate(retrieved_texts[:3]):
                                st.markdown(f"**Source {i+1}:**")
                                st.text(source[:500] + "..." if len(source) > 500 else source)
                    
                    if show_metrics:
                        with st.expander("📊 Evaluation Metrics"):
                            cols = st.columns(3)
                            cols[0].metric("Answer Relevance", f"{metrics.get('answer_relevance', 0):.2f}")
                            cols[1].metric("Context Relevance", f"{metrics.get('context_relevance', 0):.2f}")
                            cols[2].metric("Groundedness", f"{metrics.get('groundedness', 0):.2f}")
                    
                    if memory:
                        try:
                            memory.add_exchange(prompt, answer)
                        except:
                            pass
                
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
    
    st.divider()
    st.caption("Built with LangChain, Qdrant, Google Gemini, and SerpAPI")


if __name__ == "__main__":
    main()