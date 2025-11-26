import os
import streamlit as st
from dotenv import load_dotenv
from rag.data_loader import DataLoader

# Load environment variables from .env file
load_dotenv()
from rag.chunker import TextChunker
from rag.embedder import Embedder
from rag.vectorstore_manager import VectorStoreManager
from rag.query_engine import QueryEngine
from rag.auto_merging_retriever import AutoMergingRetriever
from rag.rerank import Reranker
from rag.FusionRetriever import FusionRetriever
from llm.llm_generator import LLMGenerator
from rag.intelligent_memory import IntelligentMemory
from evaluation.evaluator import evaluate_answer


# Page config
st.set_page_config(
    page_title="Python RAG Chatbot",
    page_icon="🐍",
    layout="wide"
)

# Custom CSS
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
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_rag_system():
    """Initialize RAG components (cached to avoid reloading)."""
    
    import os
    
    # Try multiple possible locations for data files
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    possible_data_dirs = [
        os.path.join(parent_dir, "data", "cleaned_texts"),  # ../data/cleaned_texts/
        os.path.join(parent_dir, "data"),                    # ../data/
        os.path.join(current_dir, "data", "cleaned_texts"),  # ./data/cleaned_texts/
        os.path.join(current_dir, "data"),                   # ./data/
    ]
    
    file_names = [
        "automate-the-boring-stuff-with-python-3rd-edition-early-access-3nbsped-9781718503403-9781718503410_com.txt",
        "fluent-python-2nbsped-9781492056348-9781492056287.txt",
        "learning-python-powerful-object-oriented-programming-6nbsped-1098171306-9781098171308.txt",
        "leetcode-python.txt",
        "python-crash-course-3nbsped-1718502702-9781718502703_compress.txt"
    ]
    
    # Find the correct data directory
    data_dir = None
    for possible_dir in possible_data_dirs:
        if os.path.exists(possible_dir):
            # Check if at least one file exists
            test_file = os.path.join(possible_dir, file_names[0])
            if os.path.exists(test_file):
                data_dir = possible_dir
                break
    
    if not data_dir:
        raise FileNotFoundError(
            f"Could not find data files. Searched in: {possible_data_dirs}"
        )
    
    text_files = [os.path.join(data_dir, fname) for fname in file_names]
    
    # Verify all files exist
    missing_files = [f for f in text_files if not os.path.exists(f)]
    if missing_files:
        raise FileNotFoundError(f"Missing files: {missing_files}")
    
    # Load and process documents
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
    
    # Initialize reranker (optional - may fail if model can't be downloaded)
    try:
        # Use a smaller, faster model (90MB instead of 1.1GB)
        reranker = Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    except Exception as e:
        print(f"Warning: Reranker failed to load: {e}")
        reranker = None
    
    # Initialize fusion retriever
    fusion_retriever = FusionRetriever(k=60)
    
    return query_engine, len(chunks), reranker, fusion_retriever


@st.cache_resource
def initialize_llm():
    """Initialize LLM generator."""
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        return None
    return LLMGenerator(api_key=gemini_key)


@st.cache_resource
def initialize_memory():
    """Initialize intelligent memory system."""
    return IntelligentMemory(
        collection_name="user_memory",
        path="./user_memory_qdrant_db"
    )


def main():
    # Header
    st.title("🐍 Python RAG Chatbot")
    st.markdown("*Your AI-powered Python learning assistant*")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Check API Key status
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            st.success("✅ API Key Loaded")
        else:
            st.error("❌ API Key Missing")
            st.info("Add GEMINI_API_KEY to your .env file")
        
        st.divider()
        
        # Display settings
        show_sources = st.checkbox("Show retrieved sources", value=False)
        show_metrics = st.checkbox("Show evaluation metrics", value=False)
        
        # Only show reranking option if reranker loaded
        if 'reranker' in st.session_state and st.session_state.get('reranker') is not None:
            use_reranking = st.checkbox("Enable Reranking", value=True, help="Use cross-encoder reranking for better results")
        else:
            use_reranking = False
            st.info("ℹ️ Reranker unavailable (model not loaded)")
        
        use_fusion = st.checkbox("Enable Fusion", value=True, help="Combine multiple retrieval strategies")
        
        st.divider()
        
        # Memory controls
        st.header("🧠 Memory")
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        
        # System status
        st.header("📊 System Status")
    
    # Initialize systems
    with st.spinner("Loading RAG system... This may take a moment."):
        try:
            query_engine, num_chunks, reranker, fusion_retriever = initialize_rag_system()
            st.sidebar.success(f"✅ RAG System Ready")
            st.sidebar.info(f"📚 {num_chunks} chunks indexed")
            
            # Store reranker status in session state
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
    
    # Initialize LLM
    generator = initialize_llm()
    if generator:
        st.sidebar.success("✅ LLM Connected")
    else:
        st.sidebar.warning("⚠️ No API Key - Enter in sidebar")
    
    # Initialize memory
    try:
        memory_system = initialize_memory()
        memory = memory_system.get_retriever()
        st.sidebar.success("✅ Memory System Ready")
    except Exception as e:
        st.sidebar.error(f"❌ Memory Error: {str(e)[:50]}")
        memory = None
        memory_system = None
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if enabled
            if show_sources and "sources" in message:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(message["sources"][:3]):
                        st.markdown(f"**Source {i+1}:**")
                        st.text(source[:500] + "..." if len(source) > 500 else source)
            
            # Show metrics if enabled
            if show_metrics and "metrics" in message:
                with st.expander("📊 Evaluation Metrics"):
                    cols = st.columns(3)
                    metrics = message["metrics"]
                    cols[0].metric("Answer Relevance", f"{metrics.get('answer_relevance', 0):.2f}")
                    cols[1].metric("Context Relevance", f"{metrics.get('context_relevance', 0):.2f}")
                    cols[2].metric("Groundedness", f"{metrics.get('groundedness', 0):.2f}")
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about Python..."):
        # Check if systems are ready
        if not generator:
            st.error("Please enter your Gemini API key in the sidebar.")
            return
        
        if not query_engine:
            st.error("RAG system failed to initialize. Check your data files.")
            return
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Retrieve from memory
                    memory_texts = []
                    memory_docs_list = []
                    if memory:
                        try:
                            memory_docs = memory.invoke(prompt)
                            memory_texts = [doc.page_content for doc in memory_docs]
                            memory_docs_list = memory_texts
                        except:
                            pass
                    
                    # Retrieve from RAG
                    merged_docs = query_engine.query(prompt)
                    rag_texts = [d.page_content for d in merged_docs[:10]]  # Get more for fusion/reranking
                    
                    # Apply Fusion if enabled
                    if use_fusion and fusion_retriever:
                        # Combine multiple retrieval strategies
                        retrieval_results = [
                            memory_docs_list,
                            rag_texts
                        ]
                        fused_texts = fusion_retriever.fuse(retrieval_results, top_k=10)
                        retrieved_texts = fused_texts
                    else:
                        retrieved_texts = memory_texts + rag_texts
                    
                    # Apply Reranking if enabled
                    if use_reranking and reranker and len(retrieved_texts) > 0:
                        reranked = reranker.rerank(prompt, retrieved_texts, top_k=5)
                        retrieved_texts = [doc for doc, score in reranked]
                    else:
                        retrieved_texts = retrieved_texts[:5]  # Top 5 without reranking
                    
                    # Generate answer
                    answer = generator.answer_generation(prompt, retrieved_texts)
                    
                    # Evaluate answer
                    metrics = evaluate_answer(
                        question=prompt,
                        answer=answer,
                        contexts=retrieved_texts,
                        sources=retrieved_texts
                    )

                    st.markdown(answer)

                    message_data = {
                        "role": "assistant",
                        "content": answer,
                        "sources": retrieved_texts,
                        "metrics": metrics
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

                    if memory_system:
                        try:
                            memory_system.add_memory(f"Q: {prompt}\nA: {answer[:500]}")
                        except:
                            pass
                            
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")

    st.divider()
    st.caption("Built with LangChain, Qdrant, and Google Gemini")


if __name__ == "__main__":
    main()