"""
Main Streamlit Application
Entry point for the Python Mentor RAG Assistant UI
Supports all 6 RAG strategies with comprehensive evaluation
"""

import streamlit as st
import logging
from ui.config.settings import get_settings
from ui.services.api_client import APIClient
from ui.components.chat_interface import ChatInterface
from ui.components.sidebar import Sidebar
from ui.utils.session_state import SessionStateManager

# ==========================================
# Logging Configuration
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==========================================
# Page Configuration
# ==========================================
st.set_page_config(
    page_title="Python Mentor RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com',
        'Report a bug': 'https://github.com',
        'About': """
        # Python Mentor RAG Assistant
        
        Advanced RAG system with 6 strategies and comprehensive evaluation.
        
        **Version:** 2.0
        **Strategies:** Basic, Fusion, ReRank, Sentence Window, Auto-Merging, Web Search
        **Evaluation:** 6 comprehensive metrics
        """
    }
)

# ==========================================
# Custom CSS
# ==========================================
def load_custom_css():
    """Load custom CSS for better UI."""
    st.markdown("""
    <style>
    /* Main container */
    .main {
        padding-top: 1rem;
    }
    
    /* Chat messages */
    .stChatMessage {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    
    /* Buttons */
    .stButton button {
        border-radius: 5px;
        transition: all 0.3s;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 600;
        border-radius: 5px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    </style>
    """, unsafe_allow_html=True)


# ==========================================
# Application State Management
# ==========================================
@st.cache_resource
def get_api_client(base_url: str, timeout: float) -> APIClient:
    """
    Get or create API client (singleton pattern).
    
    Args:
        base_url: API base URL
        timeout: Request timeout
        
    Returns:
        APIClient instance
    """
    logger.info(f"Initializing API client: {base_url}")
    return APIClient(base_url=base_url, timeout=timeout)


def initialize_app():
    """Initialize application dependencies."""
    try:
        # Load settings
        settings = get_settings()
        logger.info("Settings loaded successfully")
        
        # Initialize API client
        api_client = get_api_client(
            base_url=settings.API_BASE_URL,
            timeout=settings.API_TIMEOUT
        )
        
        # Initialize session state
        session_manager = SessionStateManager()
        session_manager.initialize()
        
        return settings, api_client, session_manager
        
    except Exception as e:
        logger.error(f"Initialization error: {e}", exc_info=True)
        st.error(f"⚠️ **Initialization Error**\n\n{str(e)}\n\nPlease check your configuration.")
        st.stop()


# ==========================================
# Main Application
# ==========================================
def main():
    """Main application entry point."""
    
    # Load custom CSS
    load_custom_css()
    
    # Initialize application
    settings, api_client, session_manager = initialize_app()
    
    # Render sidebar and get configuration
    sidebar = Sidebar(api_client=api_client)
    project_id, rag_type, chat_settings = sidebar.render()
    
    # Render chat interface
    chat_interface = ChatInterface(
        api_client=api_client,
        project_id=project_id,
        rag_type=rag_type,
        chat_settings=chat_settings
    )
    chat_interface.render()
    
    # Debug info (if enabled)
    if settings.SHOW_DEBUG_INFO:
        with st.expander("🔍 Debug Information", expanded=False):
            st.json({
                "project_id": project_id,
                "rag_type": rag_type,
                "session_id": st.session_state.get('current_session_id', 'N/A')[:16],
                "message_count": len(st.session_state.get('messages', [])),
                "settings": {
                    "api_url": settings.API_BASE_URL,
                    "doc_limit": chat_settings.get('doc_limit'),
                    "history_limit": chat_settings.get('history_limit'),
                    "evaluation_enabled": chat_settings.get('enable_evaluation')
                }
            })


# ==========================================
# Error Boundary
# ==========================================
def run_with_error_handling():
    """Run application with global error handling."""
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        st.error(f"""
        ## 🚨 Application Error
        
        An unexpected error occurred:
        
        ```
        {str(e)}
        ```
        
        **Troubleshooting:**
        1. Check if backend server is running
        2. Verify `.env` configuration
        3. Check logs for detailed error information
        4. Restart the application
        
        **Still having issues?** Check the documentation or contact support.
        """)


# ==========================================
# Entry Point
# ==========================================
if __name__ == "__main__":
    run_with_error_handling()