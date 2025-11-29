"""
Professional Streamlit UI for RAG System
Main Application Entry Point
"""

import streamlit as st
from ui.components.chat_interface import ChatInterface
from ui.components.sidebar import Sidebar
from ui.config.settings import UISettings
from ui.services.api_client import APIClient
from ui.utils.session_state import SessionStateManager
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="MyRAG - AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main application entry point."""
    
    # Initialize settings
    settings = UISettings()
    
    # Initialize session state manager
    session_manager = SessionStateManager()
    session_manager.initialize()
    
    # Initialize API client
    api_client = APIClient(base_url=settings.API_BASE_URL)
    
    # Inject custom CSS
    _inject_custom_css()
    
    # Render sidebar and get settings
    sidebar = Sidebar(api_client=api_client)
    project_id, rag_type, chat_settings = sidebar.render()
    
    # Render main chat interface
    chat_interface = ChatInterface(
        api_client=api_client,
        project_id=project_id,
        rag_type=rag_type,
        chat_settings=chat_settings
    )
    chat_interface.render()

def _inject_custom_css():
    """Inject custom CSS for professional styling."""
    st.markdown("""
        <style>
        /* Main container */
        .stApp {
            background-color: #ffffff;
        }
        
        /* Chat messages */
        .stChatMessage {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 0.5rem;
        }
        
        /* User message */
        [data-testid="stChatMessageContent"] {
            background-color: transparent;
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
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb {
            background: #888;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #555;
        }
        
        /* Smooth animations */
        .stChatMessage {
            animation: fadeIn 0.3s ease-in;
        }
        
        @keyframes fadeIn {
            from { 
                opacity: 0; 
                transform: translateY(10px); 
            }
            to { 
                opacity: 1; 
                transform: translateY(0); 
            }
        }
        
        /* Button styling */
        .stButton > button {
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        
        /* Input styling */
        .stTextInput > div > div > input {
            border-radius: 8px;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #f8f9fa;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            border-radius: 8px;
            font-weight: 500;
        }
        
        /* Metric styling */
        [data-testid="stMetricValue"] {
            font-size: 1.2rem;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        st.error("❌ An unexpected error occurred. Please refresh the page.")