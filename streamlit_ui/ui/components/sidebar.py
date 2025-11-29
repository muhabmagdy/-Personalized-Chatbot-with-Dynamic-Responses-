"""
Sidebar Component
Handles settings, configuration, and RAG strategy selection
"""

import streamlit as st
import asyncio
from typing import Tuple, Dict, Any, List
from ui.services.api_client import APIClient
import logging

logger = logging.getLogger(__name__)

class Sidebar:
    """Sidebar component for RAG settings and configuration."""
    
    def __init__(self, api_client: APIClient):
        """Initialize sidebar with API client."""
        self.api_client = api_client
    
    def render(self) -> Tuple[int, str, Dict[str, Any]]:
        """
        Render sidebar and return selected settings.
        
        Returns:
            Tuple of (project_id, rag_type, chat_settings)
        """
        with st.sidebar:
            # Logo/Branding
            self._render_header()
            
            st.divider()
            
            # Project selection
            project_id = self._render_project_selector()
            
            st.divider()
            
            # RAG strategy selection
            rag_type = self._render_strategy_selector()
            
            st.divider()
            
            # Chat settings
            chat_settings = self._render_chat_settings()
            
            st.divider()
            
            # Session management
            self._render_session_management()
            
            st.divider()
            
            # Info section
            self._render_info()
        
        return project_id, rag_type, chat_settings
    
    def _render_header(self):
        """Render sidebar header."""
        st.markdown("""
        <div style='text-align: center; padding: 1rem 0;'>
            <h2>⚙️ Settings</h2>
            <p style='color: #666; font-size: 0.9rem;'>Configure your RAG experience</p>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_project_selector(self) -> int:
        """Render project selection UI."""
        st.markdown("#### 📁 Project")
        
        project_id = st.number_input(
            "Project ID",
            min_value=1,
            max_value=9999,
            value=st.session_state.get('project_id', 1),
            help="Select the project to query from your knowledge base",
            key="project_id_input"
        )
        
        # Store in session state
        st.session_state.project_id = project_id
        
        # Show project info
        st.caption(f"Currently using **Project {project_id}**")
        
        return project_id
    
    def _render_strategy_selector(self) -> str:
        """Render RAG strategy selection UI."""
        st.markdown("#### 🧠 RAG Strategy")
        
        # Fetch available strategies
        strategies = self._get_strategies()
        
        if not strategies:
            st.error("❌ Could not load strategies")
            return "basic"
        
        # Create options mapping
        strategy_options = {
            f"{s['name']}": s['type'] 
            for s in strategies
        }
        
        # Description mapping
        strategy_descriptions = {
            s['type']: s.get('description', 'No description available')
            for s in strategies
        }
        
        # Selection dropdown
        selected_name = st.selectbox(
            "Choose Strategy",
            options=list(strategy_options.keys()),
            index=0,
            help="Different strategies offer different trade-offs between speed and accuracy",
            key="strategy_selector"
        )
        
        selected_type = strategy_options[selected_name]
        
        # Show description
        st.info(f"ℹ️ {strategy_descriptions.get(selected_type, 'No description')}")
        
        # Show strategy badges
        self._render_strategy_badges(selected_type)
        
        return selected_type
    
    def _render_strategy_badges(self, strategy_type: str):
        """Render badges showing strategy characteristics."""
        badges = {
            "basic": {
                "Speed": "⚡⚡⚡",
                "Quality": "⭐⭐⭐",
                "Cost": "💰"
            },
            "fusion": {
                "Speed": "⚡⚡",
                "Quality": "⭐⭐⭐⭐",
                "Cost": "💰💰"
            },
            "rerank": {
                "Speed": "⚡⚡",
                "Quality": "⭐⭐⭐⭐⭐",
                "Cost": "💰"
            }
        }
        
        if strategy_type in badges:
            cols = st.columns(3)
            for idx, (label, value) in enumerate(badges[strategy_type].items()):
                with cols[idx]:
                    st.caption(f"**{label}**")
                    st.text(value)
    
    def _render_chat_settings(self) -> Dict[str, Any]:
        """Render chat configuration settings."""
        st.markdown("#### 💬 Chat Settings")
        
        with st.expander("⚙️ Advanced Settings", expanded=False):
            st.markdown("##### Retrieval Settings")
            
            doc_limit = st.slider(
                "Documents to Retrieve",
                min_value=5,
                max_value=50,
                value=10,
                step=5,
                help="Number of documents to retrieve from vector database",
                key="doc_limit"
            )
            
            st.caption(f"Will retrieve **{doc_limit}** most relevant documents")
            
            st.markdown("##### History Settings")
            
            history_limit = st.slider(
                "Chat History Limit",
                min_value=0,
                max_value=50,
                value=10,
                step=5,
                help="Number of previous messages to include in context",
                key="history_limit"
            )
            
            st.caption(f"Using last **{history_limit}** messages for context")
        
        return {
            "doc_limit": doc_limit,
            "history_limit": history_limit
        }
    
    def _render_session_management(self):
        """Render session management UI."""
        st.markdown("#### 💾 Session")
        
        # Show current session
        if st.session_state.get('current_session_id'):
            session_id = st.session_state.current_session_id
            st.text_input(
                "Current Session ID",
                value=session_id[:16] + "...",
                disabled=True,
                help="Your unique conversation session"
            )
        
        # New session button
        if st.button("🔄 Start New Session", use_container_width=True):
            self._create_new_session()
    
    def _render_info(self):
        """Render info and help section."""
        st.markdown("#### ℹ️ About")
        
        with st.expander("📖 Help & Information", expanded=False):
            st.markdown("""
            **MyRAG System** uses advanced retrieval techniques to answer your questions.
            
            **Available Strategies:**
            
            🚀 **Basic RAG**
            - Fast and simple
            - Best for straightforward questions
            - Low latency
            
            🔍 **Fusion RAG**
            - Multi-query approach
            - Better for complex questions
            - Higher accuracy
            
            🎯 **Gemini ReRank**
            - Two-stage retrieval
            - Highest precision
            - FREE with Gemini API
            
            **Tips for Better Results:**
            - Be specific in your questions
            - Use natural language
            - Check response details for insights
            - Adjust retrieval settings if needed
            
            **Need Help?**
            - Check the documentation
            - Review response details
            - Try different strategies
            """)
        
        # Links
        st.markdown("---")
        st.markdown("**Quick Links:**")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("📖 [Documentation](#)")
        with col2:
            st.markdown("💬 [Support](#)")
        
        # Footer
        st.markdown("---")
        st.caption("Python Mentor RAG v1.0 | Made with ❤️")
    
    def _get_strategies(self) -> List[Dict[str, str]]:
        """Get available strategies from API or use defaults."""
        try:
            strategies = asyncio.run(
                self.api_client.get_available_strategies()
            )
            return strategies
        except Exception as e:
            logger.error(f"Error fetching strategies: {e}")
            # Return default strategies
            return [
                {
                    "type": "basic",
                    "name": "Basic RAG",
                    "description": "Fast, simple retrieval"
                },
                {
                    "type": "fusion",
                    "name": "Fusion RAG",
                    "description": "Multi-query expansion"
                },
                {
                    "type": "rerank",
                    "name": "Gemini ReRank",
                    "description": "Two-stage with reranking (FREE)"
                }
            ]
    
    def _create_new_session(self):
        """Create a new chat session."""
        import uuid
        st.session_state.current_session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.success("✅ New session created!")
        st.rerun()