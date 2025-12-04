"""
Sidebar Component
Handles settings, configuration, and RAG strategy selection
Now supports all 6 advanced RAG strategies with evaluation controls
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
        
        # Strategy icons and badges
        self.strategy_icons = {
            "basic": "⚡",
            "fusion": "🔍",
            "rerank": "🎯",
            "sentence_window": "📝",
            "auto_merging": "🧠",
            "web_search": "🌐"
        }
    
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
            
            # Evaluation settings
            self._render_evaluation_settings()
            
            st.divider()
            
            # Chat settings
            chat_settings = self._render_chat_settings()
            
            st.divider()
            
            # Session management
            self._render_session_management()
            
            st.divider()
            
            # System status
            self._render_system_status()
            
            st.divider()
            
            # Info section
            self._render_info()
        
        return project_id, rag_type, chat_settings
    
    def _render_header(self):
        """Render sidebar header."""
        st.markdown("""
        <div style='text-align: center; padding: 1rem 0;'>
            <h2>⚙️ Configuration</h2>
            <p style='color: #666; font-size: 0.9rem;'>Advanced RAG Settings</p>
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
        """Render RAG strategy selection UI with all 6 strategies."""
        st.markdown("#### 🧠 RAG Strategy")
        
        # Fetch available strategies
        strategies = self._get_strategies()
        
        if not strategies:
            st.error("❌ Could not load strategies")
            return "basic"
        
        # Create options with icons
        strategy_options = {}
        strategy_descriptions = {}
        strategy_requirements = {}
        
        for s in strategies:
            strategy_type = s.get('type', '')
            icon = self.strategy_icons.get(strategy_type, "📌")
            name = f"{icon} {strategy_type.replace('_', ' ').title()}"
            
            strategy_options[name] = strategy_type
            strategy_descriptions[strategy_type] = s.get('description', 'No description')
            strategy_requirements[strategy_type] = s.get('requirements', {})
        
        # Selection dropdown
        selected_name = st.selectbox(
            "Choose Strategy",
            options=list(strategy_options.keys()),
            index=0,
            help="Select the RAG strategy that best fits your query type",
            key="strategy_selector"
        )
        
        selected_type = strategy_options[selected_name]
        
        # Show description
        st.info(f"ℹ️ {strategy_descriptions.get(selected_type, 'No description')}")
        
        # Show use cases
        strategy_data = next((s for s in strategies if s.get('type') == selected_type), None)
        if strategy_data:
            use_cases = strategy_data.get('use_cases', [])
            if use_cases:
                with st.expander("💡 Best For", expanded=False):
                    for use_case in use_cases[:3]:
                        st.markdown(f"• {use_case}")
        
        # Show requirements and warnings
        requirements = strategy_requirements.get(selected_type, {})
        self._show_strategy_requirements(selected_type, requirements)
        
        # Show strategy badges
        self._render_strategy_badges(selected_type, requirements)
        
        return selected_type
    
    def _show_strategy_requirements(self, strategy_type: str, requirements: Dict[str, Any]):
        """Show requirements and warnings for selected strategy."""
        # Web search requires API key
        if strategy_type == "web_search":
            api_keys = requirements.get('api_keys', [])
            if 'tavily_api_key' in api_keys:
                st.warning(
                    "⚠️ **Web Search Strategy requires Tavily API key**\n\n"
                    "Get free key at [tavily.com](https://tavily.com) (1000 searches/month)\n\n"
                    "Configure in backend environment variables."
                )
        
        # Show complexity
        complexity = requirements.get('complexity', '').upper()
        if complexity == 'HIGH':
            st.info("🔬 **High Complexity** - Best for complex queries, may take longer")
    
    def _render_strategy_badges(self, strategy_type: str, requirements: Dict[str, Any]):
        """Render badges showing strategy characteristics."""
        # Get latency
        latency = requirements.get('avg_latency_ms', 0)
        complexity = requirements.get('complexity', 'medium')
        
        # Determine speed rating
        if latency < 300:
            speed = "⚡⚡⚡"
            speed_label = "Fast"
        elif latency < 600:
            speed = "⚡⚡"
            speed_label = "Medium"
        else:
            speed = "⚡"
            speed_label = "Slower"
        
        # Determine quality rating based on strategy
        quality_map = {
            "basic": ("⭐⭐⭐", "Good"),
            "fusion": ("⭐⭐⭐⭐", "Very Good"),
            "rerank": ("⭐⭐⭐⭐⭐", "Excellent"),
            "sentence_window": ("⭐⭐⭐⭐", "Very Good"),
            "auto_merging": ("⭐⭐⭐⭐", "Very Good"),
            "web_search": ("⭐⭐⭐⭐⭐", "Excellent")
        }
        quality, quality_label = quality_map.get(strategy_type, ("⭐⭐⭐", "Good"))
        
        # Determine cost
        complexity_cost = {
            "low": ("💰", "Low"),
            "medium": ("💰💰", "Medium"),
            "high": ("💰💰💰", "High")
        }
        cost, cost_label = complexity_cost.get(complexity, ("💰💰", "Medium"))
        
        # Display badges
        cols = st.columns(3)
        
        with cols[0]:
            st.caption("**Speed**")
            st.text(speed)
            st.caption(f"*{speed_label}*")
        
        with cols[1]:
            st.caption("**Quality**")
            st.text(quality)
            st.caption(f"*{quality_label}*")
        
        with cols[2]:
            st.caption("**Cost**")
            st.text(cost)
            st.caption(f"*{cost_label}*")
    
    def _render_evaluation_settings(self):
        """Render evaluation configuration."""
        st.markdown("#### 📊 Evaluation")
        
        enable_eval = st.checkbox(
            "Enable Quality Evaluation",
            value=st.session_state.get('enable_evaluation', False),
            help="Automatically evaluate response quality using 6 metrics",
            key="enable_evaluation"
        )
        
        if enable_eval:
            st.success("✅ Quality metrics will be shown for each response")
            
            with st.expander("ℹ️ Evaluation Metrics", expanded=False):
                st.markdown("""
                **Core Metrics:**
                - 📝 Answer Relevance
                - 🔍 Context Relevance
                - ✅ Groundedness (Hallucination Detection)
                
                **Advanced Metrics:**
                - 🎯 Context Precision
                - 📊 Context Recall
                - ✔️ Answer Correctness
                
                *Note: Evaluation may slightly increase response time*
                """)
        else:
            st.info("💡 Enable to see quality metrics for responses")
    
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
            "history_limit": history_limit,
            "enable_evaluation": st.session_state.get('enable_evaluation', False)
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
        
        # Session stats
        message_count = len(st.session_state.get('messages', []))
        st.metric("Messages in Session", message_count)
        
        # New session button
        if st.button("🔄 Start New Session", use_container_width=True):
            self._create_new_session()
    
    def _render_system_status(self):
        """Render system status indicators."""
        st.markdown("#### 🔌 System Status")
        
        # Check API health
        with st.spinner("Checking..."):
            try:
                is_healthy = asyncio.run(self.api_client.health_check())
                
                if is_healthy:
                    st.success("✅ Backend Connected")
                else:
                    st.error("❌ Backend Unreachable")
            except Exception:
                st.error("❌ Backend Unreachable")
    
    def _render_info(self):
        """Render info and help section."""
        st.markdown("#### ℹ️ About")
        
        with st.expander("📖 Help & Information", expanded=False):
            st.markdown("""
            **Python Mentor RAG System** 
            
            Advanced retrieval-augmented generation with 6 strategies and comprehensive evaluation.
            
            **🚀 Available Strategies:**
            
            1. **Basic RAG** ⚡
               - Fast vector search
               - Best for simple queries
            
            2. **Fusion RAG** 🔍
               - Multi-query expansion
               - Better for complex questions
            
            3. **ReRank RAG** 🎯
               - Two-stage retrieval
               - Highest precision
            
            4. **Sentence Window** 📝
               - Focused context retrieval
               - Great for specific facts
            
            5. **Auto-Merging** 🧠
               - Hierarchical retrieval
               - Maintains structure
            
            6. **Web Search** 🌐
               - Real-time information
               - Requires API key
            
            **📊 Evaluation Features:**
            - Real-time quality metrics
            - 6 comprehensive scores
            - Actionable recommendations
            
            **💡 Tips:**
            - Enable evaluation for quality insights
            - Try different strategies
            - Check system status regularly
            """)
        
        # Links
        st.markdown("---")
        st.markdown("**Quick Links:**")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("📖 [Docs](https://github.com)")
        with col2:
            st.markdown("💬 [Support](https://github.com)")
        
        # Footer
        st.markdown("---")
        st.caption("Python Mentor RAG v2.0 | Advanced Edition")
    
    def _get_strategies(self) -> List[Dict[str, Any]]:
        """Get available strategies from API or use defaults."""
        try:
            strategies = asyncio.run(
                self.api_client.get_available_strategies()
            )
            return strategies
        except Exception as e:
            logger.error(f"Error fetching strategies: {e}")
            # Return default strategies
            return self._get_default_strategies()
    
    def _get_default_strategies(self) -> List[Dict[str, Any]]:
        """Return default strategies if API is unavailable."""
        return [
            {
                "type": "basic",
                "description": "Simple vector similarity search - Fast and reliable",
                "use_cases": ["Simple Q&A", "Fast responses"],
                "requirements": {"complexity": "low", "avg_latency_ms": 200}
            },
            {
                "type": "fusion",
                "description": "Query expansion with result fusion - Better for complex queries",
                "use_cases": ["Complex queries", "Multi-perspective"],
                "requirements": {"complexity": "medium", "avg_latency_ms": 500}
            },
            {
                "type": "rerank",
                "description": "Two-stage retrieval with reranking - High precision",
                "use_cases": ["High precision", "Technical docs"],
                "requirements": {"complexity": "medium", "avg_latency_ms": 600}
            },
            {
                "type": "sentence_window",
                "description": "Sentence-level retrieval with context - Focused and precise",
                "use_cases": ["Precise facts", "Detailed info"],
                "requirements": {"complexity": "medium", "avg_latency_ms": 400}
            },
            {
                "type": "auto_merging",
                "description": "Hierarchical retrieval with smart merging",
                "use_cases": ["Structured content", "Technical docs"],
                "requirements": {"complexity": "high", "avg_latency_ms": 450}
            },
            {
                "type": "web_search",
                "description": "Hybrid vector + web search - Real-time information",
                "use_cases": ["Current events", "Recent info"],
                "requirements": {
                    "complexity": "medium",
                    "avg_latency_ms": 800,
                    "api_keys": ["tavily_api_key"]
                }
            }
        ]
    
    def _create_new_session(self):
        """Create a new chat session."""
        import uuid
        st.session_state.current_session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.success("✅ New session created!")
        st.rerun()