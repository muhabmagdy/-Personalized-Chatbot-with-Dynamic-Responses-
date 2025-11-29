"""
Chat Interface Component
Handles the main chat display and user interactions
"""

import streamlit as st
import asyncio
from typing import Dict, Any
from ui.services.api_client import APIClient, APIError
import logging
import time

logger = logging.getLogger(__name__)

class ChatInterface:
    """Professional chat interface similar to ChatGPT/Claude."""
    
    def __init__(
        self,
        api_client: APIClient,
        project_id: int,
        rag_type: str,
        chat_settings: Dict[str, Any]
    ):
        """Initialize chat interface with dependencies."""
        self.api_client = api_client
        self.project_id = project_id
        self.rag_type = rag_type
        self.chat_settings = chat_settings
    
    def render(self):
        """Render the complete chat interface."""
        # Header
        self._render_header()
        
        # Chat messages container
        self._render_messages()
        
        # Input area
        self._render_input()
    
    def _render_header(self):
        """Render chat header with title and controls."""
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.title("🤖 Python Mentor RAG Assistant")
            st.caption("Ask me anything about your documents")
        
        with col2:
            if st.session_state.get('current_session_id'):
                session_short = st.session_state.current_session_id[:8]
                st.caption(f"**Session:** `{session_short}...`")
        
        with col3:
            if st.button("🗑️ Clear Chat", use_container_width=True, type="secondary"):
                self._clear_chat()
    
    def _render_messages(self):
        """Render all messages in chat history."""
        if not st.session_state.get('messages'):
            # Welcome message
            self._render_welcome_message()
        else:
            # Render message history
            for message in st.session_state.messages:
                with st.chat_message(message["role"], avatar=self._get_avatar(message["role"])):
                    st.markdown(message["content"])
                    
                    # Show metadata for assistant messages
                    if message["role"] == "assistant" and message.get("metadata"):
                        self._render_message_metadata(message["metadata"])
    
    def _render_welcome_message(self):
        """Render initial welcome message."""
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown("""
            ### 👋 Welcome to Python Mentor RAG Assistant!
            
            I'm here to help you find information from your documents using advanced RAG techniques.
            
            **How to use:**
            - 📝 Ask me any question about your documents
            - 🔍 I'll search through your knowledge base
            - 💡 Get accurate, context-aware answers
            
            **Tips:**
            - Use the sidebar to select your preferred RAG strategy
            - Adjust retrieval settings for better results
            - Your conversation history is saved automatically
            
            **Ready to start? Ask me anything!** 🚀
            """)
    
    def _render_input(self):
        """Render chat input area."""
        if prompt := st.chat_input(
            "💭 Ask me anything about your documents...",
            key="chat_input"
        ):
            self._handle_user_input(prompt)
    
    def _handle_user_input(self, prompt: str):
        """
        Handle user message submission.
        
        Args:
            prompt: User's input text
        """
        # Validate input
        if not prompt.strip():
            st.warning("⚠️ Please enter a message")
            return
        
        # Add user message to history
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Display user message
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        # Get assistant response
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            metadata_container = st.container()
            
            with st.spinner("🤔 Thinking..."):
                try:
                    start_time = time.time()
                    
                    # Call API
                    response = asyncio.run(
                        self.api_client.send_message(
                            project_id=self.project_id,
                            text=prompt,
                            session_id=st.session_state.current_session_id,
                            rag_type=self.rag_type,
                            limit=self.chat_settings.get('doc_limit', 10),
                            chat_history_limit=self.chat_settings.get('history_limit', 10)
                        )
                    )
                    
                    elapsed_time = time.time() - start_time
                    
                    # Extract response
                    answer = response.get('answer', 'Sorry, I could not generate a response.')
                    
                    # Display answer
                    message_placeholder.markdown(answer)
                    
                    # Prepare metadata
                    metadata = {
                        "strategy": response.get('rag_strategy', 'Unknown'),
                        "rag_type": response.get('rag_type', 'Unknown'),
                        "session_id": response.get('session_id'),
                        "response_time": f"{elapsed_time:.2f}s"
                    }
                    
                    # Add to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "metadata": metadata
                    })
                    
                    # Show metadata
                    with metadata_container:
                        self._render_message_metadata(metadata)
                    
                except APIError as e:
                    error_msg = f"❌ **Error:** {str(e)}\n\nPlease check:\n- Backend server is running\n- Project ID is correct\n- Network connection is stable"
                    message_placeholder.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
                except Exception as e:
                    logger.error(f"Unexpected error: {e}", exc_info=True)
                    error_msg = f"❌ **Unexpected error occurred**\n\n`{str(e)}`\n\nPlease try again or refresh the page."
                    message_placeholder.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
    
    def _render_message_metadata(self, metadata: Dict[str, Any]):
        """Render response metadata in an expander."""
        with st.expander("📊 Response Details", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="RAG Strategy",
                    value=metadata.get('strategy', 'N/A'),
                    help="The retrieval strategy used"
                )
            
            with col2:
                st.metric(
                    label="Strategy Type",
                    value=metadata.get('rag_type', 'N/A'),
                    help="The type of RAG approach"
                )
            
            with col3:
                st.metric(
                    label="Response Time",
                    value=metadata.get('response_time', 'N/A'),
                    help="Time taken to generate response"
                )
            
            # Additional info
            if metadata.get('session_id'):
                st.caption(f"**Session ID:** `{metadata['session_id'][:16]}...`")
    
    def _clear_chat(self):
        """Clear current chat session."""
        if st.session_state.get('messages'):
            # Show confirmation
            if len(st.session_state.messages) > 0:
                with st.spinner("🗑️ Clearing chat..."):
                    # Clear in backend
                    try:
                        asyncio.run(
                            self.api_client.clear_chat_session(
                                project_id=self.project_id,
                                session_id=st.session_state.current_session_id
                            )
                        )
                    except Exception as e:
                        logger.error(f"Error clearing session: {e}")
                    
                    # Clear locally
                    st.session_state.messages = []
                    st.success("✅ Chat cleared!")
                    time.sleep(0.5)
                    st.rerun()
        else:
            st.info("ℹ️ Chat is already empty")
    
    def _get_avatar(self, role: str) -> str:
        """Get avatar emoji for role."""
        return "👤" if role == "user" else "🤖"