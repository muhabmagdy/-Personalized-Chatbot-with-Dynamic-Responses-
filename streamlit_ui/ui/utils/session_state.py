"""
Session State Manager
Handles Streamlit session state initialization and management
"""

import streamlit as st
import uuid
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)

class SessionStateManager:
    """
    Manager for Streamlit session state.
    
    Centralizes session state initialization and management
    following the Singleton pattern.
    """
    
    # Session state keys
    KEY_MESSAGES = 'messages'
    KEY_SESSION_ID = 'current_session_id'
    KEY_PROJECT_ID = 'project_id'
    KEY_RAG_TYPE = 'rag_type'
    KEY_DOC_LIMIT = 'doc_limit'
    KEY_HISTORY_LIMIT = 'history_limit'
    KEY_STRATEGIES = 'strategies'
    
    def initialize(self):
        """
        Initialize all required session state variables.
        
        Called once at app startup to set up initial state.
        """
        logger.info("Initializing session state")
        
        # Chat state
        self._init_if_not_exists(self.KEY_MESSAGES, [])
        self._init_if_not_exists(self.KEY_SESSION_ID, str(uuid.uuid4()))
        
        # Project state
        self._init_if_not_exists(self.KEY_PROJECT_ID, 1)
        
        # RAG configuration
        self._init_if_not_exists(self.KEY_RAG_TYPE, 'basic')
        self._init_if_not_exists(self.KEY_DOC_LIMIT, 10)
        self._init_if_not_exists(self.KEY_HISTORY_LIMIT, 10)
        
        # Cache
        self._init_if_not_exists(self.KEY_STRATEGIES, None)
        
        logger.info(f"Session initialized with ID: {st.session_state[self.KEY_SESSION_ID][:8]}...")
    
    def reset_chat(self):
        """
        Reset chat state for a new conversation.
        
        Keeps project and settings, only clears messages and creates new session.
        """
        logger.info("Resetting chat session")
        st.session_state[self.KEY_MESSAGES] = []
        st.session_state[self.KEY_SESSION_ID] = str(uuid.uuid4())
        logger.info(f"New session created: {st.session_state[self.KEY_SESSION_ID][:8]}...")
    
    def reset_all(self):
        """
        Reset all session state.
        
        Complete reset to initial state.
        """
        logger.info("Resetting all session state")
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        self.initialize()
    
    def _init_if_not_exists(self, key: str, default_value: Any):
        """
        Initialize a session state key if it doesn't exist.
        
        Args:
            key: Session state key
            default_value: Default value if key doesn't exist
        """
        if key not in st.session_state:
            st.session_state[key] = default_value
            logger.debug(f"Initialized session state key: {key}")
    
    # ==========================================
    # Getters and Setters
    # ==========================================
    
    @staticmethod
    def get(key: str, default: Any = None) -> Any:
        """
        Get a session state value.
        
        Args:
            key: Session state key
            default: Default value if key doesn't exist
            
        Returns:
            Value from session state or default
        """
        return st.session_state.get(key, default)
    
    @staticmethod
    def set(key: str, value: Any):
        """
        Set a session state value.
        
        Args:
            key: Session state key
            value: Value to set
        """
        st.session_state[key] = value
        logger.debug(f"Set session state: {key}")
    
    @staticmethod
    def delete(key: str):
        """
        Delete a session state key.
        
        Args:
            key: Session state key to delete
        """
        if key in st.session_state:
            del st.session_state[key]
            logger.debug(f"Deleted session state key: {key}")
    
    # ==========================================
    # Convenience Methods
    # ==========================================
    
    @staticmethod
    def get_messages() -> list:
        """Get chat messages."""
        return st.session_state.get('messages', [])
    
    @staticmethod
    def add_message(role: str, content: str, metadata: Optional[dict] = None):
        """
        Add a message to chat history.
        
        Args:
            role: Message role ('user' or 'assistant')
            content: Message content
            metadata: Optional metadata dictionary
        """
        message = {
            "role": role,
            "content": content
        }
        if metadata:
            message["metadata"] = metadata
        
        st.session_state.messages.append(message)
        logger.debug(f"Added {role} message to chat history")
    
    @staticmethod
    def get_session_id() -> str:
        """Get current session ID."""
        return st.session_state.get('current_session_id', '')
    
    @staticmethod
    def get_project_id() -> int:
        """Get current project ID."""
        return st.session_state.get('project_id', 1)
    
    @staticmethod
    def get_stats() -> dict:
        """
        Get session statistics.
        
        Returns:
            Dictionary with session stats
        """
        return {
            'message_count': len(st.session_state.get('messages', [])),
            'session_id': st.session_state.get('current_session_id', 'N/A')[:8],
            'project_id': st.session_state.get('project_id', 'N/A'),
            'rag_type': st.session_state.get('rag_type', 'N/A'),
        }
    
    @staticmethod
    def has_messages() -> bool:
        """Check if there are any messages in chat history."""
        return len(st.session_state.get('messages', [])) > 0
    
    @staticmethod
    def get_last_message() -> Optional[dict]:
        """Get the last message from chat history."""
        messages = st.session_state.get('messages', [])
        return messages[-1] if messages else None
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        stats = self.get_stats()
        return f"SessionStateManager(messages={stats['message_count']}, session={stats['session_id']}...)"