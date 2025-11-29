"""
UI Configuration Settings
Manages all configuration for the Streamlit UI
"""
from pydantic_settings import BaseSettings
# Create a global instance or use a simple function without request arguments
from functools import lru_cache

class UISettings(BaseSettings):
    """
    Configuration settings for the Streamlit UI.
    """
    # ==========================================
    # API Configuration
    # ==========================================
    API_BASE_URL: str
    API_TIMEOUT: float

    # ==========================================
    # Project Configuration
    # ==========================================
    DEFAULT_PROJECT_ID: int 
    
    # ==========================================
    # RAG Configuration
    # ==========================================
    DEFAULT_RAG_TYPE: str 
    DEFAULT_DOC_LIMIT: int 
    DEFAULT_HISTORY_LIMIT: int 

    # ==========================================
    # Display Configuration
    # ==========================================
    MAX_MESSAGE_LENGTH: int 
    SHOW_DEBUG_INFO: bool 
    
    # ==========================================
    # UI Theme Configuration
    # ==========================================
    PRIMARY_COLOR: str 
    BACKGROUND_COLOR: str
    SECONDARY_BG_COLOR: str 
    TEXT_COLOR: str 
    
    # ==========================================
    # Feature Flags
    # ==========================================
    ENABLE_SESSION_MANAGEMENT: bool 
    ENABLE_ADVANCED_SETTINGS: bool
    ENABLE_RESPONSE_METADATA: bool 
    
    def __post_init__(self):
        """Validate settings after initialization."""
        # Validate API URL
        if not self.API_BASE_URL:
            raise ValueError("API_BASE_URL must be set")
        
        # Validate numeric ranges
        if self.DEFAULT_DOC_LIMIT < 1 or self.DEFAULT_DOC_LIMIT > 100:
            raise ValueError("DEFAULT_DOC_LIMIT must be between 1 and 100")
        
        if self.DEFAULT_HISTORY_LIMIT < 0 or self.DEFAULT_HISTORY_LIMIT > 100:
            raise ValueError("DEFAULT_HISTORY_LIMIT must be between 0 and 100")
        
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"UISettings(API={self.API_BASE_URL}, Project={self.DEFAULT_PROJECT_ID})"
    
    class Config:
        env_file = "../../.env"

@lru_cache() # Use lru_cache for efficient singleton retrieval
def get_settings() -> UISettings:
    # Calling Settings() with no arguments automatically loads
    # the configuration from the environment and .env file.
    return UISettings()
    