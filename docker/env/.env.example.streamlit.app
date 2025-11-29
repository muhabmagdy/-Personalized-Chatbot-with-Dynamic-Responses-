# ==========================================
# STREAMLIT UI - ENVIRONMENT CONFIGURATION
# ==========================================

# ==========================================
# API Configuration
# ==========================================
# Backend API base URL (where your FastAPI server is running)
API_BASE_URL=""
# API request timeout in seconds
API_TIMEOUT=30.0

# ==========================================
# Project Configuration
# ==========================================
# Default project ID to use
DEFAULT_PROJECT_ID=1

# ==========================================
# RAG Configuration
# ==========================================
# Default RAG strategy: basic, fusion, rerank
DEFAULT_RAG_TYPE=basic

# Default number of documents to retrieve
DEFAULT_DOC_LIMIT=10

# Default chat history limit
DEFAULT_HISTORY_LIMIT=10

# ==========================================
# Display Configuration
# ==========================================
# Maximum message length to display
MAX_MESSAGE_LENGTH=2000

# Show debug information (true/false)
SHOW_DEBUG_INFO=false

# ==========================================
# UI Theme Configuration (Optional)
# ==========================================
# Primary accent color (hex code)
PRIMARY_COLOR=#FF4B4B

# Main background color
BACKGROUND_COLOR=#FFFFFF

# Secondary background color (sidebar)
SECONDARY_BG_COLOR=#F0F2F6

# Text color
TEXT_COLOR=#262730

# ==========================================
# Feature Flags (Optional)
# ==========================================
# Enable session management UI
ENABLE_SESSION_MANAGEMENT=true

# Enable advanced settings expander
ENABLE_ADVANCED_SETTINGS=true

# Show response metadata
ENABLE_RESPONSE_METADATA=true

# ==========================================
# NOTES
# ==========================================
# 
# Update API_BASE_URL to match your backend server
# Adjust other settings as needed
#
# For production:
# - Use environment-specific .env files
# - Set API_BASE_URL to your production API
# - Disable debug mode
# ==========================================