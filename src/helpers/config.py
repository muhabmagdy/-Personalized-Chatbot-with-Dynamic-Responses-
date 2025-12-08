from pydantic_settings import BaseSettings
from typing import List, Optional
# Create a global instance or use a simple function without request arguments
from functools import lru_cache

class Settings(BaseSettings):

    APP_NAME: str
    APP_VERSION: str

    FILE_ALLOWED_TYPES: list
    FILE_MAX_SIZE: int
    FILE_DEFAULT_CHUNK_SIZE: int

    POSTGRES_USERNAME: str
    POSTGRES_PASSWORD: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int
    POSTGRES_MAIN_DATABASE: str

    GENERATION_BACKEND: str
    GENERATION_BACKEND_FOR_EVALUATION: str
    OLLAMA_HOST: str
    EMBEDDING_BACKEND_LITERAL: Optional[List[str]] = None
    EMBEDDING_BACKEND: str

    OPENAI_API_KEY: Optional[str] = None
    OPENAI_API_URL: Optional[str] = None
    COHERE_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None

    GENERATION_MODEL_ID_LITERAL: Optional[List[str]] = None
    GENERATION_MODEL_ID: Optional[str] = None
    GENERATION_MODEL_ID_FOR_EVALUATION: Optional[str] = None


    EMBEDDING_ID_LITERAL: Optional[List[str]] = None
    EMBEDDING_MODEL_ID: Optional[str] = None
    EMBEDDING_MODEL_SIZE: Optional[int] = None

    HUGGINGFACE_DEVICE: Optional[str] = None
    INPUT_DAFAULT_MAX_CHARACTERS: Optional[int] = None
    GENERATION_DAFAULT_MAX_TOKENS: Optional[int] = None
    GENERATION_DAFAULT_TEMPERATURE: Optional[float] = None

    TAVILY_API_KEY: Optional[str] = None

    VECTOR_DB_BACKEND_LITERAL: Optional[List[str]] = None
    VECTOR_DB_BACKEND : str
    VECTOR_DB_PATH : str
    VECTOR_DB_DISTANCE_METHOD: Optional[str] = None
    VECTOR_DB_PGVEC_INDEX_THRESHOLD: int = 100

    PRIMARY_LANG: str = "en"
    DEFAULT_LANG: str = "en"

    # ==========================================
    # DAGSHUB CONFIGURATION
    # ==========================================
    DAGSHUB_USERNAME: Optional[str] = None
    DAGSHUB_REPO_NAME: Optional[str] = None
    DAGSHUB_TOKEN: Optional[str] = None

    # ==========================================
    # MLFLOW CONFIGURATION
    # ==========================================
    MLFLOW_TRACKING_URI: Optional[str] = None  # Auto-generated from DagsHub if not provided
    MLFLOW_EXPERIMENT_NAME: str = "rag-experiments"
    MLFLOW_ENABLE_TRACKING: bool = True

    # ==========================================
    # RAGAS EVALUATION CONFIGURATION
    # ==========================================
    RAGAS_ENABLE_FAITHFULNESS: bool = True
    RAGAS_ENABLE_ANSWER_RELEVANCY: bool = True
    RAGAS_ENABLE_CONTEXT_PRECISION: bool = True
    RAGAS_BATCH_SIZE: int = 10
    RAGAS_TIMEOUT_SECONDS: int = 60

    class Config:
        env_file = ".env"

    def get_mlflow_tracking_uri(self) -> str:
        """
        Get MLflow tracking URI with fallback to DagsHub.
        
        Priority:
        1. MLFLOW_TRACKING_URI (if explicitly set)
        2. Auto-generate from DagsHub credentials
        3. Local file store as fallback
        
        Returns:
            MLflow tracking URI string
        """
        if self.MLFLOW_TRACKING_URI:
            return self.MLFLOW_TRACKING_URI
        
        if self.DAGSHUB_USERNAME and self.DAGSHUB_REPO_NAME:
            return f"https://dagshub.com/{self.DAGSHUB_USERNAME}/{self.DAGSHUB_REPO_NAME}.mlflow"
        
        return "file:./mlruns"
    
    def is_dagshub_configured(self) -> bool:
        """Check if DagsHub is properly configured."""
        return bool(
            self.DAGSHUB_USERNAME and 
            self.DAGSHUB_REPO_NAME and 
            self.DAGSHUB_TOKEN
        )
    
    def get_dagshub_credentials(self) -> dict:
        """Get DagsHub credentials for authentication."""
        if not self.is_dagshub_configured():
            return {}
        
        return {
            "username": self.DAGSHUB_USERNAME,
            "token": self.DAGSHUB_TOKEN
        }

@lru_cache() # Use lru_cache for efficient singleton retrieval
def get_settings() -> Settings:
    # Calling Settings() with no arguments automatically loads
    # the configuration from the environment and .env file.
    return Settings()