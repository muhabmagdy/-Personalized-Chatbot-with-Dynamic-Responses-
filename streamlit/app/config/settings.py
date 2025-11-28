# app/config/settings.py
from pydantic_settings import BaseSettings
# Create a global instance or use a simple function without request arguments
from functools import lru_cache

class Settings(BaseSettings):
    API_BASE_URL: str
    DEFAULT_PROJECT_ID: int = 1
    DEFAULT_LIMIT: int = 5

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache() # Use lru_cache for efficient singleton retrieval
def get_settings() -> Settings:
    # Calling Settings() with no arguments automatically loads
    # the configuration from the environment and .env file.
    return Settings()
