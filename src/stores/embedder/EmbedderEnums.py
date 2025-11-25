from enum import Enum

class DocumentTypeEnum(Enum):
    DOCUMENT = "document"
    QUERY = "query"

class EmbedderEnums(Enum):
    """Enumeration of supported embedding providers."""
    HUGGINGFACE = "HUGGINGFACE"
    OPENAI = "OPENAI"
    COHERE = "COHERE"
    GEMINI = "GEMINI"
    # Future providers would be added here,
