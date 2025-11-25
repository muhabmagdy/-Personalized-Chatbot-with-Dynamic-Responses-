APP_NAME="Python-Mentor-RAG"
APP_VERSION="0.1"
OPENAI_API_KEY="sk-"

FILE_ALLOWED_TYPES=["text/plain", "application/pdf"]
FILE_MAX_SIZE=10
FILE_DEFAULT_CHUNK_SIZE=512000 # 512KB

POSTGRES_USERNAME=""
POSTGRES_PASSWORD=""
POSTGRES_HOST="" # Use the service name defined in docker-compose.yml - all in the same network ==> backend
POSTGRES_PORT=5432
POSTGRES_MAIN_DATABASE="mrag"

# ========================= LLM Config =========================
GENERATION_BACKEND = "GEMINI"
EMBEDDING_BACKEND_LITERAL = ["HUGGINGFACE"]
EMBEDDING_BACKEND = "HUGGINGFACE"

OPENAI_API_KEY="sk-"
OPENAI_API_URL= ""
COHERE_API_KEY="m8-"
GEMINI_API_KEY=""

GENERATION_MODEL_ID_LITERAL = ["gpt-4o-mini", "gpt-4o","gemini-2.5-flash"]
GENERATION_MODEL_ID="gemini-2.5-flash"
EMBEDDING_MODEL_ID="sentence-transformers/all-MiniLM-L6-v2" #gemini-embedding-001, sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_MODEL_SIZE=384
HUGGINGFACE_DEVICE ="cpu"

INPUT_DAFAULT_MAX_CHARACTERS=2000
GENERATION_DAFAULT_MAX_TOKENS=4000
GENERATION_DAFAULT_TEMPERATURE=0.1

# ========================= Vector DB Config =========================
VECTOR_DB_BACKEND_LITERAL = ["QDRANT", "PGVECTOR"]
VECTOR_DB_BACKEND = "PGVECTOR"
VECTOR_DB_PATH = "qdrant_db"
VECTOR_DB_DISTANCE_METHOD = "cosine"
VECTOR_DB_PGVEC_INDEX_THRESHOLD = 1000

# ========================= Template Configs =========================
PRIMARY_LANG = "en"
DEFAULT_LANG = "en"
