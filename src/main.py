from fastapi import FastAPI
from routes import base, data, nlp, evaluation, tracking, answer_generation
from helpers.config import get_settings
from stores.llm.LLMProviderFactory import LLMProviderFactory
from stores.embedder.EmbedderProviderFactory import EmbedderProviderFactory
from stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory
from stores.llm.templates.template_parser import TemplateParser
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import logging

# Import metrics setup
from utils.metrics import setup_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("uvicorn")

app = FastAPI(
    title="RAG Application with Evaluation & Tracking",
    description="Advanced RAG system with RAGAs evaluation and MLflow tracking",
    version="2.0.0"
)

# Setup Prometheus metrics
setup_metrics(app)

@app.get("/welcome")
def welcome():
    return {
        "message": "Welcome to RAG API with Evaluation & Tracking",
        "version": "2.0.0",
        "features": [
            "6 RAG strategies",
            "RAGAs evaluation (faithfulness, answer_relevancy, context_precision)",
            "MLflow tracking with DagsHub",
            "Separated answer generation and evaluation",
            "Batch evaluation support"
        ]
    }


async def startup_span():
    """Initialize all services on startup."""
    settings = get_settings()
    
    # Store settings in app state for access in routes
    app.settings = settings
    
    logger.info("Starting RAG Application initialization...")
    
    # ==========================================
    # DATABASE SETUP
    # ==========================================
    postgres_conn = (
        f"postgresql+asyncpg://{settings.POSTGRES_USERNAME}:"
        f"{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:"
        f"{settings.POSTGRES_PORT}/{settings.POSTGRES_MAIN_DATABASE}"
    )
    
    app.db_engine = create_async_engine(postgres_conn)
    app.db_client = sessionmaker(
        app.db_engine, class_=AsyncSession, expire_on_commit=False
    )
    logger.info("✓ Database connection established")
    
    # ==========================================
    # LLM CLIENTS SETUP
    # ==========================================
    llm_provider_factory = LLMProviderFactory(config=settings)
    
    # Generation client (for answers)
    app.generation_client = llm_provider_factory.create(
        provider=settings.GENERATION_BACKEND
    )
    app.generation_client.set_generation_model(
        model_id=settings.GENERATION_MODEL_ID
    )
    logger.info(
        f"✓ Generation client initialized: {settings.GENERATION_BACKEND} "
        f"({settings.GENERATION_MODEL_ID})"
    )
    
    # Evaluation client (separate LLM for evaluation)
    app.generation_client_for_evaluation = llm_provider_factory.create(
        provider=settings.GENERATION_BACKEND_FOR_EVALUATION
    )
    app.generation_client_for_evaluation.set_generation_model(
        model_id=settings.GENERATION_MODEL_ID_FOR_EVALUATION
    )
    logger.info(
        f"✓ Evaluation client initialized: {settings.GENERATION_BACKEND_FOR_EVALUATION} "
        f"({settings.GENERATION_MODEL_ID_FOR_EVALUATION})"
    )
    
    # ==========================================
    # EMBEDDING CLIENT SETUP
    # ==========================================
    embedder_provider_factory = EmbedderProviderFactory(config=settings)
    app.embedding_client = embedder_provider_factory.create(
        provider_name=settings.EMBEDDING_BACKEND
    )
    logger.info(
        f"✓ Embedding client initialized: {settings.EMBEDDING_BACKEND} "
        f"({settings.EMBEDDING_MODEL_ID})"
    )
    
    # ==========================================
    # VECTOR DATABASE SETUP
    # ==========================================
    vectordb_provider_factory = VectorDBProviderFactory(
        config=settings,
        db_client=app.db_client
    )
    app.vectordb_client = vectordb_provider_factory.create(
        provider=settings.VECTOR_DB_BACKEND
    )
    await app.vectordb_client.connect()
    logger.info(f"✓ Vector database connected: {settings.VECTOR_DB_BACKEND}")
    
    # ==========================================
    # TEMPLATE PARSER SETUP
    # ==========================================
    app.template_parser = TemplateParser(
        language=settings.PRIMARY_LANG,
        default_language=settings.DEFAULT_LANG,
    )
    logger.info("✓ Template parser initialized")
    
    # ==========================================
    # MLFLOW TRACKING SETUP
    # ==========================================
    if settings.MLFLOW_ENABLE_TRACKING:
        try:
            from services.tracking.MLflowTrackingService import MLflowTrackingService
            
            tracker = MLflowTrackingService(settings=settings)
            
            if tracker.is_tracking_enabled():
                logger.info(
                    f"✓ MLflow tracking enabled: {settings.get_mlflow_tracking_uri()}"
                )
                
                if settings.is_dagshub_configured():
                    logger.info(
                        f"✓ DagsHub integration: "
                        f"{settings.DAGSHUB_USERNAME}/{settings.DAGSHUB_REPO_NAME}"
                    )
            else:
                logger.warning("⚠ MLflow tracking configured but not available")
                
        except Exception as e:
            logger.error(f"✗ MLflow tracking initialization failed: {e}")
            logger.warning("Continuing without MLflow tracking")
    else:
        logger.info("MLflow tracking disabled")
    
    # ==========================================
    # RAGAS EVALUATION SETUP
    # ==========================================
    enabled_metrics = []
    if settings.RAGAS_ENABLE_FAITHFULNESS:
        enabled_metrics.append("faithfulness")
    if settings.RAGAS_ENABLE_ANSWER_RELEVANCY:
        enabled_metrics.append("answer_relevancy")
    if settings.RAGAS_ENABLE_CONTEXT_PRECISION:
        enabled_metrics.append("context_precision")
    
    logger.info(f"✓ RAGAs evaluation enabled with metrics: {', '.join(enabled_metrics)}")
    
    logger.info("=" * 60)
    logger.info("RAG Application initialized successfully!")
    logger.info("=" * 60)
    logger.info(f"Generation Model: {settings.GENERATION_MODEL_ID}")
    logger.info(f"Evaluation Model: {settings.GENERATION_MODEL_ID_FOR_EVALUATION}")
    logger.info(f"Embedding Model: {settings.EMBEDDING_MODEL_ID}")
    logger.info(f"Vector DB: {settings.VECTOR_DB_BACKEND}")
    logger.info(f"MLflow Tracking: {'Enabled' if settings.MLFLOW_ENABLE_TRACKING else 'Disabled'}")
    logger.info(f"RAGAs Metrics: {', '.join(enabled_metrics)}")
    logger.info("=" * 60)


async def shutdown_span():
    """Cleanup on shutdown."""
    logger.info("Shutting down RAG Application...")
    
    try:
        app.db_engine.dispose()
        logger.info("✓ Database connection closed")
    except Exception as e:
        logger.error(f"Error closing database: {e}")
    
    try:
        await app.vectordb_client.disconnect()
        logger.info("✓ Vector database disconnected")
    except Exception as e:
        logger.error(f"Error disconnecting vector database: {e}")
    
    logger.info("Shutdown complete")

# Register startup/shutdown handlers
app.on_event("startup")(startup_span)
app.on_event("shutdown")(shutdown_span)

app.include_router(base.base_router)
app.include_router(data.data_router)
app.include_router(nlp.nlp_router)

app.include_router(answer_generation.answer_router)  # Answer generation (no evaluation)
app.include_router(evaluation.evaluation_router)     # Evaluation endpoints
app.include_router(tracking.tracking_router)         # MLflow tracking endpoints

logger.info("All routes registered")
