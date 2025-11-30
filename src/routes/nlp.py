from fastapi import APIRouter, status, Request
from fastapi.responses import JSONResponse
from routes.schemes.nlp import (
    PushProjectRequest, 
    PushAssetRequest, 
    SearchRequest,
    AnswerRAGRequest  # NEW: Extended request model
)
from models.ProjectModel import ProjectModel
from models.ChunkModel import ChunkModel
from models.AssetModel import AssetModel
from controllers import NLPController
from models import ResponseSignal
from models.enums.RAGTypeEnum import RAGTypeEnum
from services.chat_memory.DatabaseChatMemory import DatabaseChatMemory
import typing
import logging
import gc
import uuid

logger = logging.getLogger('uvicorn.error')

nlp_router = APIRouter(
    prefix="/api/v1/nlp",
    tags=["api_v1", "nlp"],
)

# Helper function to create NLP controller with chat memory
def create_nlp_controller(request: Request) -> NLPController:
    """
    Factory function to create NLP controller with all dependencies.
    """
    chat_memory = DatabaseChatMemory(db_client=request.app.db_client)
    
    return NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
        chat_memory=chat_memory
    )

# ==========================================
# INDEXING ENDPOINTS (UNCHANGED)
# ==========================================

@nlp_router.post("/index/push/project/{project_id}")
async def index_project(
    request: Request, 
    project_id: int, 
    push_request: PushProjectRequest
):
    """Index all chunks for a project into vector database."""
    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )
    chunk_model = await ChunkModel.create_instance(
        db_client=request.app.db_client
    )

    project = await project_model.get_project_or_create_one(project_id=project_id)
    if not project:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignal.PROJECT_NOT_FOUND_ERROR.value}
        )
    
    nlp_controller = create_nlp_controller(request)
    
    collection_name = nlp_controller.create_collection_name(
        project_id=typing.cast(int, project.project_id)
    )

    _ = await request.app.vectordb_client.create_collection(
        collection_name=collection_name,
        embedding_size=request.app.embedding_client.embedding_size,
        do_reset=push_request.do_reset,
    )

    total_chunks_count = await chunk_model.get_total_chunks_count_per_project(
        project_id=typing.cast(int, project.project_id)
    )
    
    inserted_items_count = 0
    page_no = 1
    page_size = 50
    
    logger.info(f"Starting indexing of {total_chunks_count} chunks for project {project_id}")

    while True:
        page_data = await chunk_model.get_project_chunks_minimal(
            project_id=typing.cast(int, project.project_id),
            page_no=page_no,
            page_size=page_size
        )
        
        if not page_data or len(page_data) == 0:
            break

        try:
            is_inserted = await nlp_controller.index_into_vector_db(
                project=project,
                chunks_data=page_data,
                collection_name=collection_name
            )

            if not is_inserted:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"signal": ResponseSignal.INSERT_INTO_VECTORDB_ERROR.value}
                )

            inserted_items_count += len(page_data)
            
            if page_no % 10 == 0:
                logger.info(
                    f"Progress: {inserted_items_count}/{total_chunks_count} chunks "
                    f"({(inserted_items_count/total_chunks_count)*100:.1f}%)"
                )
            
        finally:
            del page_data
            if page_no % 5 == 0:
                gc.collect()
        
        page_no += 1

    gc.collect()
    logger.info(f"Completed indexing of {inserted_items_count} chunks for project {project_id}")
    
    return JSONResponse(
        content={
            "signal": ResponseSignal.INSERT_INTO_VECTORDB_SUCCESS.value,
            "inserted_items_count": inserted_items_count
        }
    )

# ==========================================
# 1. OPTIMIZED ENDPOINT WITH MEMORY MANAGEMENT
# ==========================================

@nlp_router.post("/index/push/asset/{project_id}")
async def index_asset(request: Request, project_id: int, push_request: PushAssetRequest):
    """Index all chunks for a specific asset into vector database."""

    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )

    chunk_model = await ChunkModel.create_instance(
        db_client=request.app.db_client
    )

    asset_model = await AssetModel.create_instance(
        db_client=request.app.db_client
    )

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )

    if not project:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseSignal.PROJECT_NOT_FOUND_ERROR.value
            }
        )

    asset = await asset_model.get_asset_record_by_id(
        asset_id=push_request.asset_id
    )

    if not asset:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseSignal.ASSET_NOT_FOUND_ERROR.value
            }
        )
    
    nlp_controller = create_nlp_controller(request=request)

    # Create collection once
    collection_name = nlp_controller.create_collection_name(
        project_id=typing.cast(int, project.project_id)
    )

    _ = await request.app.vectordb_client.create_collection(
        collection_name=collection_name,
        embedding_size=request.app.embedding_client.embedding_size,
        do_reset=False,
    )

    if push_request.do_reset:
        chunk_ids = await chunk_model.get_chunk_ids_by_asset_id(
            asset_id=asset.asset_id
        )
        deleted_count = await request.app.vectordb_client.delete_many_by_chunk_ids(
            collection_name=collection_name,
            chunk_ids=chunk_ids
        )
        logger.info(f"Deleted {deleted_count} vectors for asset ID {asset.asset_id}")
        
        # Clear the chunk_ids list immediately
        del chunk_ids
        gc.collect()

    # Get total count
    total_chunks_count = await chunk_model.get_total_chunks_count_per_asset(
        asset_id=typing.cast(int, asset.asset_id)
    )
    
    # Use simpler progress tracking to avoid holding references
    inserted_items_count = 0
    page_no = 1
    page_size = 50  # Match your original page size
    
    logger.info(f"Starting indexing of {total_chunks_count} chunks")

    while True:
        # Get minimal data - convert to dict immediately to release ORM objects
        page_data = await chunk_model.get_asset_chunks_minimal(
            asset_id=typing.cast(int, asset.asset_id),
            page_no=page_no,
            page_size=page_size
        )
        
        if not page_data or len(page_data) == 0:
            break

        # Process this page
        try:
            is_inserted = await nlp_controller.index_into_vector_db(
                project=project,
                chunks_data=page_data,  # Pass dict data, not ORM objects
                collection_name=collection_name
            )

            if not is_inserted:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={
                        "signal": ResponseSignal.INSERT_INTO_VECTORDB_ERROR.value
                    }
                )

            inserted_items_count += len(page_data)
            
            # Log progress every 10 pages
            if page_no % 10 == 0:
                logger.info(f"Processed {inserted_items_count}/{total_chunks_count} chunks")
            
        finally:
            # CRITICAL: Explicitly delete references
            del page_data
            
            # Force garbage collection every 5 pages
            if page_no % 5 == 0:
                gc.collect()
        
        page_no += 1

    # Final cleanup
    gc.collect()
    
    logger.info(f"Completed indexing of {inserted_items_count} chunks")
    
    return JSONResponse(
        content={
            "signal": ResponseSignal.INSERT_INTO_VECTORDB_SUCCESS.value,
            "inserted_items_count": inserted_items_count
        }
    )

@nlp_router.get("/index/info/{project_id}")
async def get_project_index_info(request: Request, project_id: int):
    """Get vector database collection info."""
    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )
    project = await project_model.get_project_or_create_one(project_id=project_id)
    
    nlp_controller = create_nlp_controller(request)
    collection_info = await nlp_controller.get_vector_db_collection_info(project=project)

    return JSONResponse(
        content={
            "signal": ResponseSignal.VECTORDB_COLLECTION_RETRIEVED.value,
            "collection_info": collection_info
        }
    )

@nlp_router.post("/index/search/{project_id}")
async def search_index(
    request: Request, 
    project_id: int, 
    search_request: SearchRequest
):
    """Simple vector search (for testing)."""
    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )
    project = await project_model.get_project_or_create_one(project_id=project_id)
    
    nlp_controller = create_nlp_controller(request)
    results = await nlp_controller.search_vector_db_collection(
        project=project, 
        text=search_request.text, 
        limit=search_request.limit
    )

    if not results:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignal.VECTORDB_SEARCH_ERROR.value}
        )
    
    return JSONResponse(
        content={
            "signal": ResponseSignal.VECTORDB_SEARCH_SUCCESS.value,
            "results": [result.dict() for result in results]
        }
    )

# ==========================================
# NEW: ENHANCED RAG ANSWER ENDPOINT
# ==========================================

@nlp_router.post("/index/answer/{project_id}")
async def answer_rag(
    request: Request, 
    project_id: int, 
    answer_request: AnswerRAGRequest
):
    """
    Answer RAG question with chat memory and multiple RAG strategies.
    
    NEW FEATURES:
    - Chat memory (persistent conversation history)
    - Multiple RAG strategies (basic/fusion/rerank)
    - Session management
    - Automatic session ID generation
    
    Request Body:
        {
            "text": "Your question here",
            "limit": 10,
            "session_id": "optional-session-id",  # Auto-generated if not provided
            "rag_type": "basic",  # Options: basic, fusion, rerank
            "chat_history_limit": 10  # Max messages from history to include
        }
    """
    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )
    project = await project_model.get_project_or_create_one(project_id=project_id)
    
    if not project:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignal.PROJECT_NOT_FOUND_ERROR.value}
        )
    
    # Generate session ID if not provided
    session_id = answer_request.session_id or str(uuid.uuid4())
    
    # Validate RAG type
    rag_type = answer_request.rag_type or RAGTypeEnum.BASIC.value
    if rag_type not in [e.value for e in RAGTypeEnum]:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": "INVALID_RAG_TYPE",
                "message": f"Invalid RAG type: {rag_type}",
                "available_types": [e.value for e in RAGTypeEnum]
            }
        )
    
    nlp_controller = create_nlp_controller(request)
    
    # Execute RAG with strategy and chat memory
    answer, full_prompt, chat_history, strategy_name = await nlp_controller.answer_rag_question(
        project=project,
        query=answer_request.text,
        session_id=session_id,
        limit=answer_request.limit,
        rag_type=rag_type,
        chat_history_limit=answer_request.chat_history_limit
    )

    if not answer:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignal.RAG_ANSWER_ERROR.value}
        )
    
    return JSONResponse(
        content={
            "signal": ResponseSignal.RAG_ANSWER_SUCCESS.value,
            "answer": answer,
            "session_id": session_id,
            "rag_strategy": strategy_name,
            "rag_type": rag_type,
            "full_prompt": full_prompt,
            "chat_history_length": len(chat_history)
        }
    )

# ==========================================
# NEW: CHAT MEMORY MANAGEMENT ENDPOINTS
# ==========================================

@nlp_router.delete("/chat/session/{project_id}/{session_id}")
async def clear_chat_session(
    request: Request,
    project_id: int,
    session_id: str
):
    """Clear chat history for a specific session."""
    nlp_controller = create_nlp_controller(request)
    
    success = await nlp_controller.clear_chat_session(
        session_id=session_id,
        project_id=project_id
    )
    
    if success:
        return JSONResponse(
            content={
                "signal": "CHAT_SESSION_CLEARED",
                "session_id": session_id
            }
        )
    else:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": "CHAT_SESSION_CLEAR_ERROR"}
        )

@nlp_router.get("/chat/sessions/{project_id}")
async def get_chat_sessions(
    request: Request,
    project_id: int,
    limit: int = 10
):
    """Get list of recent chat sessions for a project."""
    nlp_controller = create_nlp_controller(request)
    
    sessions = await nlp_controller.get_chat_sessions(
        project_id=project_id,
        limit=limit
    )
    
    return JSONResponse(
        content={
            "signal": "CHAT_SESSIONS_RETRIEVED",
            "project_id": project_id,
            "sessions": sessions,
            "count": len(sessions)
        }
    )

@nlp_router.get("/rag/strategies")
async def get_available_rag_strategies():
    """Get list of available RAG strategies."""
    return JSONResponse(
        content={
            "signal": "RAG_STRATEGIES_RETRIEVED",
            "strategies": [
                {
                    "type": RAGTypeEnum.BASIC.value,
                    "name": "Basic RAG",
                    "description": "Single query → Vector search → Generate answer"
                },
                {
                    "type": RAGTypeEnum.FUSION.value,
                    "name": "Fusion RAG",
                    "description": "Query expansion → Multiple searches → Reciprocal Rank Fusion"
                },
                {
                    "type": RAGTypeEnum.RERANK.value,
                    "name": "ReRank RAG",
                    "description": "Retrieve many → Cross-encoder reranking → Generate"
                }
            ]
        }
    )