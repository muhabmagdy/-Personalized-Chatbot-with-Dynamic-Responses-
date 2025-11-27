from fastapi import FastAPI, APIRouter, status, Request
from fastapi.responses import JSONResponse
from routes.schemes.nlp import PushProjectRequest, PushAssetRequest, SearchRequest
from models.ProjectModel import ProjectModel
from models.ChunkModel import ChunkModel
from models.AssetModel import AssetModel
from controllers import NLPController
from models import ResponseSignal
from tqdm.auto import tqdm
import typing
import logging
import gc

logger = logging.getLogger('uvicorn.error')

nlp_router = APIRouter(
    prefix="/api/v1/nlp",
    tags=["api_v1", "nlp"],
)

# ==========================================
# 1. OPTIMIZED PROJECT INDEXING ENDPOINT
# ==========================================

@nlp_router.post("/index/push/project/{project_id}")
async def index_project(request: Request, project_id: int, push_request: PushProjectRequest):

    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )

    chunk_model = await ChunkModel.create_instance(
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
    
    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
    )

    # Create collection once
    collection_name = nlp_controller.create_collection_name(
        project_id=typing.cast(int, project.project_id)
    )

    _ = await request.app.vectordb_client.create_collection(
        collection_name=collection_name,
        embedding_size=request.app.embedding_client.embedding_size,
        do_reset=push_request.do_reset,
    )

    # Get total count for logging
    total_chunks_count = await chunk_model.get_total_chunks_count_per_project(
        project_id=typing.cast(int, project.project_id)
    )
    
    # Initialize counters
    inserted_items_count = 0
    page_no = 1
    page_size = 50  # Match your page size
    
    logger.info(f"Starting indexing of {total_chunks_count} chunks for project {project_id}")

    while True:
        # MEMORY OPTIMIZATION: Get chunks as dictionaries, not ORM objects
        page_data = await chunk_model.get_project_chunks_minimal(
            project_id=typing.cast(int, project.project_id),
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
            
            # Log progress every 10 pages (500 chunks)
            if page_no % 10 == 0:
                logger.info(
                    f"Progress: {inserted_items_count}/{total_chunks_count} chunks "
                    f"({(inserted_items_count/total_chunks_count)*100:.1f}%)"
                )
            
        finally:
            # CRITICAL: Explicitly delete references to free memory
            del page_data
            
            # Force garbage collection every 5 pages (250 chunks)
            if page_no % 5 == 0:
                gc.collect()
        
        page_no += 1

    # Final cleanup
    gc.collect()
    
    logger.info(f"Completed indexing of {inserted_items_count} chunks for project {project_id}")
    
    return JSONResponse(
        content={
            "signal": ResponseSignal.INSERT_INTO_VECTORDB_SUCCESS.value,
            "inserted_items_count": inserted_items_count
        }
    )

# @nlp_router.post("/index/push/project/{project_id}")
# async def index_project(request: Request, project_id: int, push_request: PushProjectRequest):

#     project_model = await ProjectModel.create_instance(
#         db_client=request.app.db_client
#     )

#     chunk_model = await ChunkModel.create_instance(
#         db_client=request.app.db_client
#     )

#     project = await project_model.get_project_or_create_one(
#         project_id=project_id
#     )

#     if not project:
#         return JSONResponse(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             content={
#                 "signal": ResponseSignal.PROJECT_NOT_FOUND_ERROR.value
#             }
#         )
    
#     nlp_controller = NLPController(
#         vectordb_client=request.app.vectordb_client,
#         generation_client=request.app.generation_client,
#         embedding_client=request.app.embedding_client,
#         template_parser=request.app.template_parser,
#     )

#     has_records = True
#     page_no = 1
#     inserted_items_count = 0
#     idx = 0

#     # create collection if not exists
#     collection_name = nlp_controller.create_collection_name(project_id=typing.cast(int, project.project_id))

#     _ = await request.app.vectordb_client.create_collection(
#         collection_name=collection_name,
#         embedding_size=request.app.embedding_client.embedding_size,
#         do_reset=push_request.do_reset,
#     )

#     # setup batching
#     total_chunks_count = await chunk_model.get_total_chunks_count_per_project(project_id=typing.cast(int, project.project_id))
#     pbar = tqdm(total=total_chunks_count, desc="Vector Indexing", position=0)

#     while has_records:
#         page_chunks = await chunk_model.get_project_chunks(project_id=typing.cast(int, project.project_id), page_no=page_no)
#         if len(page_chunks):
#             page_no += 1
        
#         if not page_chunks or len(page_chunks) == 0:
#             has_records = False
#             break

#         chunks_ids =  [ c.chunk_id for c in page_chunks ]
#         idx += len(page_chunks)
        
#         is_inserted = await nlp_controller.index_into_vector_db(
#             project=project,
#             chunks=page_chunks,
#             chunks_ids=chunks_ids
#         )

#         if not is_inserted:
#             return JSONResponse(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 content={
#                     "signal": ResponseSignal.INSERT_INTO_VECTORDB_ERROR.value
#                 }
#             )

#         pbar.update(len(page_chunks))
#         inserted_items_count += len(page_chunks)
        
#     return JSONResponse(
#         content={
#             "signal": ResponseSignal.INSERT_INTO_VECTORDB_SUCCESS.value,
#             "inserted_items_count": inserted_items_count
#         }
#     )

# @nlp_router.post("/index/push/asset/{project_id}")
# async def index_asset(request: Request, project_id: int, push_request: PushAssetRequest):

#     project_model = await ProjectModel.create_instance(
#         db_client=request.app.db_client
#     )

#     chunk_model = await ChunkModel.create_instance(
#         db_client=request.app.db_client
#     )

#     asset_model = await AssetModel.create_instance(
#         db_client=request.app.db_client
#     )

#     project = await project_model.get_project_or_create_one(
#         project_id=project_id
#     )

#     if not project:
#         return JSONResponse(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             content={
#                 "signal": ResponseSignal.PROJECT_NOT_FOUND_ERROR.value
#             }
#         )

#     asset = await asset_model.get_asset_record_by_id(
#         asset_id=push_request.asset_id
#     )

#     if not asset:
#         return JSONResponse(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             content={
#                 "signal": ResponseSignal.ASSET_NOT_FOUND_ERROR.value
#             }
#         )
    
#     nlp_controller = NLPController(
#         vectordb_client=request.app.vectordb_client,
#         generation_client=request.app.generation_client,
#         embedding_client=request.app.embedding_client,
#         template_parser=request.app.template_parser,
#     )

#     has_records = True
#     page_no = 1
#     inserted_items_count = 0
#     idx = 0

#     # create collection if not exists
#     collection_name = nlp_controller.create_collection_name(project_id=typing.cast(int, project.project_id))

#     _ = await request.app.vectordb_client.create_collection(
#         collection_name=collection_name,
#         embedding_size=request.app.embedding_client.embedding_size,
#         do_reset=False,
#     )

#     if push_request.do_reset:
#         # delete existing chunks for the asset
#         chuns_ids =  await chunk_model.get_chunk_ids_by_asset_id(asset_id=asset.asset_id)
#         deleted_count = await request.app.vectordb_client.delete_many_by_chunk_ids(
#             collection_name=collection_name,
#             chunk_ids=chuns_ids
#         )
#         logger.info(f"Deleted {deleted_count} vectors for asset ID {asset.asset_id}")

#     # setup batching
#     total_chunks_count = await chunk_model.get_total_chunks_count_per_asset(asset_id=typing.cast(int, asset.asset_id))
#     pbar = tqdm(total=total_chunks_count, desc="Vector Indexing", position=0)

#     while has_records:
#         page_chunks = await chunk_model.get_asset_chunks(asset_id=typing.cast(int, asset.asset_id), page_no=page_no)
#         if len(page_chunks):
#             page_no += 1
        
#         if not page_chunks or len(page_chunks) == 0:
#             has_records = False
#             break

#         chunks_ids =  [ c.chunk_id for c in page_chunks ]
#         idx += len(page_chunks)
        
#         is_inserted = await nlp_controller.index_into_vector_db(
#             project=project,
#             chunks=page_chunks,
#             chunks_ids=chunks_ids,
#             do_reset=False
#         )

#         if not is_inserted:
#             return JSONResponse(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 content={
#                     "signal": ResponseSignal.INSERT_INTO_VECTORDB_ERROR.value
#                 }
#             )

#         pbar.update(len(page_chunks))
#         inserted_items_count += len(page_chunks)

#         gc.collect()
        
#     return JSONResponse(
#         content={
#             "signal": ResponseSignal.INSERT_INTO_VECTORDB_SUCCESS.value,
#             "inserted_items_count": inserted_items_count
#         }
#     )

# ==========================================
# 1. OPTIMIZED ENDPOINT WITH MEMORY MANAGEMENT
# ==========================================

@nlp_router.post("/index/push/asset/{project_id}")
async def index_asset(request: Request, project_id: int, push_request: PushAssetRequest):
    
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
    
    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
    )

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
    
    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )

    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
    )

    collection_info = await nlp_controller.get_vector_db_collection_info(project=project)

    return JSONResponse(
        content={
            "signal": ResponseSignal.VECTORDB_COLLECTION_RETRIEVED.value,
            "collection_info": collection_info
        }
    )

@nlp_router.post("/index/search/{project_id}")
async def search_index(request: Request, project_id: int, search_request: SearchRequest):
    
    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )

    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
    )

    results = await nlp_controller.search_vector_db_collection(
        project=project, text=search_request.text, limit=search_request.limit
    )

    if not results:
        return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "signal": ResponseSignal.VECTORDB_SEARCH_ERROR.value
                }
            )
    
    return JSONResponse(
        content={
            "signal": ResponseSignal.VECTORDB_SEARCH_SUCCESS.value,
            "results": [ result.dict()  for result in results ]
        }
    )

@nlp_router.post("/index/answer/{project_id}")
async def answer_rag(request: Request, project_id: int, search_request: SearchRequest):
    
    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )

    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
    )

    answer, full_prompt, chat_history = await nlp_controller.answer_rag_question(
        project=project,
        query=search_request.text,
        limit=search_request.limit,
    )

    if not answer:
        return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "signal": ResponseSignal.RAG_ANSWER_ERROR.value
                }
        )
    
    return JSONResponse(
        content={
            "signal": ResponseSignal.RAG_ANSWER_SUCCESS.value,
            "answer": answer,
            "full_prompt": full_prompt,
            "chat_history": chat_history
        }
    )
