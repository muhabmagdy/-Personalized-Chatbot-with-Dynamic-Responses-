from .BaseDataModel import BaseDataModel
from .db_schemes import DataChunk
from .enums.DataBaseEnum import DataBaseEnum
from bson.objectid import ObjectId
from pymongo import InsertOne
from sqlalchemy.future import select
from sqlalchemy import func, delete
from sqlalchemy.orm import sessionmaker


class ChunkModel(BaseDataModel):

    def __init__(self, db_client: sessionmaker):
        super().__init__(db_client=db_client)
        self.db_client = db_client

    @classmethod
    async def create_instance(cls, db_client: sessionmaker):
        instance = cls(db_client)
        return instance

    async def create_chunk(self, chunk: DataChunk):
        async with self.db_client() as session:
            async with session.begin():
                session.add(chunk)
            await session.commit()
            await session.refresh(chunk)
        return chunk

    async def get_chunk(self, chunk_id: str):
        async with self.db_client() as session:
            result = await session.execute(select(DataChunk).where(DataChunk.chunk_id == chunk_id))
            chunk = result.scalar_one_or_none()
        return chunk
    
    async def get_chunk_ids_by_asset_id(self, asset_id: int) -> list[int]:
        """
        Retrieves a list of all chunk IDs (int) associated with a specific asset_id.

        Args:
            asset_id (int): The ID of the asset whose chunks are to be retrieved.
        
        Returns:
            list[int]: A list of chunk IDs. Returns an empty list if no chunks are found.
        """
        async with self.db_client() as session:
            # 1. Construct the SELECT statement
            #    - select(DataChunk.chunk_id): Only selects the 'chunk_id' column, 
            #      not the entire DataChunk object, making the query faster.
            #    - where(...): Filters the chunks by the matching asset ID.
            stmt = select(DataChunk.chunk_id).where(
                DataChunk.chunk_asset_id == asset_id
            )
            
            # 2. Execute the statement
            result = await session.execute(stmt)
            
            # 3. Process the results
            #    - result.scalars(): Fetches only the values of the first column (chunk_id).
            #    - .all(): Returns the results as a list of those values (list[int]).
            chunk_ids = result.scalars().all()
            
        return chunk_ids

    async def insert_many_chunks(self, chunks: list, batch_size: int=100):
        async with self.db_client() as session:
            async with session.begin():
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i+batch_size]
                    session.add_all(batch)
            await session.commit()
        return len(chunks)

    async def delete_chunks_by_project_id(self, project_id: int):
        async with self.db_client() as session:
            stmt = delete(DataChunk).where(DataChunk.chunk_project_id == project_id)
            result = await session.execute(stmt)
            await session.commit()
        return result.rowcount
    
    async def delete_chunks_by_asset_id(self, asset_id: int):
        async with self.db_client() as session:
            stmt = delete(DataChunk).where(DataChunk.chunk_asset_id == asset_id)
            result = await session.execute(stmt)
            await session.commit()
        return result.rowcount
    
    async def get_project_chunks(self, project_id: int, page_no: int=1, page_size: int=50):
        async with self.db_client() as session:
            stmt = select(DataChunk).where(DataChunk.chunk_project_id == project_id).offset((page_no - 1) * page_size).limit(page_size)
            result = await session.execute(stmt)
            records = result.scalars().all()
        return 
    
    async def get_asset_chunks(self, asset_id: int, page_no: int=1, page_size: int=50):
        async with self.db_client() as session:
            stmt = select(DataChunk).where(DataChunk.chunk_asset_id == asset_id).offset((page_no - 1) * page_size).limit(page_size)
            result = await session.execute(stmt)
            records = result.scalars().all()
        return records
    
    async def get_total_chunks_count_per_project(self, project_id: int):
        total_count = 0
        async with self.db_client() as session:
            count_sql = select(func.count(DataChunk.chunk_id)).where(DataChunk.chunk_project_id == project_id)
            records_count = await session.execute(count_sql)
            total_count = records_count.scalar()
        
        return total_count
    
    async def get_total_chunks_count_per_asset(self, asset_id: int):
        total_count = 0
        async with self.db_client() as session:
            count_sql = select(func.count(DataChunk.chunk_id)).where(DataChunk.chunk_asset_id == asset_id)
            records_count = await session.execute(count_sql)
            total_count = records_count.scalar()
        
        return 
    
    # ==========================================
    # 2. OPTIMIZED CHUNK MODEL METHOD
    # ==========================================

    async def get_project_chunks_minimal(
        self, 
        project_id: int, 
        page_no: int = 1, 
        page_size: int = 50
    ):
        """
        Returns chunks as dictionaries instead of ORM objects to reduce memory usage.
        This prevents SQLAlchemy from keeping objects in the session identity map.
        
        Args:
            project_id: The project ID to fetch chunks for
            page_no: Page number (1-indexed)
            page_size: Number of chunks per page
            
        Returns:
            List of dictionaries with chunk data
        """
        async with self.db_client() as session:
            stmt = (
                select(
                    DataChunk.chunk_id,
                    DataChunk.chunk_text,
                    DataChunk.chunk_metadata
                )
                .where(DataChunk.chunk_project_id == project_id)
                .offset((page_no - 1) * page_size)
                .limit(page_size)
            )
            
            result = await session.execute(stmt)
            rows = result.all()
            
            # Convert to list of dicts immediately to release ORM objects
            chunks_data = [
                {
                    "chunk_id": row.chunk_id,
                    "chunk_text": row.chunk_text,
                    "chunk_metadata": row.chunk_metadata
                }
                for row in rows
            ]
            
            # Explicitly expunge all objects from session to clear identity map
            session.expunge_all()
            
        return chunks_data

    # ==========================================
    # 2. OPTIMIZED CHUNK MODEL - RETURN DICTS NOT ORM OBJECTS
    # ==========================================

    async def get_asset_chunks_minimal(
        self, 
        asset_id: int, 
        page_no: int = 1, 
        page_size: int = 50
    ):
        """
        Returns chunks as dictionaries instead of ORM objects to reduce memory usage.
        This prevents SQLAlchemy from keeping objects in the session identity map.
        """
        async with self.db_client() as session:
            stmt = (
                select(
                    DataChunk.chunk_id,
                    DataChunk.chunk_text,
                    DataChunk.chunk_metadata
                )
                .where(DataChunk.chunk_asset_id == asset_id)
                .offset((page_no - 1) * page_size)
                .limit(page_size)
            )
            
            result = await session.execute(stmt)
            rows = result.all()
            
            # Convert to list of dicts immediately
            chunks_data = [
                {
                    "chunk_id": row.chunk_id,
                    "chunk_text": row.chunk_text,
                    "chunk_metadata": row.chunk_metadata
                }
                for row in rows
            ]
            
            # Explicitly expunge all objects from session
            session.expunge_all()
            
        return chunks_data


