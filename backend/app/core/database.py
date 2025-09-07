"""
Database connection and session management for the AI Data Science Platform.

Provides async SQLAlchemy engine and session management with support for
both SQLite (development) and PostgreSQL (production).
"""

import os
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool
from loguru import logger

from app.core.config import settings
from app.models.session import Base


class DatabaseManager:
    """Manages database connections and sessions"""
    
    def __init__(self):
        self.engine = None
        self.async_session_maker = None
        self._setup_engine()
    
    def _setup_engine(self):
        """Set up the async database engine"""
        database_url = settings.DATABASE_URL
        
        # Handle SQLite vs PostgreSQL
        if database_url.startswith("sqlite:"):
            # Convert sync SQLite URL to async
            if "sqlite:///" in database_url:
                async_database_url = database_url.replace("sqlite:///", "sqlite+aiosqlite:///")
            else:
                async_database_url = database_url.replace("sqlite://", "sqlite+aiosqlite://")
            
            # SQLite-specific engine configuration
            self.engine = create_async_engine(
                async_database_url,
                echo=settings.DEBUG,
                poolclass=StaticPool,
                pool_pre_ping=True,
                connect_args={
                    "check_same_thread": False,
                    "timeout": 20,
                }
            )
            logger.info(f"Using SQLite database: {async_database_url}")
            
        elif database_url.startswith("postgresql:"):
            # Convert sync PostgreSQL URL to async
            async_database_url = database_url.replace("postgresql://", "postgresql+asyncpg://")
            
            # PostgreSQL-specific engine configuration
            self.engine = create_async_engine(
                async_database_url,
                echo=settings.DEBUG,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=3600,
            )
            logger.info(f"Using PostgreSQL database: {async_database_url}")
            
        else:
            raise ValueError(f"Unsupported database URL: {database_url}")
        
        # Create session maker
        self.async_session_maker = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False
        )
    
    async def create_tables(self):
        """Create all database tables"""
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise
    
    async def drop_tables(self):
        """Drop all database tables (useful for testing)"""
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
            logger.info("Database tables dropped successfully")
        except Exception as e:
            logger.error(f"Failed to drop database tables: {e}")
            raise
    
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get an async database session.
        
        Usage:
            async with database_manager.get_session() as session:
                # Use session here
                pass
        """
        if not self.async_session_maker:
            raise RuntimeError("Database not initialized")
        
        async with self.async_session_maker() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def close(self):
        """Close the database engine"""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database engine closed")


# Global database manager instance
database_manager = DatabaseManager()


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency function for FastAPI to get database sessions.
    
    Usage in FastAPI endpoints:
        @router.post("/example")
        async def example_endpoint(session: AsyncSession = Depends(get_db_session)):
            # Use session here
            pass
    """
    async with database_manager.get_session() as session:
        yield session


async def init_database():
    """Initialize the database by creating all tables"""
    try:
        await database_manager.create_tables()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


async def close_database():
    """Close database connections (call during app shutdown)"""
    await database_manager.close()


# Health check function
async def check_database_health() -> bool:
    """Check if database connection is healthy"""
    try:
        async with database_manager.get_session() as session:
            # Simple query to test connection
            await session.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False






