"""
Main FastAPI application for AI Data Science Platform
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from app.api import agents, data, jobs, health, workflows
from app.core.config import settings
from app.core.logging import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    setup_logging()
    
    # Create upload directory if it doesn't exist
    os.makedirs(settings.UPLOAD_PATH, exist_ok=True)
    
    yield
    
    # Shutdown
    pass


# Create FastAPI app
app = FastAPI(
    title="AI Data Science Platform API",
    description="API for executing AI agents and managing data science workflows",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
if os.path.exists("uploads"):
    app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Include routers
app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(agents.router, prefix="/api/agents", tags=["agents"])
app.include_router(data.router, prefix="/api/data", tags=["data"])
app.include_router(jobs.router, prefix="/api/jobs", tags=["jobs"])
app.include_router(workflows.router, prefix="/api", tags=["workflows"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Data Science Platform API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/api/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    ) 