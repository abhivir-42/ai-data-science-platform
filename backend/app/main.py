"""
Main FastAPI application for AI Data Science Platform
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from app.api import agents, data, jobs, health, workflows, simple_auth
from app.api.agent_routes import (
    data_loader, data_cleaning, data_visualization,
    feature_engineering, ml_training, ml_prediction,
)
from app.core.config import settings
from app.core.logging import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    setup_logging()
    os.makedirs(settings.UPLOAD_PATH, exist_ok=True)
    os.makedirs("./temp", exist_ok=True)
    os.makedirs("./temp/models", exist_ok=True)

    try:
        from app.core.database import init_database
        await init_database()
        print("Database initialized successfully")
    except Exception as e:
        print(f"Database initialization failed: {e}")
        raise

    yield


app = FastAPI(
    title="AI Data Science Platform API",
    description="API for executing AI agents and managing data science workflows",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.exists("uploads"):
    app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Core API routers
app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(agents.router, prefix="/api/agents", tags=["agents"])
app.include_router(data.router, prefix="/api/data", tags=["data"])
app.include_router(jobs.router, prefix="/api/jobs", tags=["jobs"])
app.include_router(workflows.router, prefix="/api", tags=["workflows"])
app.include_router(simple_auth.router, prefix="/api/auth", tags=["authentication"])

# Agent-specific routers (consolidated from separate uAgent microservices)
app.include_router(data_loader.router, prefix="/api/agents", tags=["data-loader"])
app.include_router(data_cleaning.router, prefix="/api/agents", tags=["data-cleaning"])
app.include_router(data_visualization.router, prefix="/api/agents", tags=["data-visualization"])
app.include_router(feature_engineering.router, prefix="/api/agents", tags=["feature-engineering"])
app.include_router(ml_training.router, prefix="/api/agents", tags=["ml-training"])
app.include_router(ml_prediction.router, prefix="/api/agents", tags=["ml-prediction"])


@app.get("/")
async def root():
    return {
        "message": "AI Data Science Platform API",
        "version": "0.2.0",
        "docs": "/docs",
        "health": "/api/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
