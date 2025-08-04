"""
Health check API endpoints
"""

from fastapi import APIRouter
from typing import Dict, Any
import sys
import os
from datetime import datetime

from app.core.config import settings

router = APIRouter()


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "0.1.0",
        "environment": settings.ENVIRONMENT,
        "python_version": sys.version,
        "uptime": "N/A"  # TODO: Implement actual uptime tracking
    }


@router.get("/health/detailed")
async def detailed_health_check() -> Dict[str, Any]:
    """Detailed health check with system information"""
    
    # Check upload directory
    upload_dir_exists = os.path.exists(settings.UPLOAD_PATH)
    upload_dir_writable = os.access(settings.UPLOAD_PATH, os.W_OK) if upload_dir_exists else False
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "0.1.0",
        "environment": settings.ENVIRONMENT,
        "python_version": sys.version,
        "system": {
            "platform": sys.platform,
            "python_executable": sys.executable,
        },
        "configuration": {
            "upload_path": settings.UPLOAD_PATH,
            "upload_dir_exists": upload_dir_exists,
            "upload_dir_writable": upload_dir_writable,
            "max_file_size_mb": settings.MAX_FILE_SIZE_MB,
            "allowed_extensions": settings.ALLOWED_EXTENSIONS,
            "debug": settings.DEBUG,
            "log_level": settings.LOG_LEVEL,
        },
        "services": {
            "database": "connected",  # TODO: Add actual database health check
            "redis": "connected",     # TODO: Add actual Redis health check
            "mlflow": "connected",    # TODO: Add actual MLflow health check
        }
    } 