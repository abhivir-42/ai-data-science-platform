"""
Application configuration settings
"""

import os
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # Database
    DATABASE_URL: str = "sqlite:///./ai_data_science_dev.db"
    DATABASE_URL_DEV: str = "sqlite:///./ai_data_science_dev.db"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"
    
    # API Keys
    OPENAI_API_KEY: Optional[str] = None
    AGENTVERSE_API_TOKEN: Optional[str] = None
    
    # Application
    SECRET_KEY: str = "dev-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # File Storage
    UPLOAD_PATH: str = "./uploads"
    MAX_FILE_SIZE_MB: int = 100
    ALLOWED_EXTENSIONS: List[str] = Field(default=[".csv", ".xlsx", ".json", ".parquet", ".pdf"])
    
    # MLflow
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "ai_data_science_platform"
    
    # H2O AutoML
    H2O_NTHREADS: int = -1
    H2O_MAX_MEM_SIZE: str = "4g"
    
    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    
    # CORS
    ALLOWED_ORIGINS: List[str] = Field(default=["http://localhost:3000", "http://127.0.0.1:3000"])
    
    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 8001
    
    class Config:
        env_file = "../.env"
        case_sensitive = True
        env_file_encoding = 'utf-8'


# Global settings instance
settings = Settings() 