"""
Training API endpoints that proxy to the H2O ML agent
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
import httpx
import asyncio
from loguru import logger

from app.core.config import settings

router = APIRouter()

# H2O ML Agent configuration
H2O_ML_AGENT_URL = "http://h2o-ml-agent:8008"

class SessionRequest(BaseModel):
    session_id: str
    user_id: Optional[str] = None

class TrainModelRequest(BaseModel):
    data: Dict[str, List[Any]]
    target_variable: str
    user_instructions: Optional[str] = "Train machine learning models"
    max_runtime_secs: int = 300
    cv_folds: int = 5
    balance_classes: bool = True
    exclude_algos: List[str] = []
    max_models: int = 20
    seed: int = 42
    max_retries: int = 3

class TrainModelFromSessionRequest(BaseModel):
    source_session_id: str
    target_variable: str
    user_instructions: Optional[str] = "Train machine learning models"
    max_runtime_secs: int = 300
    cv_folds: int = 5
    balance_classes: bool = True
    exclude_algos: List[str] = []
    max_models: int = 20
    seed: int = 42
    max_retries: int = 3

class LeaderboardResponse(BaseModel):
    success: bool
    message: str
    leaderboard: Optional[Dict[str, Any]] = None
    best_model_id: Optional[str] = None
    error: Optional[str] = None

class SessionResponse(BaseModel):
    success: bool
    message: str
    session_id: str
    execution_time_seconds: Optional[float] = None
    error: Optional[str] = None

class CodeResponse(BaseModel):
    success: bool
    message: str
    code: Optional[str] = None
    error: Optional[str] = None

class GenericResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None
    error: Optional[str] = None

class ModelInfoResponse(BaseModel):
    success: bool
    message: str
    model_id: Optional[str] = None
    error: Optional[str] = None

async def _make_request_to_h2o_agent(endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Make a request to the H2O ML agent"""
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{H2O_ML_AGENT_URL}{endpoint}",
                json=data,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Request to H2O ML agent timed out")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"H2O ML agent error: {e.response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to communicate with H2O ML agent: {str(e)}")

@router.post("/train-model", response_model=SessionResponse)
async def train_model(request: TrainModelRequest) -> SessionResponse:
    """Train ML model with H2O AutoML"""
    try:
        logger.info(f"Training model with target variable: {request.target_variable}")
        
        response = await _make_request_to_h2o_agent("/train-model", request.dict())
        
        return SessionResponse(
            success=response.get("success", False),
            message=response.get("message", ""),
            session_id=response.get("session_id", ""),
            execution_time_seconds=response.get("execution_time_seconds"),
            error=response.get("error")
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to train model: {e}")
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")

@router.post("/train-model-from-session", response_model=SessionResponse)
async def train_model_from_session(request: TrainModelFromSessionRequest) -> SessionResponse:
    """Train ML model using data from a previous session"""
    try:
        logger.info(f"Training model from session: {request.source_session_id}")
        
        response = await _make_request_to_h2o_agent("/train-model-from-session", request.dict())
        
        return SessionResponse(
            success=response.get("success", False),
            message=response.get("message", ""),
            session_id=response.get("session_id", ""),
            execution_time_seconds=response.get("execution_time_seconds"),
            error=response.get("error")
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to train model from session: {e}")
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")

@router.post("/get-leaderboard", response_model=LeaderboardResponse)
async def get_leaderboard(request: SessionRequest) -> LeaderboardResponse:
    """Get leaderboard from training session"""
    try:
        logger.info(f"Getting leaderboard for session: {request.session_id}")
        
        response = await _make_request_to_h2o_agent("/get-leaderboard", request.dict())
        
        return LeaderboardResponse(
            success=response.get("success", False),
            message=response.get("message", ""),
            leaderboard=response.get("leaderboard"),
            best_model_id=response.get("best_model_id"),
            error=response.get("error")
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get leaderboard: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get leaderboard: {str(e)}")

@router.post("/get-best-model-id", response_model=ModelInfoResponse)
async def get_best_model_id(request: SessionRequest) -> ModelInfoResponse:
    """Get best model ID from training session"""
    try:
        logger.info(f"Getting best model ID for session: {request.session_id}")
        
        response = await _make_request_to_h2o_agent("/get-best-model-id", request.dict())
        
        return ModelInfoResponse(
            success=response.get("success", False),
            message=response.get("message", ""),
            model_id=response.get("model_id"),
            error=response.get("error")
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get best model ID: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get best model ID: {str(e)}")

@router.post("/get-model-path", response_model=ModelInfoResponse)
async def get_model_path(request: SessionRequest) -> ModelInfoResponse:
    """Get model path from training session"""
    try:
        logger.info(f"Getting model path for session: {request.session_id}")
        
        response = await _make_request_to_h2o_agent("/get-model-path", request.dict())
        
        return ModelInfoResponse(
            success=response.get("success", False),
            message=response.get("message", ""),
            model_id=response.get("model_id"),  # Using model_id field for model_path
            error=response.get("error")
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model path: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model path: {str(e)}")

@router.post("/get-training-function", response_model=CodeResponse)
async def get_training_function(request: SessionRequest) -> CodeResponse:
    """Get training function code from session"""
    try:
        logger.info(f"Getting training function for session: {request.session_id}")
        
        response = await _make_request_to_h2o_agent("/get-training-function", request.dict())
        
        return CodeResponse(
            success=response.get("success", False),
            message=response.get("message", ""),
            code=response.get("code"),
            error=response.get("error")
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get training function: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get training function: {str(e)}")

@router.post("/get-ml-steps", response_model=GenericResponse)
async def get_ml_steps(request: SessionRequest) -> GenericResponse:
    """Get ML steps from session"""
    try:
        logger.info(f"Getting ML steps for session: {request.session_id}")
        
        response = await _make_request_to_h2o_agent("/get-ml-steps", request.dict())
        
        return GenericResponse(
            success=response.get("success", False),
            message=response.get("message", ""),
            data=response.get("data"),
            error=response.get("error")
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get ML steps: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get ML steps: {str(e)}")

@router.post("/get-original-data", response_model=GenericResponse)
async def get_original_data(request: SessionRequest) -> GenericResponse:
    """Get original data from session"""
    try:
        logger.info(f"Getting original data for session: {request.session_id}")
        
        response = await _make_request_to_h2o_agent("/get-original-data", request.dict())
        
        return GenericResponse(
            success=response.get("success", False),
            message=response.get("message", ""),
            data=response.get("data"),
            error=response.get("error")
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get original data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get original data: {str(e)}")

@router.post("/get-workflow-summary", response_model=GenericResponse)
async def get_workflow_summary(request: SessionRequest) -> GenericResponse:
    """Get workflow summary from session"""
    try:
        logger.info(f"Getting workflow summary for session: {request.session_id}")
        
        response = await _make_request_to_h2o_agent("/get-workflow-summary", request.dict())
        
        return GenericResponse(
            success=response.get("success", False),
            message=response.get("message", ""),
            data=response.get("data"),
            error=response.get("error")
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get workflow summary: {str(e)}")

@router.post("/get-logs", response_model=GenericResponse)
async def get_logs(request: SessionRequest) -> GenericResponse:
    """Get logs from session"""
    try:
        logger.info(f"Getting logs for session: {request.session_id}")
        
        response = await _make_request_to_h2o_agent("/get-logs", request.dict())
        
        return GenericResponse(
            success=response.get("success", False),
            message=response.get("message", ""),
            data=response.get("data"),
            error=response.get("error")
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get logs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get logs: {str(e)}")

@router.post("/get-training-full-response", response_model=GenericResponse)
async def get_training_full_response(request: SessionRequest) -> GenericResponse:
    """Get full training response from session"""
    try:
        logger.info(f"Getting full training response for session: {request.session_id}")
        
        response = await _make_request_to_h2o_agent("/get-training-full-response", request.dict())
        
        return GenericResponse(
            success=response.get("success", False),
            message=response.get("message", ""),
            data=response.get("data"),
            error=response.get("error")
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get full training response: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get full training response: {str(e)}")

@router.post("/delete-session", response_model=GenericResponse)
async def delete_session(request: SessionRequest) -> GenericResponse:
    """Delete training session"""
    try:
        logger.info(f"Deleting session: {request.session_id}")
        
        response = await _make_request_to_h2o_agent("/delete-session", request.dict())
        
        return GenericResponse(
            success=response.get("success", False),
            message=response.get("message", ""),
            data=response.get("data"),
            error=response.get("error")
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check for training API"""
    try:
        # Try to ping the H2O ML agent
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{H2O_ML_AGENT_URL}/health")
            response.raise_for_status()
            h2o_health = response.json()
        
        return {
            "status": "healthy",
            "h2o_ml_agent": h2o_health,
            "message": "Training API is healthy and H2O ML agent is accessible"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Training API health check failed: {str(e)}")
