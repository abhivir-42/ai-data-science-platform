"""H2O ML Training agent endpoints - replaces the uAgent on port 8008."""

import time
from typing import Optional, Dict, Any, List
from fastapi import APIRouter
from pydantic import BaseModel
from loguru import logger

from app.agents.ml_agents.h2o_ml_agent import H2OMLAgent
from app.api.agent_routes.common import (
    get_llm, decode_csv_content, dataframe_to_json_safe, make_json_serializable,
    session_service, SessionRequest, SessionResponse, DataResponse, CodeResponse, GenericResponse,
)

router = APIRouter(prefix="/training", tags=["ml-training"])


class TrainModelRequest(BaseModel):
    data: Dict[str, List[Any]]
    target_variable: str
    user_instructions: str = "Train machine learning models"
    max_runtime_secs: int = 300
    cv_folds: int = 5
    balance_classes: bool = True
    exclude_algos: List[str] = []
    max_models: int = 20
    seed: int = 42
    max_retries: int = 3


class TrainModelCsvRequest(BaseModel):
    filename: Optional[str] = None
    file_content: str  # base64
    target_variable: str
    user_instructions: str = "Train machine learning models"
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
    user_instructions: str = "Train machine learning models"
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
    leaderboard: Optional[List[Dict[str, Any]]] = None
    best_model_id: Optional[str] = None
    error: Optional[str] = None


class ModelInfoResponse(BaseModel):
    success: bool
    message: str
    model_id: Optional[str] = None
    error: Optional[str] = None


def _create_agent():
    llm = get_llm()
    return H2OMLAgent(
        model=llm,
        log=True,
        log_path="./temp",
        model_directory="./temp/models",
        overwrite=True,
        human_in_the_loop=False,
        bypass_recommended_steps=False,
        bypass_explain_code=False,
    )


def _train_and_store(agent, df, request, operation: str):
    """Common training logic."""
    start = time.time()
    agent.invoke_agent(
        data_raw=df,
        user_instructions=request.user_instructions,
        target_variable=request.target_variable,
        max_retries=request.max_retries,
    )
    execution_time = time.time() - start
    return execution_time


@router.post("/train-model", response_model=SessionResponse)
async def train_model(request: TrainModelRequest):
    """Train model with dict data."""
    try:
        import pandas as pd
        df = pd.DataFrame(request.data)
        agent = _create_agent()
        execution_time = _train_and_store(agent, df, request, "train_model")

        metadata = {
            "operation": "train_model",
            "target_variable": request.target_variable,
            "original_shape": list(df.shape),
            "execution_time": execution_time,
            "max_runtime_secs": request.max_runtime_secs,
        }
        session_id = await session_service.create_session(agent, "training", metadata)

        return SessionResponse(success=True, message="Model trained", session_id=session_id, execution_time_seconds=execution_time)
    except Exception as e:
        logger.error(f"train_model failed: {e}")
        return SessionResponse(success=False, message=str(e), session_id="", error=str(e))


@router.post("/train-model-csv", response_model=SessionResponse)
async def train_model_csv(request: TrainModelCsvRequest):
    """Train model with base64-encoded CSV."""
    try:
        df = decode_csv_content(request.file_content, request.filename or "upload.csv")
        agent = _create_agent()
        execution_time = _train_and_store(agent, df, request, "train_model_csv")

        metadata = {
            "operation": "train_model_csv",
            "filename": request.filename,
            "target_variable": request.target_variable,
            "original_shape": list(df.shape),
            "execution_time": execution_time,
        }
        session_id = await session_service.create_session(agent, "training", metadata)

        return SessionResponse(success=True, message="Model trained from CSV", session_id=session_id, execution_time_seconds=execution_time)
    except Exception as e:
        logger.error(f"train_model_csv failed: {e}")
        return SessionResponse(success=False, message=str(e), session_id="", error=str(e))


@router.post("/train-model-from-session", response_model=SessionResponse)
async def train_model_from_session(request: TrainModelFromSessionRequest):
    """Train model using data from a previous session."""
    try:
        source_session = await session_service.get_session(request.source_session_id)
        if not source_session or "agent" not in source_session:
            return SessionResponse(success=False, message="Source session not found", session_id="", error="Source session not found")

        source_agent = source_session["agent"]
        df = None
        if hasattr(source_agent, "get_data_engineered"):
            df = source_agent.get_data_engineered()
        if df is None and hasattr(source_agent, "get_data_cleaned"):
            df = source_agent.get_data_cleaned()
        if df is None and hasattr(source_agent, "get_data_raw"):
            df = source_agent.get_data_raw()
        if df is None and hasattr(source_agent, "get_artifacts"):
            df = source_agent.get_artifacts(as_dataframe=True)
        if df is None:
            return SessionResponse(success=False, message="No data in source session", session_id="", error="No data found")

        agent = _create_agent()
        execution_time = _train_and_store(agent, df, request, "train_model_from_session")

        metadata = {
            "operation": "train_model_from_session",
            "source_session_id": request.source_session_id,
            "target_variable": request.target_variable,
            "original_shape": list(df.shape),
            "execution_time": execution_time,
        }
        session_id = await session_service.create_session(agent, "training", metadata)

        return SessionResponse(success=True, message="Model trained from session", session_id=session_id, execution_time_seconds=execution_time)
    except Exception as e:
        logger.error(f"train_model_from_session failed: {e}")
        return SessionResponse(success=False, message=str(e), session_id="", error=str(e))


@router.post("/get-leaderboard", response_model=LeaderboardResponse)
async def get_leaderboard(request: SessionRequest):
    """Get model leaderboard from training session."""
    try:
        session = await session_service.get_session(request.session_id)
        if not session or "agent" not in session:
            return LeaderboardResponse(success=False, message="Session not found", error="Session not found")

        agent = session["agent"]
        leaderboard = agent.get_leaderboard() if hasattr(agent, "get_leaderboard") else None
        best_model_id = agent.get_best_model_id() if hasattr(agent, "get_best_model_id") else None

        lb_data = None
        if leaderboard is not None:
            if hasattr(leaderboard, "to_dict"):
                lb_data = leaderboard.to_dict(orient="records")
            else:
                lb_data = make_json_serializable(leaderboard)

        return LeaderboardResponse(
            success=True, message="Leaderboard retrieved",
            leaderboard=lb_data, best_model_id=str(best_model_id) if best_model_id else None,
        )
    except Exception as e:
        logger.error(f"get_leaderboard failed: {e}")
        return LeaderboardResponse(success=False, message=str(e), error=str(e))


@router.post("/get-best-model-id", response_model=ModelInfoResponse)
async def get_best_model_id(request: SessionRequest):
    try:
        session = await session_service.get_session(request.session_id)
        if not session or "agent" not in session:
            return ModelInfoResponse(success=False, message="Session not found", error="Session not found")
        agent = session["agent"]
        model_id = agent.get_best_model_id() if hasattr(agent, "get_best_model_id") else None
        return ModelInfoResponse(success=True, message="Best model ID retrieved", model_id=str(model_id) if model_id else None)
    except Exception as e:
        return ModelInfoResponse(success=False, message=str(e), error=str(e))


@router.post("/get-model-path", response_model=ModelInfoResponse)
async def get_model_path(request: SessionRequest):
    try:
        session = await session_service.get_session(request.session_id)
        if not session or "agent" not in session:
            return ModelInfoResponse(success=False, message="Session not found", error="Session not found")
        agent = session["agent"]
        path = agent.get_model_path() if hasattr(agent, "get_model_path") else None
        return ModelInfoResponse(success=True, message="Model path retrieved", model_id=str(path) if path else None)
    except Exception as e:
        return ModelInfoResponse(success=False, message=str(e), error=str(e))


@router.post("/get-training-function", response_model=CodeResponse)
async def get_training_function(request: SessionRequest):
    try:
        session = await session_service.get_session(request.session_id)
        if not session or "agent" not in session:
            return CodeResponse(success=False, message="Session not found", error="Session not found")
        agent = session["agent"]
        code = agent.get_h2o_training_function() if hasattr(agent, "get_h2o_training_function") else None
        return CodeResponse(success=True, message="Training function retrieved", generated_code=code)
    except Exception as e:
        return CodeResponse(success=False, message=str(e), error=str(e))


@router.post("/get-ml-steps", response_model=GenericResponse)
async def get_ml_steps(request: SessionRequest):
    try:
        session = await session_service.get_session(request.session_id)
        if not session or "agent" not in session:
            return GenericResponse(success=False, message="Session not found", error="Session not found")
        agent = session["agent"]
        steps = agent.get_recommended_ml_steps() if hasattr(agent, "get_recommended_ml_steps") else None
        return GenericResponse(success=True, message="ML steps retrieved", data=make_json_serializable(steps))
    except Exception as e:
        return GenericResponse(success=False, message=str(e), error=str(e))


@router.post("/get-original-data", response_model=DataResponse)
async def get_original_data(request: SessionRequest):
    try:
        session = await session_service.get_session(request.session_id)
        if not session or "agent" not in session:
            return DataResponse(success=False, message="Session not found", error="Session not found")
        agent = session["agent"]
        raw = agent.get_data_raw() if hasattr(agent, "get_data_raw") else None
        return DataResponse(success=True, message="Original data retrieved", data=dataframe_to_json_safe(raw) if raw is not None else None)
    except Exception as e:
        return DataResponse(success=False, message=str(e), error=str(e))


@router.post("/get-workflow-summary", response_model=GenericResponse)
async def get_workflow_summary(request: SessionRequest):
    try:
        session = await session_service.get_session(request.session_id)
        if not session or "agent" not in session:
            return GenericResponse(success=False, message="Session not found", error="Session not found")
        agent = session["agent"]
        summary = agent.get_workflow_summary() if hasattr(agent, "get_workflow_summary") else None
        return GenericResponse(success=True, message="Summary retrieved", data=make_json_serializable(summary))
    except Exception as e:
        return GenericResponse(success=False, message=str(e), error=str(e))


@router.post("/get-logs", response_model=GenericResponse)
async def get_logs(request: SessionRequest):
    try:
        session = await session_service.get_session(request.session_id)
        if not session or "agent" not in session:
            return GenericResponse(success=False, message="Session not found", error="Session not found")
        agent = session["agent"]
        logs = agent.get_log_summary() if hasattr(agent, "get_log_summary") else None
        return GenericResponse(success=True, message="Logs retrieved", data=make_json_serializable(logs))
    except Exception as e:
        return GenericResponse(success=False, message=str(e), error=str(e))


@router.post("/get-training-full-response", response_model=GenericResponse)
async def get_training_full_response(request: SessionRequest):
    try:
        session = await session_service.get_session(request.session_id)
        if not session or "agent" not in session:
            return GenericResponse(success=False, message="Session not found", error="Session not found")
        agent = session["agent"]
        response = agent.response if hasattr(agent, "response") else None
        return GenericResponse(success=True, message="Full response retrieved", data=make_json_serializable(response))
    except Exception as e:
        return GenericResponse(success=False, message=str(e), error=str(e))


@router.post("/delete-session", response_model=GenericResponse)
async def delete_session(request: SessionRequest):
    try:
        await session_service.delete_session(request.session_id)
        return GenericResponse(success=True, message="Session deleted")
    except Exception as e:
        return GenericResponse(success=False, message=str(e), error=str(e))


@router.get("/health")
async def health():
    return {"status": "healthy", "agent": "h2o_ml_training"}
