"""Feature Engineering agent endpoints - replaces the uAgent on port 8007."""

import time
from typing import Optional, Dict, Any, List
from fastapi import APIRouter
from pydantic import BaseModel
from loguru import logger

from app.agents.feature_engineering_agent import FeatureEngineeringAgent
from app.api.agent_routes.common import (
    get_llm, decode_csv_content, dataframe_to_json_safe, make_json_serializable,
    session_service, SessionRequest, SessionResponse, DataResponse, CodeResponse, GenericResponse,
)

router = APIRouter(prefix="/engineering", tags=["feature-engineering"])


class EngineerFeaturesRequest(BaseModel):
    data: Optional[Dict[str, List[Any]]] = None
    target_variable: str
    user_instructions: str = "Engineer features for machine learning"
    max_retries: int = 3


class EngineerFeaturesCsvRequest(BaseModel):
    filename: Optional[str] = None
    file_content: str  # base64
    target_variable: str
    user_instructions: str = "Engineer features for machine learning"
    max_retries: int = 3


class EngineerFeaturesFromSessionRequest(BaseModel):
    session_id: str
    target_variable: str
    user_instructions: str = "Engineer features for machine learning"
    max_retries: int = 3


def _create_agent():
    llm = get_llm()
    return FeatureEngineeringAgent(
        model=llm,
        log=True,
        log_path="./temp",
        overwrite=True,
        human_in_the_loop=False,
        bypass_recommended_steps=False,
        bypass_explain_code=False,
        n_samples=30,
    )


@router.post("/engineer-features", response_model=SessionResponse)
async def engineer_features(request: EngineerFeaturesRequest):
    """Engineer features from dict data."""
    start = time.time()
    try:
        import pandas as pd
        df = pd.DataFrame(request.data)
        agent = _create_agent()
        agent.invoke_agent(data_raw=df, user_instructions=request.user_instructions,
                          target_variable=request.target_variable, max_retries=request.max_retries)

        execution_time = time.time() - start
        metadata = {
            "operation": "engineer_features",
            "target_variable": request.target_variable,
            "original_shape": list(df.shape),
            "execution_time": execution_time,
        }
        session_id = await session_service.create_session(agent, "engineering", metadata)

        return SessionResponse(success=True, message="Features engineered", session_id=session_id, execution_time_seconds=execution_time)
    except Exception as e:
        logger.error(f"engineer_features failed: {e}")
        return SessionResponse(success=False, message=str(e), session_id="", error=str(e))


@router.post("/engineer-features-csv", response_model=SessionResponse)
async def engineer_features_csv(request: EngineerFeaturesCsvRequest):
    """Engineer features from base64-encoded CSV."""
    start = time.time()
    try:
        df = decode_csv_content(request.file_content, request.filename or "upload.csv")
        agent = _create_agent()
        agent.invoke_agent(data_raw=df, user_instructions=request.user_instructions,
                          target_variable=request.target_variable, max_retries=request.max_retries)

        execution_time = time.time() - start
        metadata = {
            "operation": "engineer_features_csv",
            "filename": request.filename,
            "target_variable": request.target_variable,
            "original_shape": list(df.shape),
            "execution_time": execution_time,
        }
        session_id = await session_service.create_session(agent, "engineering", metadata)

        return SessionResponse(success=True, message="Features engineered from CSV", session_id=session_id, execution_time_seconds=execution_time)
    except Exception as e:
        logger.error(f"engineer_features_csv failed: {e}")
        return SessionResponse(success=False, message=str(e), session_id="", error=str(e))


@router.post("/engineer-features-from-session", response_model=SessionResponse)
async def engineer_features_from_session(request: EngineerFeaturesFromSessionRequest):
    """Engineer features from a previous session."""
    start = time.time()
    try:
        source_session = await session_service.get_session(request.session_id)
        if not source_session or "agent" not in source_session:
            return SessionResponse(success=False, message="Source session not found", session_id="", error="Source session not found")

        source_agent = source_session["agent"]
        df = None
        if hasattr(source_agent, "get_data_cleaned"):
            df = source_agent.get_data_cleaned()
        if df is None and hasattr(source_agent, "get_data_raw"):
            df = source_agent.get_data_raw()
        if df is None and hasattr(source_agent, "get_artifacts"):
            df = source_agent.get_artifacts(as_dataframe=True)
        if df is None:
            return SessionResponse(success=False, message="No data in source session", session_id="", error="No data found")

        agent = _create_agent()
        agent.invoke_agent(data_raw=df, user_instructions=request.user_instructions,
                          target_variable=request.target_variable, max_retries=request.max_retries)

        execution_time = time.time() - start
        metadata = {
            "operation": "engineer_features_from_session",
            "source_session_id": request.session_id,
            "target_variable": request.target_variable,
            "original_shape": list(df.shape),
            "execution_time": execution_time,
        }
        session_id = await session_service.create_session(agent, "engineering", metadata)

        return SessionResponse(success=True, message="Features engineered from session", session_id=session_id, execution_time_seconds=execution_time)
    except Exception as e:
        logger.error(f"engineer_features_from_session failed: {e}")
        return SessionResponse(success=False, message=str(e), session_id="", error=str(e))


@router.post("/get-session-data", response_model=DataResponse)
async def get_session_data(request: SessionRequest):
    """Get engineered data from session."""
    try:
        session = await session_service.get_session(request.session_id)
        if not session or "agent" not in session:
            return DataResponse(success=False, message="Session not found", error="Session not found")

        agent = session["agent"]
        engineered = agent.get_data_engineered() if hasattr(agent, "get_data_engineered") else None
        raw = agent.get_data_raw() if hasattr(agent, "get_data_raw") else None

        data = dataframe_to_json_safe(engineered) if engineered is not None else None
        return DataResponse(
            success=True, message="Engineered data retrieved", data=data,
            original_shape=list(raw.shape) if raw is not None and hasattr(raw, "shape") else None,
            processed_shape=list(engineered.shape) if engineered is not None and hasattr(engineered, "shape") else None,
        )
    except Exception as e:
        logger.error(f"get_session_data failed: {e}")
        return DataResponse(success=False, message=str(e), error=str(e))


@router.post("/get-engineering-function", response_model=CodeResponse)
async def get_engineering_function(request: SessionRequest):
    try:
        session = await session_service.get_session(request.session_id)
        if not session or "agent" not in session:
            return CodeResponse(success=False, message="Session not found", error="Session not found")
        agent = session["agent"]
        code = agent.get_feature_engineer_function() if hasattr(agent, "get_feature_engineer_function") else None
        return CodeResponse(success=True, message="Code retrieved", generated_code=code)
    except Exception as e:
        return CodeResponse(success=False, message=str(e), error=str(e))


@router.post("/get-engineering-steps", response_model=GenericResponse)
async def get_engineering_steps(request: SessionRequest):
    try:
        session = await session_service.get_session(request.session_id)
        if not session or "agent" not in session:
            return GenericResponse(success=False, message="Session not found", error="Session not found")
        agent = session["agent"]
        steps = agent.get_recommended_feature_engineering_steps() if hasattr(agent, "get_recommended_feature_engineering_steps") else None
        return GenericResponse(success=True, message="Steps retrieved", data=make_json_serializable(steps))
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


@router.post("/get-logs", response_model=GenericResponse)
async def get_logs(request: SessionRequest):
    try:
        session = await session_service.get_session(request.session_id)
        if not session or "agent" not in session:
            return GenericResponse(success=False, message="Session not found", error="Session not found")
        agent = session["agent"]
        logs = agent.get_logs() if hasattr(agent, "get_logs") else []
        return GenericResponse(success=True, message="Logs retrieved", data=make_json_serializable(logs))
    except Exception as e:
        return GenericResponse(success=False, message=str(e), error=str(e))


@router.post("/get-workflow-summary", response_model=GenericResponse)
async def get_workflow_summary(request: SessionRequest):
    try:
        session = await session_service.get_session(request.session_id)
        if not session:
            return GenericResponse(success=False, message="Session not found", error="Session not found")
        metadata = session.get("metadata", {})
        summary = {
            "operation": metadata.get("operation"),
            "target_variable": metadata.get("target_variable"),
            "original_shape": metadata.get("original_shape"),
            "execution_time": metadata.get("execution_time"),
        }
        return GenericResponse(success=True, message="Summary retrieved", data=summary)
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
    return {"status": "healthy", "agent": "feature_engineering"}
