"""Data Cleaning agent endpoints - replaces the uAgent on port 8004."""

import time
from typing import Optional, Dict, Any, List
from fastapi import APIRouter
from pydantic import BaseModel
from loguru import logger

from app.agents.data_cleaning_agent import DataCleaningAgent
from app.api.agent_routes.common import (
    get_llm, decode_csv_content, dataframe_to_json_safe, make_json_serializable,
    session_service, SessionRequest, SessionResponse, DataResponse, CodeResponse, GenericResponse,
)

router = APIRouter(prefix="/cleaning", tags=["data-cleaning"])


class CleanDataRequest(BaseModel):
    data: Optional[Dict[str, List[Any]]] = None
    user_instructions: str = "Clean the data using recommended steps"
    max_retries: int = 3


class CleanCsvRequest(BaseModel):
    filename: Optional[str] = None
    file_content: str  # base64
    user_instructions: str = "Clean the data using recommended steps"
    max_retries: int = 3
    advanced_options: Optional[Dict[str, Any]] = None


class CleanFromSessionRequest(BaseModel):
    session_id: str
    user_instructions: str = "Clean the data using recommended steps"
    max_retries: int = 3
    advanced_options: Optional[Dict[str, Any]] = None


def _create_agent():
    llm = get_llm()
    return DataCleaningAgent(
        model=llm,
        log=True,
        log_path="./temp",
        overwrite=True,
        human_in_the_loop=False,
        bypass_recommended_steps=False,
        bypass_explain_code=False,
        n_samples=30,
    )


@router.post("/clean-data", response_model=SessionResponse)
async def clean_data(request: CleanDataRequest):
    """Clean data provided as a dict of columns."""
    start = time.time()
    try:
        import pandas as pd
        df = pd.DataFrame(request.data)
        agent = _create_agent()
        agent.invoke_agent(data_raw=df, user_instructions=request.user_instructions, max_retries=request.max_retries)

        execution_time = time.time() - start
        metadata = {
            "operation": "clean_data",
            "user_instructions": request.user_instructions,
            "original_shape": list(df.shape),
            "execution_time": execution_time,
        }
        session_id = await session_service.create_session(agent, "cleaning", metadata)

        return SessionResponse(success=True, message="Data cleaned successfully", session_id=session_id, execution_time_seconds=execution_time)
    except Exception as e:
        logger.error(f"clean_data failed: {e}")
        return SessionResponse(success=False, message=str(e), session_id="", error=str(e))


@router.post("/clean-csv", response_model=SessionResponse)
async def clean_csv(request: CleanCsvRequest):
    """Clean data from a base64-encoded CSV."""
    start = time.time()
    try:
        df = decode_csv_content(request.file_content, request.filename or "upload.csv")
        agent = _create_agent()
        agent.invoke_agent(data_raw=df, user_instructions=request.user_instructions, max_retries=request.max_retries)

        execution_time = time.time() - start
        metadata = {
            "operation": "clean_csv",
            "filename": request.filename,
            "user_instructions": request.user_instructions,
            "original_shape": list(df.shape),
            "execution_time": execution_time,
        }
        session_id = await session_service.create_session(agent, "cleaning", metadata)

        return SessionResponse(success=True, message="CSV data cleaned successfully", session_id=session_id, execution_time_seconds=execution_time)
    except Exception as e:
        logger.error(f"clean_csv failed: {e}")
        return SessionResponse(success=False, message=str(e), session_id="", error=str(e))


@router.post("/clean-from-session", response_model=SessionResponse)
async def clean_from_session(request: CleanFromSessionRequest):
    """Clean data from a previous session (e.g., data loader session)."""
    start = time.time()
    try:
        # Retrieve data from previous session
        source_session = await session_service.get_session(request.session_id)
        if not source_session or "agent" not in source_session:
            return SessionResponse(success=False, message="Source session not found", session_id="", error="Source session not found")

        source_agent = source_session["agent"]

        # Try to get the data from the source agent
        df = None
        if hasattr(source_agent, "get_data_cleaned"):
            df = source_agent.get_data_cleaned()
        if df is None and hasattr(source_agent, "get_data_raw"):
            df = source_agent.get_data_raw()
        if df is None and hasattr(source_agent, "get_artifacts"):
            df = source_agent.get_artifacts(as_dataframe=True)

        if df is None:
            return SessionResponse(success=False, message="No data found in source session", session_id="", error="No data in source session")

        agent = _create_agent()
        agent.invoke_agent(data_raw=df, user_instructions=request.user_instructions, max_retries=request.max_retries)

        execution_time = time.time() - start
        metadata = {
            "operation": "clean_from_session",
            "source_session_id": request.session_id,
            "user_instructions": request.user_instructions,
            "original_shape": list(df.shape),
            "execution_time": execution_time,
        }
        session_id = await session_service.create_session(agent, "cleaning", metadata)

        return SessionResponse(success=True, message="Session data cleaned successfully", session_id=session_id, execution_time_seconds=execution_time)
    except Exception as e:
        logger.error(f"clean_from_session failed: {e}")
        return SessionResponse(success=False, message=str(e), session_id="", error=str(e))


@router.post("/get-cleaned-data", response_model=DataResponse)
async def get_cleaned_data(request: SessionRequest):
    """Get cleaned data from a cleaning session."""
    try:
        session = await session_service.get_session(request.session_id)
        if not session or "agent" not in session:
            return DataResponse(success=False, message="Session not found", error="Session not found")

        agent = session["agent"]
        cleaned = agent.get_data_cleaned() if hasattr(agent, "get_data_cleaned") else None
        raw = agent.get_data_raw() if hasattr(agent, "get_data_raw") else None

        data = dataframe_to_json_safe(cleaned) if cleaned is not None else None
        original_shape = list(raw.shape) if raw is not None and hasattr(raw, "shape") else None
        processed_shape = list(cleaned.shape) if cleaned is not None and hasattr(cleaned, "shape") else (
            [len(cleaned), len(cleaned[0]) if cleaned else 0] if isinstance(cleaned, list) else None
        )

        return DataResponse(success=True, message="Cleaned data retrieved", data=data, original_shape=original_shape, processed_shape=processed_shape)
    except Exception as e:
        logger.error(f"get_cleaned_data failed: {e}")
        return DataResponse(success=False, message=str(e), error=str(e))


@router.post("/get-original-data", response_model=DataResponse)
async def get_original_data(request: SessionRequest):
    """Get original data from a cleaning session."""
    try:
        session = await session_service.get_session(request.session_id)
        if not session or "agent" not in session:
            return DataResponse(success=False, message="Session not found", error="Session not found")

        agent = session["agent"]
        raw = agent.get_data_raw() if hasattr(agent, "get_data_raw") else None
        data = dataframe_to_json_safe(raw) if raw is not None else None

        return DataResponse(success=True, message="Original data retrieved", data=data)
    except Exception as e:
        logger.error(f"get_original_data failed: {e}")
        return DataResponse(success=False, message=str(e), error=str(e))


@router.post("/get-cleaning-function", response_model=CodeResponse)
async def get_cleaning_function(request: SessionRequest):
    """Get the generated cleaning function code."""
    try:
        session = await session_service.get_session(request.session_id)
        if not session or "agent" not in session:
            return CodeResponse(success=False, message="Session not found", error="Session not found")

        agent = session["agent"]
        code = agent.get_data_cleaner_function() if hasattr(agent, "get_data_cleaner_function") else None

        return CodeResponse(success=True, message="Cleaning function retrieved", generated_code=code)
    except Exception as e:
        logger.error(f"get_cleaning_function failed: {e}")
        return CodeResponse(success=False, message=str(e), error=str(e))


@router.post("/get-cleaning-steps", response_model=GenericResponse)
async def get_cleaning_steps(request: SessionRequest):
    """Get recommended cleaning steps."""
    try:
        session = await session_service.get_session(request.session_id)
        if not session or "agent" not in session:
            return GenericResponse(success=False, message="Session not found", error="Session not found")

        agent = session["agent"]
        steps = agent.get_recommended_cleaning_steps() if hasattr(agent, "get_recommended_cleaning_steps") else None

        return GenericResponse(success=True, message="Cleaning steps retrieved", data=make_json_serializable(steps))
    except Exception as e:
        logger.error(f"get_cleaning_steps failed: {e}")
        return GenericResponse(success=False, message=str(e), error=str(e))


@router.post("/get-logs", response_model=GenericResponse)
async def get_logs(request: SessionRequest):
    """Get logs from a cleaning session."""
    try:
        session = await session_service.get_session(request.session_id)
        if not session or "agent" not in session:
            return GenericResponse(success=False, message="Session not found", error="Session not found")

        agent = session["agent"]
        logs = agent.get_log_summary() if hasattr(agent, "get_log_summary") else None

        return GenericResponse(success=True, message="Logs retrieved", data=make_json_serializable(logs))
    except Exception as e:
        logger.error(f"get_logs failed: {e}")
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
    return {"status": "healthy", "agent": "data_cleaning"}
