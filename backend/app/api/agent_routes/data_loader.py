"""Data Loader agent endpoints - replaces the uAgent on port 8005."""

import time
import os
from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel
from loguru import logger

from app.agents.data_loader_tools_agent import DataLoaderToolsAgent
from app.api.agent_routes.common import (
    get_llm, decode_csv_content, dataframe_to_json_safe, make_json_serializable,
    session_service, SessionRequest, SessionResponse, DataResponse, GenericResponse,
)

router = APIRouter(prefix="/loading", tags=["data-loader"])


class LoadFileRequest(BaseModel):
    file_path: Optional[str] = None
    filename: Optional[str] = None
    file_content: Optional[str] = None  # base64
    user_instructions: str = "Load and analyze the uploaded file"


class LoadDirectoryRequest(BaseModel):
    directory_path: str
    user_instructions: str = "Load all files in the directory"


def _create_agent():
    llm = get_llm()
    return DataLoaderToolsAgent(
        model=llm,
        create_react_agent_kwargs={},
        invoke_react_agent_kwargs={},
        checkpointer=None,
    )


@router.post("/load-file", response_model=SessionResponse)
async def load_file(request: LoadFileRequest):
    """Load data from a file (base64 CSV or file path)."""
    start = time.time()
    try:
        agent = _create_agent()

        if request.file_content:
            df = decode_csv_content(request.file_content, request.filename or "upload.csv")
            agent.invoke_agent(
                user_instructions=request.user_instructions,
                data_raw=df,
            )
        elif request.file_path:
            agent.invoke_agent(
                user_instructions=f"{request.user_instructions}. Load file: {request.file_path}",
            )
        else:
            return SessionResponse(success=False, message="", session_id="",
                                   error="Provide file_content (base64) or file_path")

        execution_time = time.time() - start
        metadata = {
            "operation": "load_file",
            "filename": request.filename or request.file_path,
            "execution_time": execution_time,
        }
        session_id = await session_service.create_session(agent, "loading", metadata)

        return SessionResponse(
            success=True,
            message=f"Data loaded successfully from {request.filename or request.file_path}",
            session_id=session_id,
            execution_time_seconds=execution_time,
        )
    except Exception as e:
        logger.error(f"load_file failed: {e}")
        return SessionResponse(success=False, message=str(e), session_id="", error=str(e))


@router.post("/load-directory", response_model=SessionResponse)
async def load_directory(request: LoadDirectoryRequest):
    """Load all files from a directory."""
    start = time.time()
    try:
        agent = _create_agent()
        agent.invoke_agent(user_instructions=f"{request.user_instructions}. Directory: {request.directory_path}")

        execution_time = time.time() - start
        metadata = {"operation": "load_directory", "directory_path": request.directory_path, "execution_time": execution_time}
        session_id = await session_service.create_session(agent, "loading", metadata)

        return SessionResponse(success=True, message="Directory loaded", session_id=session_id, execution_time_seconds=execution_time)
    except Exception as e:
        logger.error(f"load_directory failed: {e}")
        return SessionResponse(success=False, message=str(e), session_id="", error=str(e))


@router.post("/get-artifacts", response_model=DataResponse)
async def get_artifacts(request: SessionRequest):
    """Get loaded data artifacts from session."""
    try:
        session = await session_service.get_session(request.session_id)
        if not session or "agent" not in session:
            return DataResponse(success=False, message="Session not found", error="Session not found")

        agent = session["agent"]
        artifacts = agent.get_artifacts(as_dataframe=True) if hasattr(agent, "get_artifacts") else None

        if artifacts is not None:
            data = dataframe_to_json_safe(artifacts) if hasattr(artifacts, "to_dict") else make_json_serializable(artifacts)
        else:
            data = None

        return DataResponse(success=True, message="Artifacts retrieved", data=data)
    except Exception as e:
        logger.error(f"get_artifacts failed: {e}")
        return DataResponse(success=False, message=str(e), error=str(e))


@router.post("/get-ai-message", response_model=GenericResponse)
async def get_ai_message(request: SessionRequest):
    """Get the AI message from a loader session."""
    try:
        session = await session_service.get_session(request.session_id)
        if not session or "agent" not in session:
            return GenericResponse(success=False, message="Session not found", error="Session not found")

        agent = session["agent"]
        msg = agent.get_ai_message() if hasattr(agent, "get_ai_message") else None
        data = make_json_serializable(msg)
        return GenericResponse(success=True, message="AI message retrieved", data=data)
    except Exception as e:
        logger.error(f"get_ai_message failed: {e}")
        return GenericResponse(success=False, message=str(e), error=str(e))


@router.post("/get-tool-calls", response_model=GenericResponse)
async def get_tool_calls(request: SessionRequest):
    """Get tool calls from a loader session."""
    try:
        session = await session_service.get_session(request.session_id)
        if not session or "agent" not in session:
            return GenericResponse(success=False, message="Session not found", error="Session not found")

        agent = session["agent"]
        calls = agent.get_tool_calls() if hasattr(agent, "get_tool_calls") else []
        return GenericResponse(success=True, message="Tool calls retrieved", data=make_json_serializable(calls))
    except Exception as e:
        logger.error(f"get_tool_calls failed: {e}")
        return GenericResponse(success=False, message=str(e), error=str(e))


@router.post("/get-internal-messages", response_model=GenericResponse)
async def get_internal_messages(request: SessionRequest):
    """Get internal messages from a loader session."""
    try:
        session = await session_service.get_session(request.session_id)
        if not session or "agent" not in session:
            return GenericResponse(success=False, message="Session not found", error="Session not found")

        agent = session["agent"]
        msgs = agent.get_internal_messages() if hasattr(agent, "get_internal_messages") else []
        return GenericResponse(success=True, message="Internal messages retrieved", data=make_json_serializable(msgs))
    except Exception as e:
        logger.error(f"get_internal_messages failed: {e}")
        return GenericResponse(success=False, message=str(e), error=str(e))


@router.post("/get-full-response", response_model=GenericResponse)
async def get_full_response(request: SessionRequest):
    """Get full agent response from a loader session."""
    try:
        session = await session_service.get_session(request.session_id)
        if not session or "agent" not in session:
            return GenericResponse(success=False, message="Session not found", error="Session not found")

        agent = session["agent"]
        response = agent.response if hasattr(agent, "response") else None
        return GenericResponse(success=True, message="Full response retrieved", data=make_json_serializable(response))
    except Exception as e:
        logger.error(f"get_full_response failed: {e}")
        return GenericResponse(success=False, message=str(e), error=str(e))


@router.post("/delete-session", response_model=GenericResponse)
async def delete_session(request: SessionRequest):
    """Delete a loader session."""
    try:
        await session_service.delete_session(request.session_id)
        return GenericResponse(success=True, message="Session deleted")
    except Exception as e:
        return GenericResponse(success=False, message=str(e), error=str(e))


@router.get("/health")
async def health():
    return {"status": "healthy", "agent": "data_loader"}


@router.get("/supported-formats")
async def supported_formats():
    return {
        "formats": {
            "csv": {"extensions": [".csv"], "description": "Comma-separated values"},
            "excel": {"extensions": [".xlsx", ".xls"], "description": "Excel spreadsheet"},
            "json": {"extensions": [".json"], "description": "JSON data"},
            "parquet": {"extensions": [".parquet"], "description": "Apache Parquet"},
            "pdf": {"extensions": [".pdf"], "description": "PDF documents"},
        }
    }
