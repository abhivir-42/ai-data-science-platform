"""Data Visualization agent endpoints - replaces the uAgent on port 8006."""

import time
from typing import Optional, Dict, Any, List
from fastapi import APIRouter
from pydantic import BaseModel
from loguru import logger

from app.agents.data_visualisation_agent import DataVisualisationAgent
from app.api.agent_routes.common import (
    get_llm, decode_csv_content, dataframe_to_json_safe, make_json_serializable,
    session_service, SessionRequest, SessionResponse, DataResponse, CodeResponse,
    GenericResponse, ChartResponse,
)

router = APIRouter(prefix="/visualization", tags=["data-visualization"])


class CreateChartRequest(BaseModel):
    data: Optional[Dict[str, List[Any]]] = None
    user_instructions: str = "Create visualizations for the data"
    max_retries: int = 3


class CreateChartCsvRequest(BaseModel):
    filename: Optional[str] = None
    file_content: str  # base64
    user_instructions: str = "Create visualizations for the data"
    max_retries: int = 3


class CreateChartFromSessionRequest(BaseModel):
    session_id: str
    user_instructions: str = "Create comprehensive visualizations"
    max_retries: int = 3


def _create_agent():
    llm = get_llm()
    return DataVisualisationAgent(
        model=llm,
        log=True,
        log_path="./temp",
        overwrite=True,
        human_in_the_loop=False,
        bypass_recommended_steps=False,
        bypass_explain_code=False,
        n_samples=30,
    )


def _extract_chart(agent) -> Dict[str, Any]:
    """Extract plotly chart from agent response."""
    response = agent.response if hasattr(agent, "response") else None
    if not response:
        return {"plotly_chart": None, "chart_type": None}

    plotly_graph = response.get("plotly_graph") if isinstance(response, dict) else None
    chart_type = None
    if isinstance(plotly_graph, dict) and "data" in plotly_graph:
        data_list = plotly_graph.get("data", [])
        if data_list and len(data_list) > 0:
            chart_type = data_list[0].get("type", "unknown")

    return {
        "plotly_chart": make_json_serializable(plotly_graph),
        "chart_type": chart_type,
    }


@router.post("/create-chart", response_model=SessionResponse)
async def create_chart(request: CreateChartRequest):
    """Create a chart from dict data."""
    start = time.time()
    try:
        import pandas as pd
        df = pd.DataFrame(request.data)
        agent = _create_agent()
        agent.invoke_agent(data_raw=df, user_instructions=request.user_instructions, max_retries=request.max_retries)

        execution_time = time.time() - start
        metadata = {"operation": "create_chart", "user_instructions": request.user_instructions, "execution_time": execution_time}
        session_id = await session_service.create_session(agent, "visualization", metadata)

        return SessionResponse(success=True, message="Chart created", session_id=session_id, execution_time_seconds=execution_time)
    except Exception as e:
        logger.error(f"create_chart failed: {e}")
        return SessionResponse(success=False, message=str(e), session_id="", error=str(e))


@router.post("/create-chart-csv", response_model=SessionResponse)
async def create_chart_csv(request: CreateChartCsvRequest):
    """Create a chart from base64-encoded CSV."""
    start = time.time()
    try:
        df = decode_csv_content(request.file_content, request.filename or "upload.csv")
        agent = _create_agent()
        agent.invoke_agent(data_raw=df, user_instructions=request.user_instructions, max_retries=request.max_retries)

        execution_time = time.time() - start
        metadata = {"operation": "create_chart_csv", "filename": request.filename, "execution_time": execution_time}
        session_id = await session_service.create_session(agent, "visualization", metadata)

        return SessionResponse(success=True, message="Chart created from CSV", session_id=session_id, execution_time_seconds=execution_time)
    except Exception as e:
        logger.error(f"create_chart_csv failed: {e}")
        return SessionResponse(success=False, message=str(e), session_id="", error=str(e))


@router.post("/create-chart-from-session", response_model=SessionResponse)
async def create_chart_from_session(request: CreateChartFromSessionRequest):
    """Create a chart using data from a previous session."""
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
        agent.invoke_agent(data_raw=df, user_instructions=request.user_instructions, max_retries=request.max_retries)

        execution_time = time.time() - start
        metadata = {"operation": "create_chart_from_session", "source_session_id": request.session_id, "execution_time": execution_time}
        session_id = await session_service.create_session(agent, "visualization", metadata)

        return SessionResponse(success=True, message="Chart created from session data", session_id=session_id, execution_time_seconds=execution_time)
    except Exception as e:
        logger.error(f"create_chart_from_session failed: {e}")
        return SessionResponse(success=False, message=str(e), session_id="", error=str(e))


@router.post("/create-chart-direct", response_model=ChartResponse)
async def create_chart_direct(request: CreateChartCsvRequest):
    """Create a chart and return it directly (no session)."""
    try:
        df = decode_csv_content(request.file_content, request.filename or "upload.csv")
        agent = _create_agent()
        agent.invoke_agent(data_raw=df, user_instructions=request.user_instructions, max_retries=request.max_retries)

        chart_info = _extract_chart(agent)
        return ChartResponse(
            success=True,
            message="Chart created",
            plotly_chart=chart_info["plotly_chart"],
            chart_type=chart_info["chart_type"],
        )
    except Exception as e:
        logger.error(f"create_chart_direct failed: {e}")
        return ChartResponse(success=False, message=str(e), error=str(e))


@router.post("/get-plotly-graph", response_model=ChartResponse)
async def get_plotly_graph(request: SessionRequest):
    """Get the plotly chart from a visualization session."""
    try:
        session = await session_service.get_session(request.session_id)
        if not session or "agent" not in session:
            return ChartResponse(success=False, message="Session not found", error="Session not found")

        chart_info = _extract_chart(session["agent"])
        return ChartResponse(success=True, message="Chart retrieved", plotly_chart=chart_info["plotly_chart"], chart_type=chart_info["chart_type"])
    except Exception as e:
        logger.error(f"get_plotly_graph failed: {e}")
        return ChartResponse(success=False, message=str(e), error=str(e))


@router.post("/get-visualization-function", response_model=CodeResponse)
async def get_visualization_function(request: SessionRequest):
    """Get generated visualization code."""
    try:
        session = await session_service.get_session(request.session_id)
        if not session or "agent" not in session:
            return CodeResponse(success=False, message="Session not found", error="Session not found")

        agent = session["agent"]
        code = agent.get_data_visualization_function() if hasattr(agent, "get_data_visualization_function") else None
        return CodeResponse(success=True, message="Code retrieved", generated_code=code)
    except Exception as e:
        logger.error(f"get_visualization_function failed: {e}")
        return CodeResponse(success=False, message=str(e), error=str(e))


@router.post("/get-visualization-steps", response_model=GenericResponse)
async def get_visualization_steps(request: SessionRequest):
    """Get recommended visualization steps."""
    try:
        session = await session_service.get_session(request.session_id)
        if not session or "agent" not in session:
            return GenericResponse(success=False, message="Session not found", error="Session not found")

        agent = session["agent"]
        steps = agent.get_recommended_visualization_steps() if hasattr(agent, "get_recommended_visualization_steps") else None
        return GenericResponse(success=True, message="Steps retrieved", data=make_json_serializable(steps))
    except Exception as e:
        logger.error(f"get_visualization_steps failed: {e}")
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


@router.post("/delete-session", response_model=GenericResponse)
async def delete_session(request: SessionRequest):
    try:
        await session_service.delete_session(request.session_id)
        return GenericResponse(success=True, message="Session deleted")
    except Exception as e:
        return GenericResponse(success=False, message=str(e), error=str(e))


@router.get("/health")
async def health():
    return {"status": "healthy", "agent": "data_visualization"}
