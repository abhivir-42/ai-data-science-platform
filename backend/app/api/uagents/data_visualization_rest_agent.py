#!/usr/bin/env python3
"""
Data Visualization uAgent REST API.

This uAgent provides comprehensive REST endpoints for the DataVisualisationAgent,
exposing ALL agent capabilities including:
- Chart generation (Plotly visualizations)
- Generated visualization code access
- Chart recommendations
- Workflow summaries and logs
- Session-based result access

Follows the established uAgent pattern with enhanced functionality.
"""

import os
import sys
import io
import base64
import uuid
import time
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
import pandas as pd
import numpy as np

def _ensure_project_root_on_path():
    """Add project root and backend to sys.path for imports"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, "..", "..", "..", "..")
    backend_dir = os.path.join(project_root, "backend")
    
    for path in [project_root, backend_dir]:
        abs_path = os.path.abspath(path)
        if abs_path not in sys.path:
            sys.path.insert(0, abs_path)

def _load_environment():
    """Load environment variables from .env files"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for path in (
        os.path.join(current_dir, ".env"),
        os.path.join(current_dir, "..", "..", "..", "..", ".env"),
    ):
        if os.path.exists(path):
            load_dotenv(dotenv_path=path)
            break

_ensure_project_root_on_path()
_load_environment()

from uagents import Agent, Context, Model
from uagents.setup import fund_agent_if_low
from langchain_openai import ChatOpenAI
from app.agents import DataVisualisationAgent

# ============================================================================
# Session Management (In-Memory Store)
# ============================================================================

class SessionStore:
    def __init__(self):
        self._sessions = {}
        self._session_timeout_hours = 24
    
    def create_session(self, agent_instance, metadata=None):
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = {
            "agent": agent_instance,
            "created_at": time.time(),
            "metadata": metadata or {}
        }
        return session_id
    
    def get_session(self, session_id):
        return self._sessions.get(session_id)
    
    def delete_session(self, session_id):
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

# Global session store
session_store = SessionStore()

# ============================================================================
# JSON Serialization Utilities
# ============================================================================

def make_json_serializable(data):
    """Convert pandas/numpy types to JSON-serializable Python types"""
    if isinstance(data, dict):
        return {k: make_json_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [make_json_serializable(item) for item in data]
    elif pd.isna(data):
        return None
    elif hasattr(data, 'isoformat'):  # datetime/Timestamp
        return data.isoformat()
    elif isinstance(data, (np.integer, np.floating)):
        return data.item() if not np.isnan(data) else None
    elif hasattr(data, 'item') and hasattr(data, 'dtype'):
        return data.item()
    else:
        return data

def dataframe_to_json_safe(df):
    """Convert DataFrame to JSON-safe format"""
    if df is None or df.empty:
        return {"records": [], "columns": []}
    
    records = df.to_dict(orient="records")
    cleaned_records = [make_json_serializable(record) for record in records]
    
    return {
        "records": cleaned_records,
        "columns": list(map(str, df.columns.tolist())),
        "shape": [int(df.shape[0]), int(df.shape[1])]
    }

def _create_data_visualization_agent():
    """Create DataVisualisationAgent instance"""
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_key, temperature=0.1)
    agent = DataVisualisationAgent(
        model=llm, 
        log=True, 
        n_samples=30,
        bypass_explain_code=False,  # CRITICAL: Ensure chart execution happens!
        human_in_the_loop=False     # No manual approval needed
    )
    print(f"[DEBUG] Agent created with bypass_explain_code=False")
    return agent

# ============================================================================
# uAgent Setup
# ============================================================================

agent = Agent(
    name="data_visualization_rest_uagent",
    port=8006,
    seed="data_visualization_rest_uagent_secret_seed",
    endpoint=["http://127.0.0.1:8006/submit"],
)

fund_agent_if_low(agent.wallet.address())

# ============================================================================
# Pydantic Models
# ============================================================================

class HealthResponse(Model):
    status: str
    agent: str

class CreateChartRequest(Model):
    data: Dict[str, List[Any]]
    user_instructions: Optional[str] = None
    max_retries: int = 3

class CreateChartCsvRequest(Model):
    filename: Optional[str] = None
    file_content: str  # base64-encoded CSV
    user_instructions: Optional[str] = None
    max_retries: int = 3

class SessionResponse(Model):
    success: bool
    message: str
    session_id: str
    execution_time_seconds: Optional[float] = None
    error: Optional[str] = None

class DataResponse(Model):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    original_shape: Optional[List[int]] = None
    processed_shape: Optional[List[int]] = None
    error: Optional[str] = None

class ChartResponse(Model):
    success: bool
    message: str
    plotly_chart: Optional[Dict[str, Any]] = None
    chart_type: Optional[str] = None
    error: Optional[str] = None

class CodeResponse(Model):
    success: bool
    message: str
    generated_code: Optional[str] = None
    code_explanation: Optional[str] = None
    error: Optional[str] = None

class GenericResponse(Model):
    success: bool
    message: str
    data: Optional[Any] = None
    error: Optional[str] = None

# ============================================================================
# Main Processing Endpoints
# ============================================================================

@agent.on_rest_post("/create-chart", CreateChartRequest, SessionResponse)
async def create_chart(ctx: Context, req: CreateChartRequest) -> SessionResponse:
    """Create visualization from dataset provided as dictionary data and create session"""
    try:
        start_time = time.time()
        
        # Create agent instance
        viz_agent = _create_data_visualization_agent()
        
        # Convert request data to DataFrame
        df = pd.DataFrame.from_dict(req.data)
        
        if df.empty:
            return SessionResponse(
                success=False, 
                message="Empty dataset provided", 
                session_id="",
                error="Dataset contains no data"
            )
        
        # Execute data visualization
        viz_agent.invoke_agent(
            data_raw=df,
            user_instructions=req.user_instructions,
            max_retries=req.max_retries
        )
        
        execution_time = time.time() - start_time
        
        # Create session
        session_id = session_store.create_session(
            viz_agent,
            metadata={
                "operation": "create_chart",
                "user_instructions": req.user_instructions,
                "original_shape": list(df.shape),
                "execution_time": execution_time
            }
        )
        
        return SessionResponse(
            success=True,
            message="Chart creation completed successfully",
            session_id=session_id,
            execution_time_seconds=execution_time
        )
        
    except Exception as e:
        return SessionResponse(
            success=False,
            message="Chart creation failed",
            session_id="",
            error=str(e)
        )

@agent.on_rest_post("/create-chart-csv", CreateChartCsvRequest, SessionResponse)
async def create_chart_csv(ctx: Context, req: CreateChartCsvRequest) -> SessionResponse:
    """Create visualization from dataset provided as base64-encoded CSV file and create session"""
    try:
        start_time = time.time()
        
        # Decode CSV content
        try:
            decoded = base64.b64decode(req.file_content)
            csv_text = decoded.decode("utf-8", errors="replace")
            df = pd.read_csv(io.StringIO(csv_text))
        except Exception as e:
            return SessionResponse(
                success=False,
                message="Invalid CSV data",
                session_id="",
                error=f"Failed to decode CSV: {str(e)}"
            )
        
        if df.empty:
            return SessionResponse(
                success=False,
                message="Empty CSV file",
                session_id="",
                error="CSV contains no data"
            )
        
        # Create agent instance
        viz_agent = _create_data_visualization_agent()
        
        # Execute data visualization
        viz_agent.invoke_agent(
            data_raw=df,
            user_instructions=req.user_instructions,
            max_retries=req.max_retries
        )
        
        execution_time = time.time() - start_time
        
        # Create session
        session_id = session_store.create_session(
            viz_agent,
            metadata={
                "operation": "create_chart_csv",
                "filename": req.filename,
                "user_instructions": req.user_instructions,
                "original_shape": list(df.shape),
                "execution_time": execution_time
            }
        )
        
        return SessionResponse(
            success=True,
            message="CSV chart creation completed successfully",
            session_id=session_id,
            execution_time_seconds=execution_time
        )
        
    except Exception as e:
        return SessionResponse(
            success=False,
            message="CSV chart creation failed",
            session_id="",
            error=str(e)
        )

# ============================================================================
# Session-Based Result Access Endpoints
# ============================================================================

@agent.on_rest_get("/session/{session_id}/plotly-graph", ChartResponse)
async def get_plotly_graph(ctx: Context, session_id: str) -> ChartResponse:
    """Get generated Plotly chart from session"""
    try:
        session = session_store.get_session(session_id)
        if not session:
            return ChartResponse(
                success=False,
                message="Session not found",
                error=f"Session {session_id} not found or expired"
            )
        
        viz_agent = session["agent"]
        
        # Get chart data directly from the agent response
        response_data = viz_agent.get_response()
        if not response_data or 'plotly_graph' not in response_data:
            return ChartResponse(
                success=False,
                message="No chart available",
                error=f"Chart generation may have failed. Available keys: {list(response_data.keys()) if response_data else 'None'}"
            )
        
        # Get chart data directly from response (bypass problematic get_plotly_graph method)
        plotly_graph = response_data['plotly_graph']
        print(f"[DEBUG] Direct chart access: {type(plotly_graph)} - {plotly_graph is not None}")
        
        if not plotly_graph:
            return ChartResponse(
                success=False,
                message="No chart data available",
                error="Chart generation completed but no chart data found"
            )
        
        # Make plotly graph JSON serializable
        serializable_graph = make_json_serializable(plotly_graph)
        
        # Extract chart type if available
        chart_type = None
        if isinstance(plotly_graph, dict) and 'data' in plotly_graph:
            if plotly_graph['data'] and len(plotly_graph['data']) > 0:
                chart_type = plotly_graph['data'][0].get('type', 'unknown')
        
        return ChartResponse(
            success=True,
            message="Plotly chart retrieved successfully",
            plotly_chart=serializable_graph,
            chart_type=chart_type
        )
        
    except Exception as e:
        return ChartResponse(
            success=False,
            message="Failed to retrieve Plotly chart",
            error=str(e)
        )

@agent.on_rest_get("/session/{session_id}/visualization-function", CodeResponse)
async def get_visualization_function(ctx: Context, session_id: str) -> CodeResponse:
    """Get generated Python visualization function from session"""
    try:
        session = session_store.get_session(session_id)
        if not session:
            return CodeResponse(
                success=False,
                message="Session not found",
                error=f"Session {session_id} not found or expired"
            )
        
        viz_agent = session["agent"]
        viz_function = viz_agent.get_data_visualization_function()
        
        if not viz_function:
            return CodeResponse(
                success=False,
                message="No visualization function available",
                error="Visualization function was not generated or is empty"
            )
        
        return CodeResponse(
            success=True,
            message="Visualization function retrieved successfully",
            generated_code=viz_function,
            code_explanation="This function was automatically generated to create visualizations for your dataset based on the provided instructions and data characteristics."
        )
        
    except Exception as e:
        return CodeResponse(
            success=False,
            message="Failed to retrieve visualization function",
            error=str(e)
        )

@agent.on_rest_get("/session/{session_id}/visualization-steps", GenericResponse)
async def get_visualization_steps(ctx: Context, session_id: str) -> GenericResponse:
    """Get recommended visualization steps from session"""
    try:
        session = session_store.get_session(session_id)
        if not session:
            return GenericResponse(
                success=False,
                message="Session not found",
                error=f"Session {session_id} not found or expired"
            )
        
        viz_agent = session["agent"]
        viz_steps = viz_agent.get_recommended_visualization_steps()
        
        return GenericResponse(
            success=True,
            message="Visualization steps retrieved successfully",
            data=viz_steps
        )
        
    except Exception as e:
        return GenericResponse(
            success=False,
            message="Failed to retrieve visualization steps",
            error=str(e)
        )

@agent.on_rest_get("/session/{session_id}/original-data", DataResponse)
async def get_original_data(ctx: Context, session_id: str) -> DataResponse:
    """Get original dataset from session"""
    try:
        session = session_store.get_session(session_id)
        if not session:
            return DataResponse(
                success=False,
                message="Session not found",
                error=f"Session {session_id} not found or expired"
            )
        
        viz_agent = session["agent"]
        original_df = viz_agent.get_data_raw()
        
        if original_df is None:
            return DataResponse(
                success=False,
                message="No original data available",
                error="Original data not found in session"
            )
        
        return DataResponse(
            success=True,
            message="Original data retrieved successfully",
            data=dataframe_to_json_safe(original_df),
            original_shape=list(original_df.shape),
            processed_shape=list(original_df.shape)
        )
        
    except Exception as e:
        return DataResponse(
            success=False,
            message="Failed to retrieve original data",
            error=str(e)
        )

@agent.on_rest_get("/session/{session_id}/workflow-summary", GenericResponse)
async def get_workflow_summary(ctx: Context, session_id: str) -> GenericResponse:
    """Get workflow summary from session"""
    try:
        session = session_store.get_session(session_id)
        if not session:
            return GenericResponse(
                success=False,
                message="Session not found",
                error=f"Session {session_id} not found or expired"
            )
        
        viz_agent = session["agent"]
        workflow_summary = viz_agent.get_workflow_summary()
        
        return GenericResponse(
            success=True,
            message="Workflow summary retrieved successfully",
            data=workflow_summary
        )
        
    except Exception as e:
        return GenericResponse(
            success=False,
            message="Failed to retrieve workflow summary",
            error=str(e)
        )

@agent.on_rest_get("/session/{session_id}/logs", GenericResponse)
async def get_logs(ctx: Context, session_id: str) -> GenericResponse:
    """Get execution logs from session"""
    try:
        session = session_store.get_session(session_id)
        if not session:
            return GenericResponse(
                success=False,
                message="Session not found",
                error=f"Session {session_id} not found or expired"
            )
        
        viz_agent = session["agent"]
        log_summary = viz_agent.get_log_summary()
        
        return GenericResponse(
            success=True,
            message="Logs retrieved successfully",
            data=log_summary
        )
        
    except Exception as e:
        return GenericResponse(
            success=False,
            message="Failed to retrieve logs",
            error=str(e)
        )

@agent.on_rest_get("/session/{session_id}/full-response", GenericResponse)
async def get_full_response(ctx: Context, session_id: str) -> GenericResponse:
    """Get complete agent response from session"""
    try:
        session = session_store.get_session(session_id)
        if not session:
            return GenericResponse(
                success=False,
                message="Session not found",
                error=f"Session {session_id} not found or expired"
            )
        
        viz_agent = session["agent"]
        response = viz_agent.get_response()
        
        # Make response JSON serializable
        serializable_response = make_json_serializable(response)
        
        return GenericResponse(
            success=True,
            message="Full response retrieved successfully",
            data=serializable_response
        )
        
    except Exception as e:
        return GenericResponse(
            success=False,
            message="Failed to retrieve full response",
            error=str(e)
        )

# ============================================================================
# Utility Endpoints
# ============================================================================

@agent.on_rest_get("/health", HealthResponse)
async def health_check(ctx: Context) -> HealthResponse:
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        agent="data_visualization_rest_uagent"
    )

class SessionRequest(Model):
    session_id: str

class DeleteSessionRequest(Model):
    session_id: str

# ============================================================================
# DIRECT CHART GENERATION (No Session Storage) 
# ============================================================================

@agent.on_rest_post("/create-chart-direct", CreateChartCsvRequest, ChartResponse)
async def create_chart_direct(ctx: Context, req: CreateChartCsvRequest) -> ChartResponse:
    """Create visualization and return chart data immediately (no session)"""
    try:
        # Decode CSV content
        try:
            decoded = base64.b64decode(req.file_content)
            csv_text = decoded.decode("utf-8", errors="replace")
            df = pd.read_csv(io.StringIO(csv_text))
        except Exception as e:
            return ChartResponse(
                success=False,
                message="Invalid CSV data",
                error=f"Failed to decode CSV: {str(e)}"
            )
        
        if df.empty:
            return ChartResponse(
                success=False,
                message="Empty CSV file",
                error="CSV contains no data"
            )
        
        # Create agent and execute immediately
        viz_agent = _create_data_visualization_agent()
        
        try:
            print(f"[DEBUG DIRECT] Starting chart generation for {len(df)} rows")
            viz_agent.invoke_agent(
                data_raw=df,
                user_instructions=req.user_instructions,
                max_retries=req.max_retries
            )
            print(f"[DEBUG DIRECT] Chart generation completed successfully")
        except Exception as e:
            print(f"[DEBUG DIRECT] Chart generation error: {e}")
            return ChartResponse(
                success=False,
                message="Chart generation failed",
                error=f"Agent execution error: {str(e)}"
            )
        
        # Get chart data immediately from response
        response_data = viz_agent.get_response()
        print(f"[DEBUG DIRECT] Response keys: {list(response_data.keys()) if response_data else 'None'}")
        
        if not response_data:
            return ChartResponse(
                success=False,
                message="Chart generation failed",
                error="No response data from agent"
            )
        
        if 'plotly_graph' not in response_data:
            return ChartResponse(
                success=False,
                message="Chart generation failed",
                error=f"Missing plotly_graph in response. Available keys: {list(response_data.keys())}"
            )
        
        plotly_graph = response_data['plotly_graph']
        print(f"[DEBUG DIRECT] Chart data: {type(plotly_graph)} - {plotly_graph is not None}")
        
        if not plotly_graph:
            return ChartResponse(
                success=False,
                message="Empty chart data",
                error="Chart generation completed but produced no data"
            )
        
        # Make chart JSON serializable
        serializable_graph = make_json_serializable(plotly_graph)
        
        # Extract chart type
        chart_type = None
        if isinstance(plotly_graph, dict) and 'data' in plotly_graph:
            if plotly_graph['data'] and len(plotly_graph['data']) > 0:
                chart_type = plotly_graph['data'][0].get('type', 'unknown')
        
        return ChartResponse(
            success=True,
            message="Chart created successfully",
            plotly_chart=serializable_graph,
            chart_type=chart_type
        )
        
    except Exception as e:
        return ChartResponse(
            success=False,
            message="Chart creation failed",
            error=str(e)
        )

# ============================================================================
# POST Session Access Endpoints (Working)
# ============================================================================

@agent.on_rest_post("/get-plotly-graph", SessionRequest, ChartResponse)
async def get_plotly_graph_post(ctx: Context, req: SessionRequest) -> ChartResponse:
    """Get Plotly graph from session (POST version)"""
    try:
        print(f"[DEBUG] Requesting chart for session: {req.session_id}")
        session = session_store.get_session(req.session_id)
        if not session:
            print(f"[DEBUG] Session {req.session_id} not found in store")
            return ChartResponse(
                success=False,
                message="Session not found",
                error=f"Session {req.session_id} not found or expired"
            )
        
        viz_agent = session["agent"]
        
        if viz_agent.response and "plotly_graph" in viz_agent.response:
            plotly_graph = viz_agent.response["plotly_graph"]
            if plotly_graph:
                return ChartResponse(
                    success=True,
                    message="Chart retrieved successfully",
                    figure=plotly_graph
                )
        
        return ChartResponse(
            success=False,
            message="No chart available",
            error="No chart artifacts found in session"
        )
        
    except Exception as e:
        return ChartResponse(
            success=False,
            message="Failed to retrieve chart",
            error=str(e)
        )

@agent.on_rest_post("/get-visualization-function", SessionRequest, CodeResponse)
async def get_visualization_function_post(ctx: Context, req: SessionRequest) -> CodeResponse:
    """Get visualization function from session (POST version)"""
    try:
        session = session_store.get_session(req.session_id)
        if not session:
            return CodeResponse(
                success=False,
                message="Session not found",
                error=f"Session {req.session_id} not found or expired"
            )
        
        viz_agent = session["agent"]
        
        if viz_agent.response and "data_visualization_function" in viz_agent.response:
            return CodeResponse(
                success=True,
                message="Visualization function retrieved successfully",
                generated_code=viz_agent.response["data_visualization_function"]
            )
        
        return CodeResponse(
            success=False,
            message="No visualization function available",
            error="No generated code found in session"
        )
        
    except Exception as e:
        return CodeResponse(
            success=False,
            message="Failed to retrieve visualization function",
            error=str(e)
        )

@agent.on_rest_post("/get-visualization-steps", SessionRequest, GenericResponse)
async def get_visualization_steps_post(ctx: Context, req: SessionRequest) -> GenericResponse:
    """Get visualization recommendations from session (POST version)"""
    try:
        session = session_store.get_session(req.session_id)
        if not session:
            return GenericResponse(
                success=False,
                message="Session not found",
                error=f"Session {req.session_id} not found or expired"
            )
        
        viz_agent = session["agent"]
        
        if viz_agent.response and "recommended_steps" in viz_agent.response:
            return GenericResponse(
                success=True,
                message="Visualization steps retrieved successfully",
                data=viz_agent.response["recommended_steps"]
            )
        
        return GenericResponse(
            success=False,
            message="No visualization steps available",
            error="No recommendations found in session"
        )
        
    except Exception as e:
        return GenericResponse(
            success=False,
            message="Failed to retrieve visualization steps",
            error=str(e)
        )

@agent.on_rest_post("/delete-session", DeleteSessionRequest, GenericResponse)
async def delete_session(ctx: Context, req: DeleteSessionRequest) -> GenericResponse:
    """Delete a session"""
    try:
        deleted = session_store.delete_session(req.session_id)
        
        if not deleted:
            return GenericResponse(
                success=False,
                message="Session not found",
                error=f"Session {req.session_id} not found"
            )
        
        return GenericResponse(
            success=True,
            message=f"Session {req.session_id} deleted successfully"
        )
        
    except Exception as e:
        return GenericResponse(
            success=False,
            message="Failed to delete session",
            error=str(e)
        )

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("📊 Starting Data Visualization uAgent (REST)...")
    print("📡 Available endpoints:")
    print("   GET  http://127.0.0.1:8006/health")
    print("   POST http://127.0.0.1:8006/create-chart")
    print("   POST http://127.0.0.1:8006/create-chart-csv")
    print("   GET  http://127.0.0.1:8006/session/{id}/plotly-graph")
    print("   GET  http://127.0.0.1:8006/session/{id}/visualization-function")
    print("   GET  http://127.0.0.1:8006/session/{id}/visualization-steps")
    print("   GET  http://127.0.0.1:8006/session/{id}/original-data")
    print("   GET  http://127.0.0.1:8006/session/{id}/workflow-summary")
    print("   GET  http://127.0.0.1:8006/session/{id}/logs")
    print("   GET  http://127.0.0.1:8006/session/{id}/full-response")
    print("   POST http://127.0.0.1:8006/session/{id}/delete")
    print("🚀 Agent starting...")
    agent.run()
