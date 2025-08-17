#!/usr/bin/env python3
"""
Data Cleaning uAgent REST API.

This uAgent provides comprehensive REST endpoints for the DataCleaningAgent,
exposing ALL agent capabilities including:
- Data cleaning operations
- Generated Python code access
- Cleaning recommendations
- Workflow summaries and logs
- Session-based result access

Follows the established pattern from rest-endpoint-creation/data_cleaning_endpoint/
but with enhanced functionality for all agent methods.
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
from app.agents import DataCleaningAgent

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

def _create_data_cleaning_agent():
    """Create DataCleaningAgent instance"""
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_key, temperature=0.1)
    return DataCleaningAgent(model=llm, log=True, n_samples=30)

# ============================================================================
# uAgent Setup
# ============================================================================

agent = Agent(
    name="data_cleaning_rest_uagent",
    port=8004,
    seed="data_cleaning_rest_uagent_secret_seed",
    endpoint=["http://127.0.0.1:8004/submit"],
)

fund_agent_if_low(agent.wallet.address())

# ============================================================================
# Pydantic Models
# ============================================================================

class HealthResponse(Model):
    status: str
    agent: str

class CleanDataRequest(Model):
    data: Dict[str, List[Any]]
    user_instructions: Optional[str] = None
    max_retries: int = 3

class CleanCsvRequest(Model):
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

@agent.on_rest_post("/clean-data", CleanDataRequest, SessionResponse)
async def clean_data(ctx: Context, req: CleanDataRequest) -> SessionResponse:
    """Clean dataset provided as dictionary data and create session"""
    try:
        start_time = time.time()
        
        # Create agent instance
        cleaning_agent = _create_data_cleaning_agent()
        
        # Convert request data to DataFrame
        df = pd.DataFrame.from_dict(req.data)
        
        if df.empty:
            return SessionResponse(
                success=False, 
                message="Empty dataset provided", 
                session_id="",
                error="Dataset contains no data"
            )
        
        # Execute data cleaning
        cleaning_agent.invoke_agent(
            data_raw=df,
            user_instructions=req.user_instructions,
            max_retries=req.max_retries
        )
        
        execution_time = time.time() - start_time
        
        # Create session
        session_id = session_store.create_session(
            cleaning_agent,
            metadata={
                "operation": "clean_data",
                "user_instructions": req.user_instructions,
                "original_shape": list(df.shape),
                "execution_time": execution_time
            }
        )
        
        return SessionResponse(
            success=True,
            message="Data cleaning completed successfully",
            session_id=session_id,
            execution_time_seconds=execution_time
        )
        
    except Exception as e:
        return SessionResponse(
            success=False,
            message="Data cleaning failed",
            session_id="",
            error=str(e)
        )

@agent.on_rest_post("/clean-csv", CleanCsvRequest, SessionResponse)
async def clean_csv(ctx: Context, req: CleanCsvRequest) -> SessionResponse:
    """Clean dataset provided as base64-encoded CSV file and create session"""
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
        cleaning_agent = _create_data_cleaning_agent()
        
        # Execute data cleaning
        cleaning_agent.invoke_agent(
            data_raw=df,
            user_instructions=req.user_instructions,
            max_retries=req.max_retries
        )
        
        execution_time = time.time() - start_time
        
        # Create session
        session_id = session_store.create_session(
            cleaning_agent,
            metadata={
                "operation": "clean_csv",
                "filename": req.filename,
                "user_instructions": req.user_instructions,
                "original_shape": list(df.shape),
                "execution_time": execution_time
            }
        )
        
        return SessionResponse(
            success=True,
            message="CSV data cleaning completed successfully",
            session_id=session_id,
            execution_time_seconds=execution_time
        )
        
    except Exception as e:
        return SessionResponse(
            success=False,
            message="CSV cleaning failed",
            session_id="",
            error=str(e)
        )

# ============================================================================
# Session-Based Result Access Endpoints
# ============================================================================

class SessionRequest(Model):
    session_id: str

@agent.on_rest_post("/get-cleaned-data", SessionRequest, DataResponse)
async def get_cleaned_data(ctx: Context, req: SessionRequest) -> DataResponse:
    """Get cleaned dataset from session"""
    try:
        session = session_store.get_session(req.session_id)
        if not session:
            return DataResponse(
                success=False,
                message="Session not found",
                error=f"Session {req.session_id} not found or expired"
            )
        
        cleaning_agent = session["agent"]
        cleaned_df = cleaning_agent.get_data_cleaned()
        
        if cleaned_df is None:
            return DataResponse(
                success=False,
                message="No cleaned data available",
                error="Cleaning may have failed or not completed"
            )
        
        # Get original data for comparison
        original_df = cleaning_agent.get_data_raw()
        original_shape = list(original_df.shape) if original_df is not None else None
        
        return DataResponse(
            success=True,
            message="Cleaned data retrieved successfully",
            data=dataframe_to_json_safe(cleaned_df),
            original_shape=original_shape,
            processed_shape=list(cleaned_df.shape)
        )
        
    except Exception as e:
        return DataResponse(
            success=False,
            message="Failed to retrieve cleaned data",
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
        
        cleaning_agent = session["agent"]
        original_df = cleaning_agent.get_data_raw()
        
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

@agent.on_rest_get("/session/{session_id}/cleaning-function", CodeResponse)
async def get_cleaning_function(ctx: Context, session_id: str) -> CodeResponse:
    """Get generated Python cleaning function from session"""
    try:
        session = session_store.get_session(session_id)
        if not session:
            return CodeResponse(
                success=False,
                message="Session not found",
                error=f"Session {session_id} not found or expired"
            )
        
        cleaning_agent = session["agent"]
        cleaning_function = cleaning_agent.get_data_cleaner_function()
        
        if not cleaning_function:
            return CodeResponse(
                success=False,
                message="No cleaning function available",
                error="Cleaning function was not generated or is empty"
            )
        
        return CodeResponse(
            success=True,
            message="Cleaning function retrieved successfully",
            generated_code=cleaning_function,
            code_explanation="This function was automatically generated to clean your dataset based on the provided instructions and data characteristics."
        )
        
    except Exception as e:
        return CodeResponse(
            success=False,
            message="Failed to retrieve cleaning function",
            error=str(e)
        )

@agent.on_rest_get("/session/{session_id}/cleaning-steps", GenericResponse)
async def get_cleaning_steps(ctx: Context, session_id: str) -> GenericResponse:
    """Get recommended cleaning steps from session"""
    try:
        session = session_store.get_session(session_id)
        if not session:
            return GenericResponse(
                success=False,
                message="Session not found",
                error=f"Session {session_id} not found or expired"
            )
        
        cleaning_agent = session["agent"]
        cleaning_steps = cleaning_agent.get_recommended_cleaning_steps()
        
        return GenericResponse(
            success=True,
            message="Cleaning steps retrieved successfully",
            data=cleaning_steps
        )
        
    except Exception as e:
        return GenericResponse(
            success=False,
            message="Failed to retrieve cleaning steps",
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
        
        cleaning_agent = session["agent"]
        workflow_summary = cleaning_agent.get_workflow_summary()
        
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
        
        cleaning_agent = session["agent"]
        log_summary = cleaning_agent.get_log_summary()
        
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
        
        cleaning_agent = session["agent"]
        response = cleaning_agent.get_response()
        
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
        agent="data_cleaning_rest_uagent"
    )

class DeleteSessionRequest(Model):
    session_id: str

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
    print("🧹 Starting Data Cleaning uAgent (REST)...")
    print("📡 Available endpoints:")
    print("   GET  http://127.0.0.1:8004/health")
    print("   POST http://127.0.0.1:8004/clean-data")
    print("   POST http://127.0.0.1:8004/clean-csv")
    print("   POST http://127.0.0.1:8004/get-cleaned-data")
    print("   GET  http://127.0.0.1:8004/session/{id}/original-data")
    print("   GET  http://127.0.0.1:8004/session/{id}/cleaning-function")
    print("   GET  http://127.0.0.1:8004/session/{id}/cleaning-steps")
    print("   GET  http://127.0.0.1:8004/session/{id}/workflow-summary")
    print("   GET  http://127.0.0.1:8004/session/{id}/logs")
    print("   GET  http://127.0.0.1:8004/session/{id}/full-response")
    print("   POST http://127.0.0.1:8004/delete-session")
    print("🚀 Agent starting...")
    agent.run()
