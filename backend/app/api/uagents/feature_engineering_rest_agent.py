#!/usr/bin/env python3
"""
Feature Engineering uAgent REST API.

This uAgent provides comprehensive REST endpoints for the FeatureEngineeringAgent,
exposing ALL agent capabilities including:
- Feature engineering operations
- Generated feature engineering code access
- Feature engineering recommendations
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
from app.agents import FeatureEngineeringAgent

# ============================================================================
# Session Management (In-Memory Store)
# ============================================================================

class SessionStore:
    def __init__(self):
        self._sessions = {}
        self._session_timeout_hours = 24
        print(f"[SessionStore] Initialized session store")
    
    def create_session(self, agent_instance, metadata=None):
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = {
            "agent": agent_instance,
            "created_at": time.time(),
            "metadata": metadata or {}
        }
        print(f"[SessionStore] Created session {session_id}. Total sessions: {len(self._sessions)}")
        return session_id
    
    def get_session(self, session_id):
        session = self._sessions.get(session_id)
        print(f"[SessionStore] Get session {session_id}: {'Found' if session else 'Not found'}. Total sessions: {len(self._sessions)}")
        if not session:
            print(f"[SessionStore] Available sessions: {list(self._sessions.keys())}")
        return session
    
    def delete_session(self, session_id):
        if session_id in self._sessions:
            del self._sessions[session_id]
            print(f"[SessionStore] Deleted session {session_id}. Total sessions: {len(self._sessions)}")
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

def _create_feature_engineering_agent():
    """Create FeatureEngineeringAgent instance"""
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_key, temperature=0.1)
    return FeatureEngineeringAgent(model=llm, log=True, n_samples=30)

# ============================================================================
# uAgent Setup
# ============================================================================

agent = Agent(
    name="feature_engineering_rest_uagent",
    port=8007,
    seed="feature_engineering_rest_uagent_secret_seed",
    endpoint=["http://127.0.0.1:8007/submit"],
)

fund_agent_if_low(agent.wallet.address())

# ============================================================================
# Pydantic Models
# ============================================================================

class HealthResponse(Model):
    status: str
    agent: str

class EngineerFeaturesRequest(Model):
    data: Dict[str, List[Any]]
    target_variable: Optional[str] = None
    user_instructions: Optional[str] = None
    max_retries: int = 3

class EngineerFeaturesCsvRequest(Model):
    filename: Optional[str] = None
    file_content: str  # base64-encoded CSV
    target_variable: Optional[str] = None
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

class SessionRequest(Model):
    session_id: str

class DeleteSessionRequest(Model):
    session_id: str

# ============================================================================
# Main Processing Endpoints
# ============================================================================

@agent.on_rest_post("/engineer-features", EngineerFeaturesRequest, SessionResponse)
async def engineer_features(ctx: Context, req: EngineerFeaturesRequest) -> SessionResponse:
    """Engineer features for dataset provided as dictionary data and create session"""
    try:
        start_time = time.time()
        
        # Create agent instance
        fe_agent = _create_feature_engineering_agent()
        
        # Convert request data to DataFrame
        df = pd.DataFrame.from_dict(req.data)
        
        if df.empty:
            return SessionResponse(
                success=False, 
                message="Empty dataset provided", 
                session_id="",
                error="Dataset contains no data"
            )
        
        # Execute feature engineering
        fe_agent.invoke_agent(
            data_raw=df,
            user_instructions=req.user_instructions,
            target_variable=req.target_variable,
            max_retries=req.max_retries
        )
        
        execution_time = time.time() - start_time
        
        # Create session
        session_id = session_store.create_session(
            fe_agent,
            metadata={
                "operation": "engineer_features",
                "target_variable": req.target_variable,
                "user_instructions": req.user_instructions,
                "original_shape": list(df.shape),
                "execution_time": execution_time
            }
        )
        
        return SessionResponse(
            success=True,
            message="Feature engineering completed successfully",
            session_id=session_id,
            execution_time_seconds=execution_time
        )
        
    except Exception as e:
        return SessionResponse(
            success=False,
            message="Feature engineering failed",
            session_id="",
            error=str(e)
        )

@agent.on_rest_post("/engineer-features-csv", EngineerFeaturesCsvRequest, SessionResponse)
async def engineer_features_csv(ctx: Context, req: EngineerFeaturesCsvRequest) -> SessionResponse:
    """Engineer features for dataset provided as base64-encoded CSV file and create session"""
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
        fe_agent = _create_feature_engineering_agent()
        
        # Execute feature engineering
        fe_agent.invoke_agent(
            data_raw=df,
            user_instructions=req.user_instructions,
            target_variable=req.target_variable,
            max_retries=req.max_retries
        )
        
        execution_time = time.time() - start_time
        
        # Create session
        print(f"[DEBUG] Creating session for feature engineering...")
        session_id = session_store.create_session(
            fe_agent,
            metadata={
                "operation": "engineer_features_csv",
                "filename": req.filename,
                "target_variable": req.target_variable,
                "user_instructions": req.user_instructions,
                "original_shape": list(df.shape),
                "execution_time": execution_time
            }
        )
        print(f"[DEBUG] Session created: {session_id}")
        
        # Debug: Test immediate retrieval
        test_session = session_store.get_session(session_id)
        print(f"[DEBUG] Immediate session test: {'Found' if test_session else 'Not found'}")
        if test_session:
            print(f"[DEBUG] Session agent type: {type(test_session['agent'])}")
            test_data = test_session['agent'].get_data_engineered()
            print(f"[DEBUG] Engineered data available: {test_data is not None}")
        
        return SessionResponse(
            success=True,
            message="CSV feature engineering completed successfully",
            session_id=session_id,
            execution_time_seconds=execution_time
        )
        
    except Exception as e:
        return SessionResponse(
            success=False,
            message="CSV feature engineering failed",
            session_id="",
            error=str(e)
        )

# ============================================================================
# Session-Based Result Access Endpoints
# ============================================================================

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
# POST Session Access Endpoints (Working) - For Frontend Compatibility
# ============================================================================

@agent.on_rest_post("/get-session-data", SessionRequest, DataResponse)
async def get_session_data_post(ctx: Context, req: SessionRequest) -> DataResponse:
    """Get engineered dataset from session (POST version for frontend compatibility)"""
    try:
        print(f"[DEBUG] POST: Getting engineered data for session {req.session_id}")
        session = session_store.get_session(req.session_id)
        if not session:
            return DataResponse(
                success=False,
                message="Session not found",
                error=f"Session {req.session_id} not found or expired"
            )
        
        fe_agent = session["agent"]
        engineered_df = fe_agent.get_data_engineered()
        
        if engineered_df is None:
            return DataResponse(
                success=False,
                message="No engineered data available",
                error="Feature engineering may have failed or not completed"
            )
        
        # Get original data for comparison
        original_df = fe_agent.get_data_raw()
        original_shape = list(original_df.shape) if original_df is not None else None
        
        return DataResponse(
            success=True,
            message="Engineered data retrieved successfully",
            data=dataframe_to_json_safe(engineered_df),
            original_shape=original_shape,
            processed_shape=list(engineered_df.shape)
        )
        
    except Exception as e:
        return DataResponse(
            success=False,
            message="Failed to retrieve engineered data",
            error=str(e)
        )

@agent.on_rest_post("/get-engineering-function", SessionRequest, CodeResponse)
async def get_engineering_function_post(ctx: Context, req: SessionRequest) -> CodeResponse:
    """Get engineering function from session (POST version)"""
    try:
        print(f"[DEBUG] POST: Getting engineering function for session {req.session_id}")
        session = session_store.get_session(req.session_id)
        if not session:
            return CodeResponse(
                success=False,
                message="Session not found",
                error=f"Session {req.session_id} not found or expired"
            )
        
        fe_agent = session["agent"]
        
        # Use the correct method to get the function
        engineering_function = fe_agent.get_feature_engineer_function()
        
        if not engineering_function:
            return CodeResponse(
                success=False,
                message="No engineering function available",
                error="Feature engineering function was not generated or is empty"
            )
        
        return CodeResponse(
            success=True,
            message="Engineering function retrieved successfully",
            generated_code=engineering_function,
            code_explanation="This function was automatically generated to engineer features for your dataset based on the provided instructions, target variable, and data characteristics."
        )
        
    except Exception as e:
        return CodeResponse(
            success=False,
            message="Failed to retrieve engineering function",
            error=str(e)
        )

@agent.on_rest_post("/get-engineering-steps", SessionRequest, GenericResponse)
async def get_engineering_steps_post(ctx: Context, req: SessionRequest) -> GenericResponse:
    """Get engineering recommendations from session (POST version)"""
    try:
        print(f"[DEBUG] POST: Getting engineering steps for session {req.session_id}")
        session = session_store.get_session(req.session_id)
        if not session:
            return GenericResponse(
                success=False,
                message="Session not found",
                error=f"Session {req.session_id} not found or expired"
            )
        
        fe_agent = session["agent"]
        
        # Use the correct method to get the steps
        engineering_steps = fe_agent.get_recommended_feature_engineering_steps()
        
        if not engineering_steps:
            return GenericResponse(
                success=False,
                message="No engineering steps available",
                error="No recommendations found in session"
            )
        
        return GenericResponse(
            success=True,
            message="Engineering steps retrieved successfully",
            data=engineering_steps
        )
        
    except Exception as e:
        return GenericResponse(
            success=False,
            message="Failed to retrieve engineering steps",
            error=str(e)
        )

@agent.on_rest_post("/get-logs", SessionRequest, GenericResponse)
async def get_logs_post(ctx: Context, req: SessionRequest) -> GenericResponse:
    """Get execution logs from session (POST version)"""
    try:
        print(f"[DEBUG] POST: Getting logs for session {req.session_id}")
        session = session_store.get_session(req.session_id)
        if not session:
            return GenericResponse(
                success=False,
                message="Session not found",
                error=f"Session {req.session_id} not found or expired"
            )
        
        fe_agent = session["agent"]
        
        # Get logs from the agent
        logs = fe_agent.get_logs() if hasattr(fe_agent, 'get_logs') else []
        
        return GenericResponse(
            success=True,
            message="Logs retrieved successfully",
            data=logs if logs else "No logs available"
        )
        
    except Exception as e:
        return GenericResponse(
            success=False,
            message="Failed to retrieve logs",
            error=str(e)
        )

@agent.on_rest_post("/get-workflow-summary", SessionRequest, GenericResponse)
async def get_workflow_summary_post(ctx: Context, req: SessionRequest) -> GenericResponse:
    """Get workflow summary from session (POST version)"""
    try:
        print(f"[DEBUG] POST: Getting workflow summary for session {req.session_id}")
        session = session_store.get_session(req.session_id)
        if not session:
            return GenericResponse(
                success=False,
                message="Session not found",
                error=f"Session {req.session_id} not found or expired"
            )
        
        fe_agent = session["agent"]
        
        # Create a summary from the session metadata
        session_data = session_store._sessions.get(req.session_id, {})
        metadata = session_data.get("metadata", {})
        
        summary = {
            "operation": metadata.get("operation", "feature_engineering"),
            "filename": metadata.get("filename", "Unknown"),
            "target_variable": metadata.get("target_variable", "Unknown"),
            "user_instructions": metadata.get("user_instructions", "None provided"),
            "original_shape": metadata.get("original_shape", "Unknown"),
            "execution_time": metadata.get("execution_time", "Unknown")
        }
        
        return GenericResponse(
            success=True,
            message="Workflow summary retrieved successfully",
            data=summary
        )
        
    except Exception as e:
        return GenericResponse(
            success=False,
            message="Failed to retrieve workflow summary",
            error=str(e)
        )

@agent.on_rest_post("/get-original-data", SessionRequest, DataResponse)
async def get_original_data_post(ctx: Context, req: SessionRequest) -> DataResponse:
    """Get original dataset from session (POST version)"""
    try:
        print(f"[DEBUG] POST: Getting original data for session {req.session_id}")
        session = session_store.get_session(req.session_id)
        if not session:
            return DataResponse(
                success=False,
                message="Session not found",
                error=f"Session {req.session_id} not found or expired"
            )
        
        fe_agent = session["agent"]
        original_df = fe_agent.get_data_raw()
        
        if original_df is None:
            return DataResponse(
                success=False,
                message="No original data available",
                error="Original data was not found in session"
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

# ============================================================================
# Utility Endpoints
# ============================================================================

@agent.on_rest_get("/health", HealthResponse)
async def health_check(ctx: Context) -> HealthResponse:
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        agent="feature_engineering_rest_uagent"
    )

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("🔧 Starting Feature Engineering uAgent (REST)...")
    print("📡 Available endpoints:")
    print("   GET  http://127.0.0.1:8007/health")
    print("   POST http://127.0.0.1:8007/engineer-features")
    print("   POST http://127.0.0.1:8007/engineer-features-csv")
    print("   POST http://127.0.0.1:8007/session/{id}/delete")
    print("   POST http://127.0.0.1:8007/get-session-data")
    print("   POST http://127.0.0.1:8007/get-engineering-function")
    print("   POST http://127.0.0.1:8007/get-engineering-steps")
    print("   POST http://127.0.0.1:8007/get-logs")
    print("   POST http://127.0.0.1:8007/get-workflow-summary")
    print("   POST http://127.0.0.1:8007/get-original-data")
    print("🚀 Agent starting...")
    agent.run()
