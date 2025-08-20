#!/usr/bin/env python3
"""
H2O ML Training uAgent REST API.

This uAgent provides comprehensive REST endpoints for the H2OMLAgent,
exposing ALL agent capabilities including:
- H2O AutoML model training
- Model leaderboard access
- Generated H2O training code access
- ML recommendations and model paths
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
from app.agents.ml_agents import H2OMLAgent

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

def _create_h2o_ml_agent():
    """Create H2OMLAgent instance"""
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_key, temperature=0.1)
    return H2OMLAgent(model=llm, log=True)

# ============================================================================
# uAgent Setup
# ============================================================================

agent = Agent(
    name="h2o_ml_rest_uagent",
    port=8008,
    seed="h2o_ml_rest_uagent_secret_seed",
    endpoint=["http://127.0.0.1:8008/submit"],
)

fund_agent_if_low(agent.wallet.address())

# ============================================================================
# Pydantic Models
# ============================================================================

class HealthResponse(Model):
    status: str
    agent: str

class TrainModelRequest(Model):
    data: Dict[str, List[Any]]
    target_variable: str
    user_instructions: Optional[str] = None
    max_retries: int = 3
    max_runtime_secs: int = 300
    cv_folds: int = 5
    balance_classes: bool = True
    exclude_algos: List[str] = ["DeepLearning"]
    max_models: int = 20
    seed: int = 42

class TrainModelCsvRequest(Model):
    filename: Optional[str] = None
    file_content: str  # base64-encoded CSV
    target_variable: str
    user_instructions: Optional[str] = None
    max_retries: int = 3
    max_runtime_secs: int = 300
    cv_folds: int = 5
    balance_classes: bool = True
    exclude_algos: List[str] = ["DeepLearning"]
    max_models: int = 20
    seed: int = 42

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

class LeaderboardResponse(Model):
    success: bool
    message: str
    leaderboard: Optional[Dict[str, Any]] = None
    best_model_id: Optional[str] = None
    error: Optional[str] = None

class ModelInfoResponse(Model):
    success: bool
    message: str
    model_id: Optional[str] = None
    model_path: Optional[str] = None
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

@agent.on_rest_post("/train-model", TrainModelRequest, SessionResponse)
async def train_model(ctx: Context, req: TrainModelRequest) -> SessionResponse:
    """Train ML model with H2O AutoML for dataset provided as dictionary data and create session"""
    try:
        start_time = time.time()
        
        # Create agent instance
        ml_agent = _create_h2o_ml_agent()
        
        # Convert request data to DataFrame
        df = pd.DataFrame.from_dict(req.data)
        
        if df.empty:
            return SessionResponse(
                success=False, 
                message="Empty dataset provided", 
                session_id="",
                error="Dataset contains no data"
            )
        
        if req.target_variable not in df.columns:
            return SessionResponse(
                success=False,
                message="Target variable not found in dataset",
                session_id="",
                error=f"Target variable '{req.target_variable}' not found in columns: {list(df.columns)}"
            )
        
        # Execute H2O ML training
        ml_agent.invoke_agent(
            data_raw=df,
            user_instructions=req.user_instructions,
            target_variable=req.target_variable,
            max_retries=req.max_retries,
            max_runtime_secs=req.max_runtime_secs,
            cv_folds=req.cv_folds,
            balance_classes=req.balance_classes,
            exclude_algos=req.exclude_algos,
            max_models=req.max_models,
            seed=req.seed
        )
        
        execution_time = time.time() - start_time
        
        # Create session
        session_id = session_store.create_session(
            ml_agent,
            metadata={
                "operation": "train_model",
                "target_variable": req.target_variable,
                "user_instructions": req.user_instructions,
                "original_shape": list(df.shape),
                "execution_time": execution_time,
                "h2o_params": {
                    "max_runtime_secs": req.max_runtime_secs,
                    "cv_folds": req.cv_folds,
                    "balance_classes": req.balance_classes,
                    "exclude_algos": req.exclude_algos,
                    "max_models": req.max_models,
                    "seed": req.seed
                }
            }
        )
        
        return SessionResponse(
            success=True,
            message="H2O ML model training completed successfully",
            session_id=session_id,
            execution_time_seconds=execution_time
        )
        
    except Exception as e:
        return SessionResponse(
            success=False,
            message="H2O ML training failed",
            session_id="",
            error=str(e)
        )

@agent.on_rest_post("/train-model-csv", TrainModelCsvRequest, SessionResponse)
async def train_model_csv(ctx: Context, req: TrainModelCsvRequest) -> SessionResponse:
    """Train ML model with H2O AutoML for dataset provided as base64-encoded CSV file and create session"""
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
        
        if req.target_variable not in df.columns:
            return SessionResponse(
                success=False,
                message="Target variable not found in CSV",
                session_id="",
                error=f"Target variable '{req.target_variable}' not found in columns: {list(df.columns)}"
            )
        
        # Create agent instance
        ml_agent = _create_h2o_ml_agent()
        
        # Execute H2O ML training
        ml_agent.invoke_agent(
            data_raw=df,
            user_instructions=req.user_instructions,
            target_variable=req.target_variable,
            max_retries=req.max_retries,
            max_runtime_secs=req.max_runtime_secs,
            cv_folds=req.cv_folds,
            balance_classes=req.balance_classes,
            exclude_algos=req.exclude_algos,
            max_models=req.max_models,
            seed=req.seed
        )
        
        execution_time = time.time() - start_time
        
        # Create session
        session_id = session_store.create_session(
            ml_agent,
            metadata={
                "operation": "train_model_csv",
                "filename": req.filename,
                "target_variable": req.target_variable,
                "user_instructions": req.user_instructions,
                "original_shape": list(df.shape),
                "execution_time": execution_time,
                "h2o_params": {
                    "max_runtime_secs": req.max_runtime_secs,
                    "cv_folds": req.cv_folds,
                    "balance_classes": req.balance_classes,
                    "exclude_algos": req.exclude_algos,
                    "max_models": req.max_models,
                    "seed": req.seed
                }
            }
        )
        
        return SessionResponse(
            success=True,
            message="CSV H2O ML training completed successfully",
            session_id=session_id,
            execution_time_seconds=execution_time
        )
        
    except Exception as e:
        return SessionResponse(
            success=False,
            message="CSV H2O ML training failed",
            session_id="",
            error=str(e)
        )

# ============================================================================
# Session-Based Result Access Endpoints
# ============================================================================

@agent.on_rest_get("/session/{session_id}/leaderboard", LeaderboardResponse)
async def get_leaderboard(ctx: Context, session_id: str) -> LeaderboardResponse:
    """Get H2O AutoML leaderboard from session"""
    try:
        session = session_store.get_session(session_id)
        if not session:
            return LeaderboardResponse(
                success=False,
                message="Session not found",
                error=f"Session {session_id} not found or expired"
            )
        
        ml_agent = session["agent"]
        leaderboard = ml_agent.get_leaderboard()
        
        if leaderboard is None:
            return LeaderboardResponse(
                success=False,
                message="No leaderboard available",
                error="Model training may have failed or not completed"
            )
        
        # Get best model ID
        best_model_id = ml_agent.get_best_model_id()
        
        # Make leaderboard JSON serializable
        serializable_leaderboard = make_json_serializable(leaderboard)
        
        return LeaderboardResponse(
            success=True,
            message="Leaderboard retrieved successfully",
            leaderboard=serializable_leaderboard,
            best_model_id=best_model_id
        )
        
    except Exception as e:
        return LeaderboardResponse(
            success=False,
            message="Failed to retrieve leaderboard",
            error=str(e)
        )

@agent.on_rest_get("/session/{session_id}/best-model-id", ModelInfoResponse)
async def get_best_model_id(ctx: Context, session_id: str) -> ModelInfoResponse:
    """Get best model ID from session"""
    try:
        session = session_store.get_session(session_id)
        if not session:
            return ModelInfoResponse(
                success=False,
                message="Session not found",
                error=f"Session {session_id} not found or expired"
            )
        
        ml_agent = session["agent"]
        best_model_id = ml_agent.get_best_model_id()
        
        if not best_model_id:
            return ModelInfoResponse(
                success=False,
                message="No best model available",
                error="Model training may have failed or not completed"
            )
        
        return ModelInfoResponse(
            success=True,
            message="Best model ID retrieved successfully",
            model_id=best_model_id
        )
        
    except Exception as e:
        return ModelInfoResponse(
            success=False,
            message="Failed to retrieve best model ID",
            error=str(e)
        )

@agent.on_rest_get("/session/{session_id}/model-path", ModelInfoResponse)
async def get_model_path(ctx: Context, session_id: str) -> ModelInfoResponse:
    """Get saved model file path from session"""
    try:
        session = session_store.get_session(session_id)
        if not session:
            return ModelInfoResponse(
                success=False,
                message="Session not found",
                error=f"Session {session_id} not found or expired"
            )
        
        ml_agent = session["agent"]
        model_path = ml_agent.get_model_path()
        
        if not model_path:
            return ModelInfoResponse(
                success=False,
                message="No model path available",
                error="Model may not have been saved or training failed"
            )
        
        return ModelInfoResponse(
            success=True,
            message="Model path retrieved successfully",
            model_path=model_path
        )
        
    except Exception as e:
        return ModelInfoResponse(
            success=False,
            message="Failed to retrieve model path",
            error=str(e)
        )

@agent.on_rest_get("/session/{session_id}/training-function", CodeResponse)
async def get_training_function(ctx: Context, session_id: str) -> CodeResponse:
    """Get generated H2O training function from session"""
    try:
        session = session_store.get_session(session_id)
        if not session:
            return CodeResponse(
                success=False,
                message="Session not found",
                error=f"Session {session_id} not found or expired"
            )
        
        ml_agent = session["agent"]
        training_function = ml_agent.get_h2o_train_function()
        
        if not training_function:
            return CodeResponse(
                success=False,
                message="No training function available",
                error="H2O training function was not generated or is empty"
            )
        
        return CodeResponse(
            success=True,
            message="Training function retrieved successfully",
            generated_code=training_function,
            code_explanation="This H2O AutoML function was automatically generated to train models on your dataset based on the provided target variable and training parameters."
        )
        
    except Exception as e:
        return CodeResponse(
            success=False,
            message="Failed to retrieve training function",
            error=str(e)
        )

@agent.on_rest_get("/session/{session_id}/ml-steps", GenericResponse)
async def get_ml_steps(ctx: Context, session_id: str) -> GenericResponse:
    """Get recommended ML steps from session"""
    try:
        session = session_store.get_session(session_id)
        if not session:
            return GenericResponse(
                success=False,
                message="Session not found",
                error=f"Session {session_id} not found or expired"
            )
        
        ml_agent = session["agent"]
        ml_steps = ml_agent.get_recommended_ml_steps()
        
        return GenericResponse(
            success=True,
            message="ML steps retrieved successfully",
            data=ml_steps
        )
        
    except Exception as e:
        return GenericResponse(
            success=False,
            message="Failed to retrieve ML steps",
            error=str(e)
        )

@agent.on_rest_get("/session/{session_id}/original-data", DataResponse)
async def get_original_data(ctx: Context, session_id: str) -> DataResponse:
    """Get original training dataset from session"""
    try:
        session = session_store.get_session(session_id)
        if not session:
            return DataResponse(
                success=False,
                message="Session not found",
                error=f"Session {session_id} not found or expired"
            )
        
        ml_agent = session["agent"]
        original_df = ml_agent.get_data_raw()
        
        if original_df is None:
            return DataResponse(
                success=False,
                message="No original data available",
                error="Original training data not found in session"
            )
        
        return DataResponse(
            success=True,
            message="Original training data retrieved successfully",
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
    """Get training workflow summary from session"""
    try:
        session = session_store.get_session(session_id)
        if not session:
            return GenericResponse(
                success=False,
                message="Session not found",
                error=f"Session {session_id} not found or expired"
            )
        
        ml_agent = session["agent"]
        workflow_summary = ml_agent.get_workflow_summary()
        
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
    """Get training execution logs from session"""
    try:
        session = session_store.get_session(session_id)
        if not session:
            return GenericResponse(
                success=False,
                message="Session not found",
                error=f"Session {session_id} not found or expired"
            )
        
        ml_agent = session["agent"]
        log_summary = ml_agent.get_log_summary()
        
        return GenericResponse(
            success=True,
            message="Training logs retrieved successfully",
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
        
        ml_agent = session["agent"]
        response = ml_agent.get_response()
        
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
        agent="h2o_ml_rest_uagent"
    )

class SessionRequest(Model):
    session_id: str

class DeleteSessionRequest(Model):
    session_id: str

# ============================================================================
# POST Session Access Endpoints (Working)
# ============================================================================

@agent.on_rest_post("/get-leaderboard", SessionRequest, LeaderboardResponse)
async def get_leaderboard_post(ctx: Context, req: SessionRequest) -> LeaderboardResponse:
    """Get leaderboard from session (POST version)"""
    try:
        session = session_store.get_session(req.session_id)
        if not session:
            return LeaderboardResponse(
                success=False,
                message="Session not found",
                error=f"Session {req.session_id} not found or expired"
            )
        
        ml_agent = session["agent"]
        
        if ml_agent.response and "leaderboard" in ml_agent.response:
            return LeaderboardResponse(
                success=True,
                message="Leaderboard retrieved successfully",
                leaderboard=ml_agent.response["leaderboard"]
            )
        
        return LeaderboardResponse(
            success=False,
            message="No leaderboard available",
            error="No leaderboard found in session"
        )
        
    except Exception as e:
        return LeaderboardResponse(
            success=False,
            message="Failed to retrieve leaderboard",
            error=str(e)
        )

@agent.on_rest_post("/get-training-function", SessionRequest, CodeResponse)
async def get_training_function_post(ctx: Context, req: SessionRequest) -> CodeResponse:
    """Get training function from session (POST version)"""
    try:
        session = session_store.get_session(req.session_id)
        if not session:
            return CodeResponse(
                success=False,
                message="Session not found",
                error=f"Session {req.session_id} not found or expired"
            )
        
        ml_agent = session["agent"]
        
        if ml_agent.response and "training_function" in ml_agent.response:
            return CodeResponse(
                success=True,
                message="Training function retrieved successfully",
                generated_code=ml_agent.response["training_function"]
            )
        
        return CodeResponse(
            success=False,
            message="No training function available",
            error="No generated code found in session"
        )
        
    except Exception as e:
        return CodeResponse(
            success=False,
            message="Failed to retrieve training function",
            error=str(e)
        )

@agent.on_rest_post("/get-ml-steps", SessionRequest, GenericResponse)
async def get_ml_steps_post(ctx: Context, req: SessionRequest) -> GenericResponse:
    """Get ML recommendations from session (POST version)"""
    try:
        session = session_store.get_session(req.session_id)
        if not session:
            return GenericResponse(
                success=False,
                message="Session not found",
                error=f"Session {req.session_id} not found or expired"
            )
        
        ml_agent = session["agent"]
        
        if ml_agent.response and "ml_steps" in ml_agent.response:
            return GenericResponse(
                success=True,
                message="ML steps retrieved successfully",
                data=ml_agent.response["ml_steps"]
            )
        
        return GenericResponse(
            success=False,
            message="No ML steps available",
            error="No recommendations found in session"
        )
        
    except Exception as e:
        return GenericResponse(
            success=False,
            message="Failed to retrieve ML steps",
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
    print("🤖 Starting H2O ML Training uAgent (REST)...")
    print("📡 Available endpoints:")
    print("   GET  http://127.0.0.1:8008/health")
    print("   POST http://127.0.0.1:8008/train-model")
    print("   POST http://127.0.0.1:8008/train-model-csv")
    print("   GET  http://127.0.0.1:8008/session/{id}/leaderboard")
    print("   GET  http://127.0.0.1:8008/session/{id}/best-model-id")
    print("   GET  http://127.0.0.1:8008/session/{id}/model-path")
    print("   GET  http://127.0.0.1:8008/session/{id}/training-function")
    print("   GET  http://127.0.0.1:8008/session/{id}/ml-steps")
    print("   GET  http://127.0.0.1:8008/session/{id}/original-data")
    print("   GET  http://127.0.0.1:8008/session/{id}/workflow-summary")
    print("   GET  http://127.0.0.1:8008/session/{id}/logs")
    print("   GET  http://127.0.0.1:8008/session/{id}/full-response")
    print("   POST http://127.0.0.1:8008/session/{id}/delete")
    print("🚀 Agent starting...")
    agent.run()
