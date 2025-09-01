#!/usr/bin/env python3
"""
ML Prediction uAgent REST API.

This uAgent provides comprehensive REST endpoints for the MLPredictionAgent,
exposing ALL agent capabilities including:
- Single and batch predictions
- Model analysis and questions
- Model loading from various sources
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
from app.agents import MLPredictionAgent

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

def _create_ml_prediction_agent(model_metrics=None, target_variable=None, config=None):
    """Create MLPredictionAgent instance with proper parameters"""
    if not model_metrics or not target_variable or not config:
        # Create dummy instances for REST endpoint testing
        from app.schemas.data_analysis_schemas import MLModelingMetrics
        from app.uagent_v2.config import UAgentConfig
        
        dummy_metrics = MLModelingMetrics(
            models_trained=1,
            best_model_type="GBM",
            best_model_id="dummy_model",
            best_model_score=0.85,
            cross_validation_score=0.82,
            test_set_score=0.83,
            training_time_seconds=120.0,
            model_size_mb=5.2,
            features_used=["feature1", "feature2", "feature3"],
            feature_importance={"feature1": 0.5, "feature2": 0.3, "feature3": 0.2},
            mlflow_experiment_id="dummy_experiment",
            mlflow_run_id="dummy_run",
            model_path=None
        )
        dummy_config = UAgentConfig()
        dummy_target = "target"
        
        return MLPredictionAgent(
            model_metrics=dummy_metrics,
            target_variable=dummy_target, 
            config=dummy_config
        )
    else:
        return MLPredictionAgent(
            model_metrics=model_metrics,
            target_variable=target_variable,
            config=config
        )

# ============================================================================
# uAgent Setup
# ============================================================================

agent = Agent(
    name="ml_prediction_rest_uagent",
    port=8009,
    seed="ml_prediction_rest_uagent_secret_seed",
    endpoint=["http://127.0.0.1:8009/submit"],
)

fund_agent_if_low(agent.wallet.address())

# ============================================================================
# Pydantic Models
# ============================================================================

class HealthResponse(Model):
    status: str
    agent: str

class PredictSingleRequest(Model):
    input_data: Dict[str, Any]
    model_session_id: Optional[str] = None  # Reference to training session
    model_path: Optional[str] = None  # Direct model path

class PredictBatchRequest(Model):
    data_source: str  # CSV URL or file path
    model_session_id: Optional[str] = None  # Reference to training session
    model_path: Optional[str] = None  # Direct model path

class AnalyzeModelRequest(Model):
    query: str
    model_session_id: Optional[str] = None  # Reference to training session
    model_path: Optional[str] = None  # Direct model path

class LoadModelRequest(Model):
    model_path: str
    model_type: Optional[str] = "h2o"  # "h2o", "sklearn", "auto"

class SessionResponse(Model):
    success: bool
    message: str
    session_id: str
    execution_time_seconds: Optional[float] = None
    error: Optional[str] = None

class PredictionResponse(Model):
    success: bool
    message: str
    prediction: Optional[Any] = None
    prediction_probability: Optional[Dict[str, float]] = None
    confidence: Optional[float] = None
    error: Optional[str] = None

class BatchPredictionResponse(Model):
    success: bool
    message: str
    predictions: Optional[List[Any]] = None
    prediction_count: Optional[int] = None
    batch_summary: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class ModelAnalysisResponse(Model):
    success: bool
    message: str
    analysis: Optional[str] = None
    model_info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class GenericResponse(Model):
    success: bool
    message: str
    data: Optional[Any] = None
    error: Optional[str] = None

class DataResponse(Model):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    original_shape: Optional[List[int]] = None
    processed_shape: Optional[List[int]] = None
    error: Optional[str] = None

class SessionRequest(Model):
    session_id: str

class DeleteSessionRequest(Model):
    session_id: str

# ============================================================================
# Main Processing Endpoints
# ============================================================================

@agent.on_rest_post("/predict-single", PredictSingleRequest, SessionResponse)
async def predict_single(ctx: Context, req: PredictSingleRequest) -> SessionResponse:
    """Make single prediction and create session"""
    try:
        start_time = time.time()
        
        # Create agent instance
        pred_agent = _create_ml_prediction_agent()
        
        # Load model if needed
        if req.model_path:
            pred_agent.load_model(req.model_path)
        elif req.model_session_id:
            # TODO: Load model from training session
            pass
        
        # Make prediction
        prediction_result = pred_agent.predict_single(req.input_data)
        
        execution_time = time.time() - start_time
        
        # Create session
        session_id = session_store.create_session(
            pred_agent,
            metadata={
                "operation": "predict_single",
                "input_data": req.input_data,
                "model_session_id": req.model_session_id,
                "model_path": req.model_path,
                "execution_time": execution_time,
                "prediction_result": prediction_result
            }
        )
        
        return SessionResponse(
            success=True,
            message="Single prediction completed successfully",
            session_id=session_id,
            execution_time_seconds=execution_time
        )
        
    except Exception as e:
        return SessionResponse(
            success=False,
            message="Single prediction failed",
            session_id="",
            error=str(e)
        )

@agent.on_rest_post("/predict-batch", PredictBatchRequest, SessionResponse)
async def predict_batch(ctx: Context, req: PredictBatchRequest) -> SessionResponse:
    """Make batch predictions and create session"""
    try:
        start_time = time.time()
        
        # Create agent instance
        pred_agent = _create_ml_prediction_agent()
        
        # Load model if needed
        if req.model_path:
            pred_agent.load_model(req.model_path)
        elif req.model_session_id:
            # TODO: Load model from training session
            pass
        
        # Make batch predictions
        batch_results = pred_agent.predict_batch(req.data_source)
        
        execution_time = time.time() - start_time
        
        # Create session
        session_id = session_store.create_session(
            pred_agent,
            metadata={
                "operation": "predict_batch",
                "data_source": req.data_source,
                "model_session_id": req.model_session_id,
                "model_path": req.model_path,
                "execution_time": execution_time,
                "batch_results": batch_results
            }
        )
        
        return SessionResponse(
            success=True,
            message="Batch predictions completed successfully",
            session_id=session_id,
            execution_time_seconds=execution_time
        )
        
    except Exception as e:
        return SessionResponse(
            success=False,
            message="Batch predictions failed",
            session_id="",
            error=str(e)
        )

@agent.on_rest_post("/analyze-model", AnalyzeModelRequest, SessionResponse)
async def analyze_model(ctx: Context, req: AnalyzeModelRequest) -> SessionResponse:
    """Analyze model and answer questions, create session"""
    try:
        start_time = time.time()
        
        # Create agent instance
        pred_agent = _create_ml_prediction_agent()
        
        # Load model if needed
        if req.model_path:
            pred_agent.load_model(req.model_path)
        elif req.model_session_id:
            # TODO: Load model from training session
            pass
        
        # Analyze model
        analysis_result = pred_agent.analyze_model(req.query)
        
        execution_time = time.time() - start_time
        
        # Create session
        session_id = session_store.create_session(
            pred_agent,
            metadata={
                "operation": "analyze_model",
                "query": req.query,
                "model_session_id": req.model_session_id,
                "model_path": req.model_path,
                "execution_time": execution_time,
                "analysis_result": analysis_result
            }
        )
        
        return SessionResponse(
            success=True,
            message="Model analysis completed successfully",
            session_id=session_id,
            execution_time_seconds=execution_time
        )
        
    except Exception as e:
        return SessionResponse(
            success=False,
            message="Model analysis failed",
            session_id="",
            error=str(e)
        )

@agent.on_rest_post("/load-model", LoadModelRequest, SessionResponse)
async def load_model(ctx: Context, req: LoadModelRequest) -> SessionResponse:
    """Load model from path and create session"""
    try:
        start_time = time.time()
        
        # Create agent instance
        pred_agent = _create_ml_prediction_agent()
        
        # Load model
        load_result = pred_agent.load_model(req.model_path)
        
        execution_time = time.time() - start_time
        
        # Create session
        session_id = session_store.create_session(
            pred_agent,
            metadata={
                "operation": "load_model",
                "model_path": req.model_path,
                "model_type": req.model_type,
                "execution_time": execution_time,
                "load_result": load_result
            }
        )
        
        return SessionResponse(
            success=True,
            message="Model loaded successfully",
            session_id=session_id,
            execution_time_seconds=execution_time
        )
        
    except Exception as e:
        return SessionResponse(
            success=False,
            message="Model loading failed",
            session_id="",
            error=str(e)
        )

# ============================================================================
# Session-Based Result Access Endpoints
# ============================================================================

@agent.on_rest_get("/session/{session_id}/prediction-results", PredictionResponse)
async def get_prediction_results(ctx: Context, session_id: str) -> PredictionResponse:
    """Get prediction results from session"""
    try:
        session = session_store.get_session(session_id)
        if not session:
            return PredictionResponse(
                success=False,
                message="Session not found",
                error=f"Session {session_id} not found or expired"
            )
        
        metadata = session["metadata"]
        
        if metadata.get("operation") == "predict_single":
            prediction_result = metadata.get("prediction_result")
            
            if prediction_result is None:
                return PredictionResponse(
                    success=False,
                    message="No prediction results available",
                    error="Prediction may have failed or not completed"
                )
            
            # Extract prediction details
            prediction = prediction_result
            prediction_probability = None
            confidence = None
            
            # If prediction result is a dict with more details
            if isinstance(prediction_result, dict):
                prediction = prediction_result.get("prediction")
                prediction_probability = prediction_result.get("probability")
                confidence = prediction_result.get("confidence")
            
            return PredictionResponse(
                success=True,
                message="Prediction results retrieved successfully",
                prediction=make_json_serializable(prediction),
                prediction_probability=make_json_serializable(prediction_probability),
                confidence=confidence
            )
        else:
            return PredictionResponse(
                success=False,
                message="Session does not contain single prediction results",
                error=f"Session operation was '{metadata.get('operation')}', not 'predict_single'"
            )
        
    except Exception as e:
        return PredictionResponse(
            success=False,
            message="Failed to retrieve prediction results",
            error=str(e)
        )

@agent.on_rest_get("/session/{session_id}/batch-results", BatchPredictionResponse)
async def get_batch_results(ctx: Context, session_id: str) -> BatchPredictionResponse:
    """Get batch prediction results from session"""
    try:
        session = session_store.get_session(session_id)
        if not session:
            return BatchPredictionResponse(
                success=False,
                message="Session not found",
                error=f"Session {session_id} not found or expired"
            )
        
        metadata = session["metadata"]
        
        if metadata.get("operation") == "predict_batch":
            batch_results = metadata.get("batch_results")
            
            if batch_results is None:
                return BatchPredictionResponse(
                    success=False,
                    message="No batch results available",
                    error="Batch prediction may have failed or not completed"
                )
            
            # Extract batch details
            predictions = batch_results
            prediction_count = None
            batch_summary = None
            
            # If batch results is more structured
            if isinstance(batch_results, dict):
                predictions = batch_results.get("predictions")
                prediction_count = batch_results.get("count")
                batch_summary = batch_results.get("summary")
            elif isinstance(batch_results, list):
                prediction_count = len(batch_results)
            
            return BatchPredictionResponse(
                success=True,
                message="Batch results retrieved successfully",
                predictions=make_json_serializable(predictions),
                prediction_count=prediction_count,
                batch_summary=make_json_serializable(batch_summary)
            )
        else:
            return BatchPredictionResponse(
                success=False,
                message="Session does not contain batch prediction results",
                error=f"Session operation was '{metadata.get('operation')}', not 'predict_batch'"
            )
        
    except Exception as e:
        return BatchPredictionResponse(
            success=False,
            message="Failed to retrieve batch results",
            error=str(e)
        )

@agent.on_rest_get("/session/{session_id}/model-analysis", ModelAnalysisResponse)
async def get_model_analysis(ctx: Context, session_id: str) -> ModelAnalysisResponse:
    """Get model analysis results from session"""
    try:
        session = session_store.get_session(session_id)
        if not session:
            return ModelAnalysisResponse(
                success=False,
                message="Session not found",
                error=f"Session {session_id} not found or expired"
            )
        
        metadata = session["metadata"]
        
        if metadata.get("operation") == "analyze_model":
            analysis_result = metadata.get("analysis_result")
            
            if analysis_result is None:
                return ModelAnalysisResponse(
                    success=False,
                    message="No analysis results available",
                    error="Model analysis may have failed or not completed"
                )
            
            # Extract analysis details
            analysis = analysis_result
            model_info = None
            
            # If analysis result is more structured
            if isinstance(analysis_result, dict):
                analysis = analysis_result.get("analysis")
                model_info = analysis_result.get("model_info")
            
            return ModelAnalysisResponse(
                success=True,
                message="Model analysis retrieved successfully",
                analysis=analysis,
                model_info=make_json_serializable(model_info)
            )
        else:
            return ModelAnalysisResponse(
                success=False,
                message="Session does not contain model analysis results",
                error=f"Session operation was '{metadata.get('operation')}', not 'analyze_model'"
            )
        
    except Exception as e:
        return ModelAnalysisResponse(
            success=False,
            message="Failed to retrieve model analysis",
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
        agent="ml_prediction_rest_uagent"
    )

# ============================================================================
# POST Session Access Endpoints (Working)
# ============================================================================

@agent.on_rest_post("/get-model-analysis", SessionRequest, ModelAnalysisResponse)
async def get_model_analysis_post(ctx: Context, req: SessionRequest) -> ModelAnalysisResponse:
    """Get model analysis from session (POST version)"""
    try:
        session = session_store.get_session(req.session_id)
        if not session:
            return ModelAnalysisResponse(
                success=False,
                message="Session not found",
                error=f"Session {req.session_id} not found or expired"
            )
        
        pred_agent = session["agent"]
        
        if pred_agent.response and "model_analysis" in pred_agent.response:
            return ModelAnalysisResponse(
                success=True,
                message="Model analysis retrieved successfully",
                analysis=pred_agent.response["model_analysis"]
            )
        
        return ModelAnalysisResponse(
            success=False,
            message="No model analysis available",
            error="No analysis found in session"
        )
        
    except Exception as e:
        return ModelAnalysisResponse(
            success=False,
            message="Failed to retrieve model analysis",
            error=str(e)
        )

@agent.on_rest_post("/get-prediction-results", SessionRequest, DataResponse)
async def get_prediction_results_post(ctx: Context, req: SessionRequest) -> DataResponse:
    """Get prediction results from session (POST version)"""
    try:
        session = session_store.get_session(req.session_id)
        if not session:
            return DataResponse(
                success=False,
                message="Session not found",
                error=f"Session {req.session_id} not found or expired"
            )
        
        pred_agent = session["agent"]
        
        # Get prediction results from session metadata or agent
        metadata = session.get("metadata", {})
        prediction_result = metadata.get("prediction_result")
        
        if prediction_result is None:
            return DataResponse(
                success=False,
                message="No prediction results available",
                error="Predictions were not found in session"
            )
        
        return DataResponse(
            success=True,
            message="Prediction results retrieved successfully",
            data=prediction_result
        )
        
    except Exception as e:
        return DataResponse(
            success=False,
            message="Failed to retrieve prediction results",
            error=str(e)
        )

@agent.on_rest_post("/get-logs", SessionRequest, GenericResponse)
async def get_logs_post(ctx: Context, req: SessionRequest) -> GenericResponse:
    """Get prediction execution logs from session (POST version)"""
    try:
        session = session_store.get_session(req.session_id)
        if not session:
            return GenericResponse(
                success=False,
                message="Session not found",
                error=f"Session {req.session_id} not found or expired"
            )
        
        pred_agent = session["agent"]
        
        # Get logs from the agent
        logs = pred_agent.get_logs() if hasattr(pred_agent, 'get_logs') else []
        
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
    print("🔮 Starting ML Prediction uAgent (REST)...")
    print("📡 Available endpoints:")
    print("   GET  http://127.0.0.1:8009/health")
    print("   POST http://127.0.0.1:8009/predict-single")
    print("   POST http://127.0.0.1:8009/predict-batch")
    print("   POST http://127.0.0.1:8009/analyze-model")
    print("   POST http://127.0.0.1:8009/load-model")
    print("   POST http://127.0.0.1:8009/get-prediction-results")
    print("   POST http://127.0.0.1:8009/get-batch-results")
    print("   POST http://127.0.0.1:8009/get-model-analysis")
    print("   POST http://127.0.0.1:8009/delete-session")
    print("🚀 Agent starting...")
    agent.run()
