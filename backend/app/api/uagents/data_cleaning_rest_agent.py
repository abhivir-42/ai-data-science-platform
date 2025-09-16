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
import requests
from uuid import uuid4
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from fastapi import Request

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
from app.services.session_service import session_service
from app.core.database import init_database

# ============================================================================
# Database Session Management (Replaced in-memory SessionStore)
# ============================================================================

# Initialize database on startup
import asyncio
_db_initialized = False

async def ensure_database_initialized():
    """Ensure database is initialized (called once on startup)"""
    global _db_initialized
    if not _db_initialized:
        await init_database()
        _db_initialized = True

# ============================================================================
# JSON Serialization Utilities
# ============================================================================

def make_json_serializable(data):
    """Convert pandas/numpy types to JSON-serializable Python types"""
    import pandas as pd
    import numpy as np

    if isinstance(data, dict):
        return {k: make_json_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [make_json_serializable(item) for item in data]
    elif pd.isna(data):
        return None
    elif isinstance(data, pd.Timestamp):
        return data.isoformat()
    elif isinstance(data, np.datetime64):
        return pd.Timestamp(data).isoformat()
    elif hasattr(data, 'isoformat'):  # datetime/Timestamp
        return data.isoformat()
    elif isinstance(data, (np.integer, np.floating)):
        return data.item() if not np.isnan(data) else None
    elif isinstance(data, np.bool_):
        return bool(data)
    elif hasattr(data, 'item') and hasattr(data, 'dtype'):
        return data.item()
    else:
        return data

def dataframe_to_json_safe(df):
    """Convert DataFrame or data object to JSON-safe format"""
    if df is None:
        return {"records": [], "columns": []}
    
    # Handle actual pandas DataFrame
    if hasattr(df, 'empty') and hasattr(df, 'to_dict'):
        if df.empty:
            return {"records": [], "columns": []}
        
        records = df.to_dict(orient="records")
        cleaned_records = [make_json_serializable(record) for record in records]
        
        return {
            "records": cleaned_records,
            "columns": list(map(str, df.columns.tolist())),
            "shape": [int(df.shape[0]), int(df.shape[1])]
        }
    
    # Handle dictionary format (from proxy objects)
    elif isinstance(df, dict):
        if "records" in df:
            # Already in the right format
            records = df["records"]
            cleaned_records = [make_json_serializable(record) for record in records]
            return {
                "records": cleaned_records,
                "columns": df.get("columns", []),
                "shape": df.get("shape", [len(records), len(records[0]) if records else 0])
            }
        else:
            # Convert dict to records format
            if not df:
                return {"records": [], "columns": []}
            
            # Check if this is the problematic nested dict format from agent proxy
            # Format: {"col1": {"0": val1, "1": val2}, "col2": {"0": val1, "1": val2}}
            values_sample = list(df.values())[0] if df else None
            if isinstance(values_sample, dict) and all(isinstance(k, str) and k.isdigit() for k in values_sample.keys()):
                # This is the nested index format - convert to proper records
                columns = list(df.keys())
                if not columns:
                    return {"records": [], "columns": []}
                
                # Get the indices from the first column
                indices = sorted(df[columns[0]].keys(), key=int)
                records = []
                
                for idx in indices:
                    record = {}
                    for col in columns:
                        if idx in df[col]:
                            record[col] = df[col][idx]
                        else:
                            record[col] = None
                    records.append(make_json_serializable(record))
                
                return {
                    "records": records,
                    "columns": columns,
                    "shape": [len(records), len(columns)]
                }
            
            # Handle normal list format: {"col1": [val1, val2], "col2": [val1, val2]}
            elif isinstance(list(df.values())[0], list):
                columns = list(df.keys())
                num_records = len(df[columns[0]]) if columns else 0
                records = []
                for i in range(num_records):
                    record = {col: df[col][i] for col in columns}
                    records.append(make_json_serializable(record))
                
                return {
                    "records": records,
                    "columns": columns,
                    "shape": [num_records, len(columns)]
                }
            else:
                # Single record: {"col1": val1, "col2": val2}
                record = make_json_serializable(df)
                return {
                    "records": [record],
                    "columns": list(df.keys()),
                    "shape": [1, len(df)]
                }
    
    # Handle list format
    elif isinstance(df, list):
        if not df:
            return {"records": [], "columns": []}
        
        cleaned_records = [make_json_serializable(record) for record in df]
        columns = list(df[0].keys()) if df and isinstance(df[0], dict) else []
        
        return {
            "records": cleaned_records,
            "columns": columns,
            "shape": [len(df), len(columns)]
        }
    
    # Fallback for other types
    else:
        return {"records": [], "columns": [], "error": f"Unsupported data type: {type(df)}"}

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
    advanced_options: Optional[Dict[str, Any]] = None

class CleanFromSessionRequest(Model):
    session_id: str  # Session ID from data loader
    user_instructions: Optional[str] = None
    max_retries: int = 3
    advanced_options: Optional[Dict[str, Any]] = None

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
        # Ensure database is initialized
        await ensure_database_initialized()
        
        # Extract user_id from request for session association
        from app.core.auth_middleware import extract_user_id_from_request
        user_id = extract_user_id_from_request(ctx)
        
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
        
        # Create session using centralized service with user association
        session_id = await session_service.create_session(
            agent_instance=cleaning_agent,
            agent_type="cleaning",
            metadata={
                "operation": "clean_data",
                "user_instructions": req.user_instructions,
                "original_shape": list(df.shape),
                "execution_time": execution_time,
                "authenticated_user": user_id is not None
            },
            user_id=user_id
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
        # Ensure database is initialized
        await ensure_database_initialized()
        
        # Extract user_id from request for session association
        from app.core.auth_middleware import extract_user_id_from_request
        user_id = extract_user_id_from_request(ctx)
        
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
        
        # Create session using centralized service with user association
        session_id = await session_service.create_session(
            agent_instance=cleaning_agent,
            agent_type="cleaning",
            metadata={
                "operation": "clean_csv",
                "filename": req.filename,
                "user_instructions": req.user_instructions,
                "original_shape": list(df.shape),
                "execution_time": execution_time,
                "authenticated_user": user_id is not None
            },
            user_id=user_id
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

@agent.on_rest_post("/clean-from-session", CleanFromSessionRequest, SessionResponse)
async def clean_from_session(ctx: Context, req: CleanFromSessionRequest) -> SessionResponse:
    """Clean data from a previous data loader session"""
    try:
        # Ensure database is initialized
        await ensure_database_initialized()
        
        # Extract user_id from request for session association
        from app.core.auth_middleware import extract_user_id_from_request
        user_id = extract_user_id_from_request(ctx)
        
        start_time = time.time()
        
        # Fetch data from the data loader session
        data_loader_url = "http://127.0.0.1:8005"
        
        try:
            response = requests.post(
                f"{data_loader_url}/get-artifacts",
                json={"session_id": req.session_id, "as_dataframe": True},
                timeout=30
            )
            
            if not response.ok:
                return SessionResponse(
                    success=False,
                    message="Failed to fetch data from session",
                    session_id="",
                    error=f"Data loader request failed: {response.status_code} {response.text}"
                )
                
            loader_result = response.json()
            
            if not loader_result.get("success", False):
                return SessionResponse(
                    success=False,
                    message="Failed to retrieve data from session",
                    session_id="",
                    error=loader_result.get("error", "Unknown error from data loader")
                )
                
            # Extract DataFrame data from enhanced serialization format
            data_dict = loader_result.get("data", {})
            
            # Handle enhanced serialization format (new)
            if "data" in data_dict and isinstance(data_dict["data"], list):
                # Enhanced format: {"data": [...], "columns": [...], "shape": [...]}
                df_data = data_dict["data"]
                if not df_data:
                    return SessionResponse(
                        success=False,
                        message="No data found in session",
                        session_id="",
                        error="Session contains no data rows"
                    )
                df = pd.DataFrame(df_data)
                
            # Handle legacy format (fallback)
            elif "records" in data_dict:
                # Legacy format: {"records": [...], "columns": [...]}
                records = data_dict["records"]
                if not records:
                    return SessionResponse(
                        success=False,
                        message="No data found in session",
                        session_id="",
                        error="Session contains no data rows"
                    )
                df = pd.DataFrame(records)
                
            else:
                return SessionResponse(
                    success=False,
                    message="No data found in session",
                    session_id="",
                    error="Session contains no usable data format"
                )
            
        except requests.RequestException as e:
            return SessionResponse(
                success=False,
                message="Cannot connect to data loader service",
                session_id="",
                error=f"Network error: {str(e)}"
            )
        
        if df.empty:
            return SessionResponse(
                success=False,
                message="Empty dataset in session",
                session_id="",
                error="Session contains no data rows"
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
        
        # Create session using centralized service with user association
        session_id = await session_service.create_session(
            agent_instance=cleaning_agent,
            agent_type="cleaning",
            metadata={
                "operation": "clean_from_session",
                "source_session_id": req.session_id,
                "user_instructions": req.user_instructions,
                "advanced_options": req.advanced_options,
                "original_shape": list(df.shape),
                "execution_time": execution_time,
                "authenticated_user": user_id is not None
            },
            user_id=user_id
        )
        
        return SessionResponse(
            success=True,
            message="Data cleaning from session completed successfully",
            session_id=session_id,
            execution_time_seconds=execution_time
        )
        
    except Exception as e:
        return SessionResponse(
            success=False,
            message="Session-based data cleaning failed",
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
        # Extract user_id from request for session ownership validation
        from app.core.auth_middleware import extract_user_id_from_request
        user_id = extract_user_id_from_request(ctx)
        
        session = await session_service.get_session_with_auth(req.session_id, user_id)
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
        original_shape = None
        if original_df is not None:
            if hasattr(original_df, 'shape'):
                original_shape = list(original_df.shape)
            elif isinstance(original_df, dict) and 'records' in original_df:
                records = original_df['records']
                if records:
                    original_shape = [len(records), len(records[0]) if records else 0]
        
        # Handle processed shape
        processed_shape = None
        if hasattr(cleaned_df, 'shape'):
            processed_shape = list(cleaned_df.shape)
        elif isinstance(cleaned_df, dict) and 'records' in cleaned_df:
            records = cleaned_df['records']
            if records:
                processed_shape = [len(records), len(records[0]) if records else 0]
        
        return DataResponse(
            success=True,
            message="Cleaned data retrieved successfully",
            data=dataframe_to_json_safe(cleaned_df),
            original_shape=original_shape,
            processed_shape=processed_shape
        )
        
    except Exception as e:
        return DataResponse(
            success=False,
            message="Failed to retrieve cleaned data",
            error=str(e)
        )

@agent.on_rest_post("/get-original-data", SessionRequest, DataResponse)
async def get_original_data(ctx: Context, req: SessionRequest) -> DataResponse:
    """Get original dataset from session"""
    try:
        # Extract user_id from request for session ownership validation
        from app.core.auth_middleware import extract_user_id_from_request
        user_id = extract_user_id_from_request(ctx)
        
        session = await session_service.get_session_with_auth(req.session_id, user_id)
        if not session:
            return DataResponse(
                success=False,
                message="Session not found",
                error=f"Session {req.session_id} not found or expired"
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

@agent.on_rest_post("/get-cleaning-function", SessionRequest, CodeResponse)
async def get_cleaning_function(ctx: Context, req: SessionRequest) -> CodeResponse:
    """Get generated Python cleaning function from session"""
    try:
        # Extract user_id from request for session ownership validation
        from app.core.auth_middleware import extract_user_id_from_request
        user_id = extract_user_id_from_request(ctx)
        
        session = await session_service.get_session_with_auth(req.session_id, user_id)
        if not session:
            return CodeResponse(
                success=False,
                message="Session not found",
                error=f"Session {req.session_id} not found or expired"
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

@agent.on_rest_post("/get-cleaning-steps", SessionRequest, GenericResponse)
async def get_cleaning_steps(ctx: Context, req: SessionRequest) -> GenericResponse:
    """Get recommended cleaning steps from session"""
    try:
        # Extract user_id from request for session ownership validation
        from app.core.auth_middleware import extract_user_id_from_request
        user_id = extract_user_id_from_request(ctx)
        
        session = await session_service.get_session_with_auth(req.session_id, user_id)
        if not session:
            return GenericResponse(
                success=False,
                message="Session not found",
                error=f"Session {req.session_id} not found or expired"
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
        # Extract user_id from request for session ownership validation
        from app.core.auth_middleware import extract_user_id_from_request
        user_id = extract_user_id_from_request(ctx)
        
        session = await session_service.get_session_with_auth(session_id, user_id)
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

@agent.on_rest_post("/get-logs", SessionRequest, GenericResponse)
async def get_logs(ctx: Context, req: SessionRequest) -> GenericResponse:
    """Get execution logs from session"""
    try:
        # Extract user_id from request for session ownership validation
        from app.core.auth_middleware import extract_user_id_from_request
        user_id = extract_user_id_from_request(ctx)
        
        session = await session_service.get_session_with_auth(req.session_id, user_id)
        if not session:
            return GenericResponse(
                success=False,
                message="Session not found",
                error=f"Session {req.session_id} not found or expired"
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
        # Extract user_id from request for session ownership validation
        from app.core.auth_middleware import extract_user_id_from_request
        user_id = extract_user_id_from_request(ctx)
        
        session = await session_service.get_session_with_auth(session_id, user_id)
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

@agent.on_rest_get("/debug/sessions", GenericResponse)
async def debug_sessions(ctx: Context) -> GenericResponse:
    """Debug endpoint to list all sessions"""
    sessions = await session_service.list_sessions(agent_type="cleaning")
    return GenericResponse(
        success=True,
        message=f"Found {len(sessions)} sessions",
        data={"sessions": sessions, "count": len(sessions)}
    )

class DeleteSessionRequest(Model):
    session_id: str

@agent.on_rest_post("/delete-session", DeleteSessionRequest, GenericResponse)
async def delete_session(ctx: Context, req: DeleteSessionRequest) -> GenericResponse:
    """Delete a session"""
    try:
        deleted = await session_service.delete_session(req.session_id)
        
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
    print("   POST http://127.0.0.1:8004/clean-from-session")
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
