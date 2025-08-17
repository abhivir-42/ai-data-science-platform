#!/usr/bin/env python3
"""
Data Loader uAgent REST API.

This uAgent provides comprehensive REST endpoints for the DataLoaderToolsAgent,
exposing ALL agent capabilities including:
- File loading (CSV, Excel, JSON, Parquet, etc.)
- Directory loading
- PDF extraction
- Access to loaded artifacts, AI messages, tool calls
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
from app.agents import DataLoaderToolsAgent

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

def _create_data_loader_agent():
    """Create DataLoaderToolsAgent instance"""
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_key, temperature=0.1)
    return DataLoaderToolsAgent(model=llm)

# ============================================================================
# uAgent Setup
# ============================================================================

agent = Agent(
    name="data_loader_rest_uagent",
    port=8005,
    seed="data_loader_rest_uagent_secret_seed",
    endpoint=["http://127.0.0.1:8005/submit"],
)

fund_agent_if_low(agent.wallet.address())

# ============================================================================
# Pydantic Models
# ============================================================================

class HealthResponse(Model):
    status: str
    agent: str

class LoadFileRequest(Model):
    file_path: str
    user_instructions: Optional[str] = None

class LoadDirectoryRequest(Model):
    directory_path: str
    user_instructions: Optional[str] = None

class ExtractPDFRequest(Model):
    pdf_path: str
    extraction_type: str = "smart"  # "text", "tables", "smart"
    user_instructions: Optional[str] = None

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
    processed_shape: Optional[List[int]] = None
    error: Optional[str] = None

class GenericResponse(Model):
    success: bool
    message: str
    data: Optional[Any] = None
    error: Optional[str] = None

# ============================================================================
# Main Processing Endpoints
# ============================================================================

@agent.on_rest_post("/load-file", LoadFileRequest, SessionResponse)
async def load_file(ctx: Context, req: LoadFileRequest) -> SessionResponse:
    """Load data from a file and create session"""
    try:
        start_time = time.time()
        
        # Create agent instance
        loader_agent = _create_data_loader_agent()
        
        # Create instructions for the agent
        instructions = f"Load the file from: {req.file_path}"
        if req.user_instructions:
            instructions += f"\n\nAdditional instructions: {req.user_instructions}"
        
        # Execute data loading
        loader_agent.invoke_agent(user_instructions=instructions)
        
        execution_time = time.time() - start_time
        
        # Create session
        session_id = session_store.create_session(
            loader_agent,
            metadata={
                "operation": "load_file",
                "file_path": req.file_path,
                "user_instructions": req.user_instructions,
                "execution_time": execution_time
            }
        )
        
        return SessionResponse(
            success=True,
            message=f"File loading completed successfully from {req.file_path}",
            session_id=session_id,
            execution_time_seconds=execution_time
        )
        
    except Exception as e:
        return SessionResponse(
            success=False,
            message="File loading failed",
            session_id="",
            error=str(e)
        )

@agent.on_rest_post("/load-directory", LoadDirectoryRequest, SessionResponse)
async def load_directory(ctx: Context, req: LoadDirectoryRequest) -> SessionResponse:
    """Load data from multiple files in a directory and create session"""
    try:
        start_time = time.time()
        
        # Create agent instance
        loader_agent = _create_data_loader_agent()
        
        # Create instructions for the agent
        instructions = f"Load all data files from directory: {req.directory_path}"
        if req.user_instructions:
            instructions += f"\n\nAdditional instructions: {req.user_instructions}"
        
        # Execute directory loading
        loader_agent.invoke_agent(user_instructions=instructions)
        
        execution_time = time.time() - start_time
        
        # Create session
        session_id = session_store.create_session(
            loader_agent,
            metadata={
                "operation": "load_directory",
                "directory_path": req.directory_path,
                "user_instructions": req.user_instructions,
                "execution_time": execution_time
            }
        )
        
        return SessionResponse(
            success=True,
            message=f"Directory loading completed successfully from {req.directory_path}",
            session_id=session_id,
            execution_time_seconds=execution_time
        )
        
    except Exception as e:
        return SessionResponse(
            success=False,
            message="Directory loading failed",
            session_id="",
            error=str(e)
        )

@agent.on_rest_post("/extract-pdf", ExtractPDFRequest, SessionResponse)
async def extract_pdf(ctx: Context, req: ExtractPDFRequest) -> SessionResponse:
    """Extract data from PDF documents and create session"""
    try:
        start_time = time.time()
        
        # Create agent instance
        loader_agent = _create_data_loader_agent()
        
        # Create instructions for the agent
        if req.extraction_type == "text":
            instructions = f"Extract text content from PDF: {req.pdf_path}"
        elif req.extraction_type == "tables":
            instructions = f"Extract tables from PDF: {req.pdf_path}"
        else:  # smart extraction
            instructions = f"Intelligently extract data from PDF: {req.pdf_path}"
        
        if req.user_instructions:
            instructions += f"\n\nAdditional instructions: {req.user_instructions}"
        
        # Execute PDF extraction
        loader_agent.invoke_agent(user_instructions=instructions)
        
        execution_time = time.time() - start_time
        
        # Create session
        session_id = session_store.create_session(
            loader_agent,
            metadata={
                "operation": "extract_pdf",
                "pdf_path": req.pdf_path,
                "extraction_type": req.extraction_type,
                "user_instructions": req.user_instructions,
                "execution_time": execution_time
            }
        )
        
        return SessionResponse(
            success=True,
            message=f"PDF extraction completed successfully from {req.pdf_path}",
            session_id=session_id,
            execution_time_seconds=execution_time
        )
        
    except Exception as e:
        return SessionResponse(
            success=False,
            message="PDF extraction failed",
            session_id="",
            error=str(e)
        )

# ============================================================================
# Session-Based Result Access Endpoints
# ============================================================================

class SessionRequest(Model):
    session_id: str
    as_dataframe: bool = True

@agent.on_rest_post("/get-artifacts", SessionRequest, DataResponse)
async def get_artifacts(ctx: Context, req: SessionRequest) -> DataResponse:
    """Get loaded data artifacts from session"""
    try:
        session = session_store.get_session(req.session_id)
        if not session:
            return DataResponse(
                success=False,
                message="Session not found",
                error=f"Session {req.session_id} not found or expired"
            )
        
        loader_agent = session["agent"]
        artifacts = loader_agent.get_artifacts(as_dataframe=req.as_dataframe)
        
        if artifacts is None:
            return DataResponse(
                success=False,
                message="No artifacts available",
                error="No data was loaded in this session"
            )
        
        # If artifacts is a DataFrame, convert to JSON-safe format
        if req.as_dataframe and hasattr(artifacts, 'shape'):
            data = dataframe_to_json_safe(artifacts)
            shape = list(artifacts.shape)
        else:
            data = make_json_serializable(artifacts)
            shape = None
        
        return DataResponse(
            success=True,
            message="Artifacts retrieved successfully",
            data=data,
            processed_shape=shape
        )
        
    except Exception as e:
        return DataResponse(
            success=False,
            message="Failed to retrieve artifacts",
            error=str(e)
        )

@agent.on_rest_get("/session/{session_id}/ai-message", GenericResponse)
async def get_ai_message(ctx: Context, session_id: str) -> GenericResponse:
    """Get AI message from session"""
    try:
        session = session_store.get_session(session_id)
        if not session:
            return GenericResponse(
                success=False,
                message="Session not found",
                error=f"Session {session_id} not found or expired"
            )
        
        loader_agent = session["agent"]
        ai_message = loader_agent.get_ai_message()
        
        return GenericResponse(
            success=True,
            message="AI message retrieved successfully",
            data=ai_message
        )
        
    except Exception as e:
        return GenericResponse(
            success=False,
            message="Failed to retrieve AI message",
            error=str(e)
        )

@agent.on_rest_get("/session/{session_id}/tool-calls", GenericResponse)
async def get_tool_calls(ctx: Context, session_id: str) -> GenericResponse:
    """Get tool calls from session"""
    try:
        session = session_store.get_session(session_id)
        if not session:
            return GenericResponse(
                success=False,
                message="Session not found",
                error=f"Session {session_id} not found or expired"
            )
        
        loader_agent = session["agent"]
        tool_calls = loader_agent.get_tool_calls()
        
        return GenericResponse(
            success=True,
            message="Tool calls retrieved successfully",
            data=tool_calls or []
        )
        
    except Exception as e:
        return GenericResponse(
            success=False,
            message="Failed to retrieve tool calls",
            error=str(e)
        )

@agent.on_rest_get("/session/{session_id}/internal-messages", GenericResponse)
async def get_internal_messages(ctx: Context, session_id: str) -> GenericResponse:
    """Get internal messages from session"""
    try:
        session = session_store.get_session(session_id)
        if not session:
            return GenericResponse(
                success=False,
                message="Session not found",
                error=f"Session {session_id} not found or expired"
            )
        
        loader_agent = session["agent"]
        internal_messages = loader_agent.get_internal_messages()
        
        # Convert messages to serializable format
        if hasattr(internal_messages, '__iter__') and not isinstance(internal_messages, str):
            serializable_messages = []
            for msg in internal_messages:
                if hasattr(msg, 'content'):
                    serializable_messages.append({
                        "type": getattr(msg, 'type', 'unknown'),
                        "content": msg.content,
                        "id": getattr(msg, 'id', None)
                    })
                else:
                    serializable_messages.append(str(msg))
            messages = serializable_messages
        else:
            messages = internal_messages
        
        return GenericResponse(
            success=True,
            message="Internal messages retrieved successfully",
            data=messages or []
        )
        
    except Exception as e:
        return GenericResponse(
            success=False,
            message="Failed to retrieve internal messages",
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
        
        loader_agent = session["agent"]
        response = loader_agent.response
        
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
        agent="data_loader_rest_uagent"
    )

@agent.on_rest_get("/supported-formats", GenericResponse)
async def get_supported_formats(ctx: Context) -> GenericResponse:
    """Get list of supported file formats"""
    return GenericResponse(
        success=True,
        message="Supported file formats retrieved",
        data={
            "structured_data": [
                "csv", "tsv", "xlsx", "xls", "json", "jsonl", 
                "parquet", "feather", "pickle", "hdf5"
            ],
            "documents": [
                "pdf", "txt", "docx", "html", "xml"
            ],
            "databases": [
                "sqlite", "postgresql", "mysql", "mongodb"
            ],
            "web": [
                "http", "https", "ftp", "sftp"
            ],
            "notes": {
                "csv": "Comma-separated values with automatic delimiter detection",
                "excel": "Both .xlsx and .xls formats supported with sheet selection",
                "json": "JSON and JSONL (newline-delimited JSON) formats",
                "pdf": "Text extraction, table extraction, and smart extraction modes",
                "web": "Direct loading from URLs with authentication support"
            }
        }
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
    print("📁 Starting Data Loader uAgent (REST)...")
    print("📡 Available endpoints:")
    print("   GET  http://127.0.0.1:8005/health")
    print("   POST http://127.0.0.1:8005/load-file")
    print("   POST http://127.0.0.1:8005/load-directory")
    print("   POST http://127.0.0.1:8005/extract-pdf")
    print("   POST http://127.0.0.1:8005/get-artifacts")
    print("   GET  http://127.0.0.1:8005/session/{id}/ai-message")
    print("   GET  http://127.0.0.1:8005/session/{id}/tool-calls")
    print("   GET  http://127.0.0.1:8005/session/{id}/internal-messages")
    print("   GET  http://127.0.0.1:8005/session/{id}/full-response")
    print("   GET  http://127.0.0.1:8005/supported-formats")
    print("   POST http://127.0.0.1:8005/delete-session")
    print("🚀 Agent starting...")
    agent.run()
