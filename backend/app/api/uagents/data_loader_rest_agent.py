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
from app.agents import DataLoaderToolsAgent
from app.services.session_service import session_service
from app.core.database import init_database
from app.core.logging import logger

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

    # Handle dictionary format (from enhanced serialization)
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
    file_path: Optional[str] = None
    filename: Optional[str] = None
    file_content: Optional[str] = None  # base64-encoded content
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
    """Load data from a file path or base64 content and create session"""
    try:
        # Ensure database is initialized
        await ensure_database_initialized()
        
        # Extract user_id from request for session association
        from app.core.auth_middleware import extract_user_id_from_request
        user_id = extract_user_id_from_request(ctx)
        
        start_time = time.time()
        
        # Validate request parameters
        if req.file_path and (req.filename or req.file_content):
            return SessionResponse(
                success=False,
                message="Cannot specify both file_path and file_content parameters",
                session_id="",
                error="Ambiguous request: provide either file_path OR filename+file_content"
            )
        
        if not req.file_path and not (req.filename and req.file_content):
            return SessionResponse(
                success=False,
                message="Missing required parameters",
                session_id="",
                error="Must provide either file_path OR filename+file_content"
            )
        
        # Create agent instance
        loader_agent = _create_data_loader_agent()
        
        if req.file_path:
            # File path mode - existing logic
            instructions = f"Load the file from: {req.file_path}"
            if req.user_instructions:
                instructions += f"\n\nAdditional instructions: {req.user_instructions}"
            
            # Execute data loading
            loader_agent.invoke_agent(user_instructions=instructions)
            
            execution_time = time.time() - start_time
            
            # Create session using centralized service
            session_id = await session_service.create_session(
                agent_instance=loader_agent,
                agent_type="loading",
                metadata={
                    "operation": "load_file",
                    "file_path": req.file_path,
                    "user_instructions": req.user_instructions,
                    "execution_time": execution_time,
                    "authenticated_user": user_id is not None
                },
                user_id=user_id
            )
            
            return SessionResponse(
                success=True,
                message=f"File loading completed successfully from {req.file_path}",
                session_id=session_id,
                execution_time_seconds=execution_time
            )
            
        else:
            # Base64 content mode - decode and process directly
            try:
                # Validate base64 content before decoding
                if not req.file_content or not req.file_content.strip():
                    return SessionResponse(
                        success=False,
                        message="Empty file content provided",
                        session_id="",
                        error="File content is empty or missing"
                    )

                # Check for basic base64 validity
                if len(req.file_content) % 4 != 0 and not req.file_content.endswith('='):
                    logger.warning(f"Base64 content length {len(req.file_content)} may be invalid (not multiple of 4)")

                # Attempt base64 decoding with detailed error handling
                try:
                    decoded = base64.b64decode(req.file_content, validate=True)
                except Exception as b64_error:
                    logger.error(f"Base64 decoding failed: {b64_error}")
                    return SessionResponse(
                        success=False,
                        message="Invalid base64 file content",
                        session_id="",
                        error=f"Base64 decoding failed: {str(b64_error)}. Content length: {len(req.file_content)}"
                    )

                # Decode to UTF-8 with error handling
                try:
                    file_content = decoded.decode("utf-8", errors="replace")
                except Exception as decode_error:
                    logger.error(f"UTF-8 decoding failed: {decode_error}")
                    return SessionResponse(
                        success=False,
                        message="File content encoding error",
                        session_id="",
                        error=f"Failed to decode content as UTF-8: {str(decode_error)}"
                    )

                # Validate that it's not empty after decoding
                if not file_content.strip():
                    return SessionResponse(
                        success=False,
                        message="Empty file content after decoding",
                        session_id="",
                        error="File content is empty after base64 decoding"
                    )

                # Log successful decoding for debugging
                logger.info(f"Successfully decoded base64 content: {len(file_content)} characters")
                
                # Parse CSV content into DataFrame with error handling
                try:
                    # Try to detect the separator
                    sample = file_content[:1000]  # First 1000 chars for detection
                    potential_separators = [',', '\t', ';', '|']
                    detected_sep = ','  # default

                    for sep in potential_separators:
                        if sep in sample and sample.count(sep) > sample.count(detected_sep):
                            detected_sep = sep

                    # Parse CSV with detected separator
                    df = pd.read_csv(io.StringIO(file_content), sep=detected_sep, engine='python')

                except pd.errors.EmptyDataError:
                    return SessionResponse(
                        success=False,
                        message="Empty or invalid CSV file",
                        session_id="",
                        error="File appears to be empty or contains no readable data"
                    )
                except pd.errors.ParserError as parse_error:
                    logger.error(f"CSV parsing failed: {parse_error}")
                    return SessionResponse(
                        success=False,
                        message="CSV parsing error",
                        session_id="",
                        error=f"Failed to parse CSV content: {str(parse_error)}. Check file format and separators."
                    )
                except Exception as csv_error:
                    logger.error(f"Unexpected error during CSV parsing: {csv_error}")
                    return SessionResponse(
                        success=False,
                        message="File processing error",
                        session_id="",
                        error=f"Failed to process file content: {str(csv_error)}"
                    )

                # Validate DataFrame after parsing
                if df.empty:
                    return SessionResponse(
                        success=False,
                        message="Empty file",
                        session_id="",
                        error="File contains no data rows after parsing"
                    )

                logger.info(f"Successfully parsed CSV: {df.shape[0]} rows, {df.shape[1]} columns")
                
                # Create instructions for the agent
                instructions = f"Process the uploaded data file"
                if req.filename:
                    instructions += f": {req.filename}"
                if req.user_instructions:
                    instructions += f"\n\nAdditional instructions: {req.user_instructions}"
                
                # Execute data loading with direct data (no file I/O)
                loader_agent.invoke_agent(
                    user_instructions=instructions,
                    data_raw=df
                )
                
                execution_time = time.time() - start_time
                
                # Create session using centralized service with user association
                session_id = await session_service.create_session(
                    agent_instance=loader_agent,
                    agent_type="loading",
                    metadata={
                        "operation": "load_file",
                        "filename": req.filename,
                        "user_instructions": req.user_instructions,
                        "original_shape": list(df.shape),
                        "direct_data_mode": True,
                        "execution_time": execution_time,
                        "authenticated_user": user_id is not None
                    },
                    user_id=user_id
                )
                
                return SessionResponse(
                    success=True,
                    message=f"File loading completed successfully: {req.filename}",
                    session_id=session_id,
                    execution_time_seconds=execution_time
                )
                
            except Exception as e:
                return SessionResponse(
                    success=False,
                    message="Invalid file content",
                    session_id="",
                    error=f"Failed to process file content: {str(e)}"
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
        # Ensure database is initialized
        await ensure_database_initialized()
        
        # Extract user_id from request for session association
        from app.core.auth_middleware import extract_user_id_from_request
        user_id = extract_user_id_from_request(ctx)
        
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
        
        # Create session with user association
        session_id = await session_service.create_session(
            agent_instance=loader_agent,
            agent_type="loading",
            metadata={
                "operation": "load_directory",
                "directory_path": req.directory_path,
                "user_instructions": req.user_instructions,
                "execution_time": execution_time,
                "authenticated_user": user_id is not None
            },
            user_id=user_id
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
        # Ensure database is initialized
        await ensure_database_initialized()
        
        # Extract user_id from request for session association
        from app.core.auth_middleware import extract_user_id_from_request
        user_id = extract_user_id_from_request(ctx)
        
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
        
        # Create session with user association
        session_id = await session_service.create_session(
            agent_instance=loader_agent,
            agent_type="loading",
            metadata={
                "operation": "extract_pdf",
                "pdf_path": req.pdf_path,
                "extraction_type": req.extraction_type,
                "user_instructions": req.user_instructions,
                "execution_time": execution_time,
                "authenticated_user": user_id is not None
            },
            user_id=user_id
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
        
        loader_agent = session["agent"]

        # Check if this is a fallback session (serialization failed)
        if isinstance(loader_agent, dict) and loader_agent.get("_fallback_mode"):
            return DataResponse(
                success=False,
                message="Cannot retrieve artifacts",
                error="Session is in fallback mode due to serialization issues. Original data cannot be retrieved."
            )

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

@agent.on_rest_post("/get-ai-message", SessionRequest, GenericResponse)
async def get_ai_message(ctx: Context, req: SessionRequest) -> GenericResponse:
    """Get AI message from session"""
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
        
        loader_agent = session["agent"]

        # Check if this is a fallback session (serialization failed)
        if isinstance(loader_agent, dict) and loader_agent.get("_fallback_mode"):
            return GenericResponse(
                success=False,
                message="Cannot retrieve AI message",
                error="Session is in fallback mode due to serialization issues. AI message cannot be retrieved."
            )

        ai_message = loader_agent.get_ai_message()

        # Convert AIMessage to serializable format
        if hasattr(ai_message, 'content'):
            serializable_message = {
                "type": getattr(ai_message, 'type', 'ai'),
                "content": ai_message.content,
                "id": getattr(ai_message, 'id', None)
            }
        else:
            serializable_message = str(ai_message) if ai_message else "No AI message available"
        
        return GenericResponse(
            success=True,
            message="AI message retrieved successfully",
            data=serializable_message
        )
        
    except Exception as e:
        return GenericResponse(
            success=False,
            message="Failed to retrieve AI message",
            error=str(e)
        )

@agent.on_rest_post("/get-tool-calls", SessionRequest, GenericResponse)
async def get_tool_calls(ctx: Context, req: SessionRequest) -> GenericResponse:
    """Get tool calls from session"""
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
        
        loader_agent = session["agent"]

        # Check if this is a fallback session (serialization failed)
        if isinstance(loader_agent, dict) and loader_agent.get("_fallback_mode"):
            return GenericResponse(
                success=False,
                message="Cannot retrieve tool calls",
                error="Session is in fallback mode due to serialization issues. Tool calls cannot be retrieved."
            )

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

@agent.on_rest_post("/get-internal-messages", SessionRequest, GenericResponse)
async def get_internal_messages(ctx: Context, req: SessionRequest) -> GenericResponse:
    """Get internal messages from session"""
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
        
        loader_agent = session["agent"]

        # Check if this is a fallback session (serialization failed)
        if isinstance(loader_agent, dict) and loader_agent.get("_fallback_mode"):
            return GenericResponse(
                success=False,
                message="Cannot retrieve internal messages",
                error="Session is in fallback mode due to serialization issues. Internal messages cannot be retrieved."
            )

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

@agent.on_rest_post("/get-full-response", SessionRequest, GenericResponse)
async def get_full_response(ctx: Context, req: SessionRequest) -> GenericResponse:
    """Get complete agent response from session"""
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
        
        loader_agent = session["agent"]

        # Check if this is a fallback session (serialization failed)
        if isinstance(loader_agent, dict) and loader_agent.get("_fallback_mode"):
            return GenericResponse(
                success=False,
                message="Cannot retrieve full response",
                error="Session is in fallback mode due to serialization issues. Full response cannot be retrieved."
            )

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
# NOTE: GET endpoints with parameters removed due to uAgent framework limitations
# Use POST endpoints instead (/get-artifacts, /get-ai-message, etc.)
# ============================================================================

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
    print("   POST http://127.0.0.1:8005/get-ai-message")
    print("   POST http://127.0.0.1:8005/get-tool-calls")
    print("   POST http://127.0.0.1:8005/get-internal-messages")
    print("   POST http://127.0.0.1:8005/get-full-response")
    print("   GET  http://127.0.0.1:8005/supported-formats")
    print("   POST http://127.0.0.1:8005/delete-session")
    print("🚀 Agent starting...")
    agent.run()
