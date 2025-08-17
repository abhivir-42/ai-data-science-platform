"""
uAgent-based REST API that wraps the LangChain DataCleaningAgent.

- Exposes REST endpoints via uAgents (@on_rest_get/@on_rest_post)
  - GET  /health
  - POST /clean-csv
  - POST /clean-data

- Internally invokes the LangChain-based DataCleaningAgent defined in
  langchain_data_cleaning_agent.py

Run:
  python data_cleaning_rest_agent.py
"""

from __future__ import annotations

import base64
import io
import os
import sys
from typing import Any, Dict, List, Optional

import pandas as pd

# Load environment variables from project root if available
try:
    from dotenv import load_dotenv
except ImportError:  # optional dependency; will work if env already set
    load_dotenv = None  # type: ignore


def _ensure_project_root_on_path() -> None:
    """Add project root and backend to sys.path so backend imports resolve."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    backend_dir = os.path.join(project_root, "backend")
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)


def _load_environment() -> None:
    if load_dotenv is None:
        return
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for path in (
        os.path.join(current_dir, ".env"),
        os.path.join(current_dir, "..", "..", ".env"),
    ):
        if os.path.exists(path):
            load_dotenv(dotenv_path=path)
            break


_ensure_project_root_on_path()
_load_environment()


# Now imports that depend on project root being on sys.path
from uagents import Agent, Context, Model  # type: ignore
from uagents.setup import fund_agent_if_low  # type: ignore
from langchain_openai import ChatOpenAI  # type: ignore
from langchain_data_cleaning_agent import DataCleaningAgent  # type: ignore


def _create_data_cleaning_agent() -> DataCleaningAgent:
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to a .env at repo root or export it in your shell."
        )
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_key, temperature=0)
    return DataCleaningAgent(model=llm, log=False)


def _invoke_cleaning(df: pd.DataFrame, instructions: Optional[str], max_retries: int) -> Dict[str, Any]:
    agent = _create_data_cleaning_agent()
    agent.invoke_agent(data_raw=df, user_instructions=instructions, max_retries=max_retries)
    cleaned_df = agent.get_data_cleaned()
    if cleaned_df is None:
        return {
            "success": False,
            "error": "Cleaning failed with no output",
        }
    
    # Convert DataFrame to dict with proper handling of pandas types
    import numpy as np
    
    # Convert to dict first, then handle pandas-specific types
    records = cleaned_df.to_dict(orient="records")
    
    # Replace pandas-specific types with JSON-serializable equivalents
    def clean_record(record):
        cleaned = {}
        for key, value in record.items():
            # Handle NaN and NaT values
            if pd.isna(value):
                cleaned[key] = None
            # Handle pandas Timestamp objects
            elif hasattr(value, 'isoformat'):  # datetime-like objects
                cleaned[key] = value.isoformat()
            # Handle numpy numeric types
            elif isinstance(value, (np.integer, np.floating)):
                if np.isnan(value):
                    cleaned[key] = None
                else:
                    cleaned[key] = value.item()  # Convert to Python native type
            # Handle other numpy types
            elif hasattr(value, 'item') and hasattr(value, 'dtype'):
                cleaned[key] = value.item()  # Convert numpy types to Python native
            else:
                cleaned[key] = value
        return cleaned
    
    cleaned_records = [clean_record(record) for record in records]
    
    steps = agent.get_recommended_cleaning_steps() or ""
    return {
        "success": True,
        "message": "Data cleaning completed successfully",
        "original_shape": [int(df.shape[0]), int(df.shape[1])],
        "cleaned_shape": [int(cleaned_df.shape[0]), int(cleaned_df.shape[1])],
        "cleaning_steps": steps,
        "cleaned_data": {
            "records": cleaned_records,
            "columns": list(map(str, cleaned_df.columns.tolist())),
        },
        "error": None,
    }


# --------- uAgents models ---------

class HealthResponse(Model):
    status: str
    agent: str


class CleanCsvRequest(Model):
    filename: Optional[str]
    file_content: str  # base64-encoded CSV
    user_instructions: Optional[str] = None
    max_retries: Optional[int] = 3


class CleanDataRequest(Model):
    data: Dict[str, List[Any]]
    user_instructions: Optional[str] = None
    max_retries: Optional[int] = 3


class CleanResponse(Model):
    success: bool
    message: Optional[str] = None
    original_shape: Optional[List[int]] = None
    cleaned_shape: Optional[List[int]] = None
    cleaning_steps: Optional[str] = None
    cleaned_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# --------- uAgent definition ---------

agent = Agent(
    name="data_cleaning_agent",
    port=8003,
    seed="data_cleaning_agent_secret_seed",
    endpoint=["http://127.0.0.1:8003/submit"],
)

# Fund if required (no-op locally, safe to call)
fund_agent_if_low(agent.wallet.address())


@agent.on_rest_get("/health", HealthResponse)
async def handle_health(ctx: Context) -> Dict[str, Any]:
    ctx.logger.info("Health check")
    return {"status": "healthy", "agent": "data_cleaning_rest_uagent"}


@agent.on_rest_post("/clean-csv", CleanCsvRequest, CleanResponse)
async def handle_clean_csv(ctx: Context, req: CleanCsvRequest) -> CleanResponse:
    try:
        decoded = base64.b64decode(req.file_content)
        csv_text = decoded.decode("utf-8", errors="replace")
        original_df = pd.read_csv(io.StringIO(csv_text))
    except Exception as e:
        return CleanResponse(success=False, error=f"Invalid CSV data: {str(e)}")

    if original_df.shape[0] == 0 or original_df.shape[1] == 0:
        return CleanResponse(success=False, error="CSV contains no data rows or columns")

    max_retries = req.max_retries if req.max_retries is not None else 3
    result = _invoke_cleaning(original_df, req.user_instructions, int(max_retries))
    return CleanResponse(**result)


@agent.on_rest_post("/clean-data", CleanDataRequest, CleanResponse)
async def handle_clean_data(ctx: Context, req: CleanDataRequest) -> CleanResponse:
    try:
        original_df = pd.DataFrame.from_dict(req.data)
    except Exception as e:
        return CleanResponse(success=False, error=f"Invalid data format: {str(e)}")

    if original_df.shape[0] == 0 or original_df.shape[1] == 0:
        return CleanResponse(success=False, error="Data contains no rows or columns")

    max_retries = req.max_retries if req.max_retries is not None else 3
    result = _invoke_cleaning(original_df, req.user_instructions, int(max_retries))
    return CleanResponse(**result)


if __name__ == "__main__":
    print("🧹 Starting Data Cleaning uAgent (REST)...")
    print("📡 Available endpoints:")
    print("   GET  http://127.0.0.1:8003/health")
    print("   POST http://127.0.0.1:8003/clean-csv")
    print("   POST http://127.0.0.1:8003/clean-data")
    print("🚀 Agent starting...")
    agent.run()


