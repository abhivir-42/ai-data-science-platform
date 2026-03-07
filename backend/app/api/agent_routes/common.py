"""Common utilities shared across agent route modules."""

import time
import base64
import io
import math
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from loguru import logger

from app.core.config import settings
from app.services.session_service import SessionService

session_service = SessionService()


def get_llm():
    """Create a ChatOpenAI LLM instance."""
    return ChatOpenAI(
        model="gpt-4o-mini",
        api_key=settings.OPENAI_API_KEY,
    )


def decode_csv_content(file_content_b64: str, filename: str = "upload.csv") -> pd.DataFrame:
    """Decode base64-encoded CSV content into a DataFrame."""
    try:
        raw_bytes = base64.b64decode(file_content_b64)
        return pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as e:
        raise ValueError(f"Failed to decode CSV content from '{filename}': {e}")


def dataframe_to_json_safe(df: pd.DataFrame) -> Dict[str, Any]:
    """Convert a DataFrame to a JSON-safe dict with records, columns, and shape."""
    if df is None:
        return {"records": [], "columns": [], "shape": [0, 0]}

    # Replace NaN/Inf with None for JSON serialization
    df_clean = df.copy()
    for col in df_clean.columns:
        if df_clean[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            df_clean[col] = df_clean[col].apply(
                lambda x: None if (isinstance(x, float) and (math.isnan(x) or math.isinf(x))) else x
            )
        df_clean[col] = df_clean[col].apply(
            lambda x: None if isinstance(x, float) and (math.isnan(x) or math.isinf(x)) else x
        )

    records = df_clean.head(100).to_dict(orient="records")
    return {
        "records": records,
        "columns": list(df_clean.columns),
        "shape": list(df_clean.shape),
    }


def make_json_serializable(obj: Any) -> Any:
    """Recursively convert an object to JSON-serializable form."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return dataframe_to_json_safe(obj)
    if isinstance(obj, pd.Series):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    if hasattr(obj, "__dict__"):
        return make_json_serializable(obj.__dict__)
    return str(obj)


# Common Pydantic models
class SessionRequest(BaseModel):
    session_id: str
    user_id: Optional[str] = None


class SessionResponse(BaseModel):
    success: bool
    message: str
    session_id: str
    execution_time_seconds: Optional[float] = None
    error: Optional[str] = None


class DataResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None
    original_shape: Optional[List[int]] = None
    processed_shape: Optional[List[int]] = None
    error: Optional[str] = None


class CodeResponse(BaseModel):
    success: bool
    message: str
    generated_code: Optional[str] = None
    code_explanation: Optional[str] = None
    error: Optional[str] = None


class GenericResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None
    error: Optional[str] = None


class ChartResponse(BaseModel):
    success: bool
    message: str
    plotly_chart: Optional[Any] = None
    chart_type: Optional[str] = None
    error: Optional[str] = None
