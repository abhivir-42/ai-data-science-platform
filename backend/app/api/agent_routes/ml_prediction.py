"""ML Prediction agent endpoints - replaces the uAgent on port 8009."""

import time
from typing import Optional, Dict, Any, List
from fastapi import APIRouter
from pydantic import BaseModel
from loguru import logger

from app.agents.ml_prediction_agent import MLPredictionAgent
from app.api.agent_routes.common import (
    get_llm, decode_csv_content, dataframe_to_json_safe, make_json_serializable,
    session_service, SessionRequest, SessionResponse, DataResponse, GenericResponse,
)

router = APIRouter(prefix="/prediction", tags=["ml-prediction"])


class PredictSingleRequest(BaseModel):
    input_data: Dict[str, Any]
    model_session_id: Optional[str] = None
    model_path: Optional[str] = None


class PredictBatchRequest(BaseModel):
    model_session_id: Optional[str] = None
    model_path: Optional[str] = None
    filename: Optional[str] = None
    file_content: Optional[str] = None  # base64


class AnalyzeModelRequest(BaseModel):
    query: str
    model_session_id: Optional[str] = None
    model_path: Optional[str] = None


class PredictionResponse(BaseModel):
    success: bool
    message: str
    session_id: str = ""
    prediction: Optional[Any] = None
    prediction_probability: Optional[Any] = None
    confidence: Optional[float] = None
    execution_time_seconds: Optional[float] = None
    error: Optional[str] = None


class ModelAnalysisResponse(BaseModel):
    success: bool
    message: str
    session_id: str = ""
    analysis: Optional[str] = None
    model_info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


def _create_agent():
    llm = get_llm()
    return MLPredictionAgent(
        model=llm,
        log=True,
        log_path="./temp",
        model_directory="./temp/models",
        overwrite=True,
        human_in_the_loop=False,
        bypass_recommended_steps=False,
        bypass_explain_code=False,
    )


async def _get_model_info_from_session(model_session_id: str) -> Dict[str, Any]:
    """Extract model info from a training session."""
    session = await session_service.get_session(model_session_id)
    if not session or "agent" not in session:
        raise ValueError(f"Training session not found: {model_session_id}")

    training_agent = session["agent"]
    model_path = training_agent.get_model_path() if hasattr(training_agent, "get_model_path") else None
    best_model_id = training_agent.get_best_model_id() if hasattr(training_agent, "get_best_model_id") else None

    return {
        "model_path": model_path,
        "best_model_id": best_model_id,
        "training_agent": training_agent,
    }


@router.post("/predict-single", response_model=PredictionResponse)
async def predict_single(request: PredictSingleRequest):
    """Make a single prediction."""
    start = time.time()
    try:
        agent = _create_agent()

        # Load model from training session or path
        if request.model_session_id:
            model_info = await _get_model_info_from_session(request.model_session_id)
            if model_info.get("model_path"):
                agent.load_model(model_info["model_path"])
        elif request.model_path:
            agent.load_model(request.model_path)
        else:
            return PredictionResponse(success=False, message="Provide model_session_id or model_path", error="No model source")

        result = agent.predict_single(request.input_data) if hasattr(agent, "predict_single") else None

        execution_time = time.time() - start
        metadata = {
            "operation": "predict_single",
            "model_session_id": request.model_session_id,
            "model_path": request.model_path,
            "prediction_result": make_json_serializable(result),
            "execution_time": execution_time,
        }
        session_id = await session_service.create_session(agent, "prediction", metadata)

        prediction = make_json_serializable(result)
        return PredictionResponse(
            success=True, message="Prediction completed", session_id=session_id,
            prediction=prediction, execution_time_seconds=execution_time,
        )
    except Exception as e:
        logger.error(f"predict_single failed: {e}")
        return PredictionResponse(success=False, message=str(e), error=str(e))


@router.post("/predict-batch", response_model=SessionResponse)
async def predict_batch(request: PredictBatchRequest):
    """Make batch predictions."""
    start = time.time()
    try:
        agent = _create_agent()

        if request.model_session_id:
            model_info = await _get_model_info_from_session(request.model_session_id)
            if model_info.get("model_path"):
                agent.load_model(model_info["model_path"])
        elif request.model_path:
            agent.load_model(request.model_path)
        else:
            return SessionResponse(success=False, message="Provide model_session_id or model_path", session_id="", error="No model source")

        # Get prediction data
        if request.file_content:
            df = decode_csv_content(request.file_content, request.filename or "batch.csv")
            result = agent.predict_batch(df) if hasattr(agent, "predict_batch") else None
        else:
            return SessionResponse(success=False, message="Provide file_content for batch prediction", session_id="", error="No data")

        execution_time = time.time() - start
        metadata = {
            "operation": "predict_batch",
            "model_session_id": request.model_session_id,
            "batch_results": make_json_serializable(result),
            "execution_time": execution_time,
        }
        session_id = await session_service.create_session(agent, "prediction", metadata)

        return SessionResponse(success=True, message="Batch prediction completed", session_id=session_id, execution_time_seconds=execution_time)
    except Exception as e:
        logger.error(f"predict_batch failed: {e}")
        return SessionResponse(success=False, message=str(e), session_id="", error=str(e))


@router.post("/analyze-model", response_model=ModelAnalysisResponse)
async def analyze_model(request: AnalyzeModelRequest):
    """Analyze a trained model."""
    start = time.time()
    try:
        agent = _create_agent()

        if request.model_session_id:
            model_info = await _get_model_info_from_session(request.model_session_id)
            if model_info.get("model_path"):
                agent.load_model(model_info["model_path"])
        elif request.model_path:
            agent.load_model(request.model_path)
        else:
            return ModelAnalysisResponse(success=False, message="Provide model_session_id or model_path", error="No model source")

        result = agent.analyze_model(request.query) if hasattr(agent, "analyze_model") else None

        execution_time = time.time() - start
        metadata = {
            "operation": "analyze_model",
            "model_session_id": request.model_session_id,
            "analysis_result": make_json_serializable(result),
            "execution_time": execution_time,
        }
        session_id = await session_service.create_session(agent, "prediction", metadata)

        analysis_text = None
        model_info_data = None
        if isinstance(result, dict):
            analysis_text = result.get("answer") or result.get("analysis")
            model_info_data = result.get("model_info")
        elif isinstance(result, str):
            analysis_text = result

        return ModelAnalysisResponse(
            success=True, message="Model analysis completed", session_id=session_id,
            analysis=analysis_text, model_info=model_info_data,
        )
    except Exception as e:
        logger.error(f"analyze_model failed: {e}")
        return ModelAnalysisResponse(success=False, message=str(e), error=str(e))


@router.post("/get-prediction-results", response_model=DataResponse)
async def get_prediction_results(request: SessionRequest):
    """Get prediction results from session."""
    try:
        session = await session_service.get_session(request.session_id)
        if not session:
            return DataResponse(success=False, message="Session not found", error="Session not found")

        metadata = session.get("metadata", {})
        prediction = metadata.get("prediction_result") or metadata.get("batch_results")
        return DataResponse(success=True, message="Prediction results retrieved", data=make_json_serializable(prediction))
    except Exception as e:
        logger.error(f"get_prediction_results failed: {e}")
        return DataResponse(success=False, message=str(e), error=str(e))


@router.post("/get-model-analysis", response_model=ModelAnalysisResponse)
async def get_model_analysis(request: SessionRequest):
    """Get model analysis from session."""
    try:
        session = await session_service.get_session(request.session_id)
        if not session:
            return ModelAnalysisResponse(success=False, message="Session not found", error="Session not found")

        metadata = session.get("metadata", {})
        result = metadata.get("analysis_result", {})
        analysis_text = None
        model_info_data = None
        if isinstance(result, dict):
            analysis_text = result.get("answer") or result.get("analysis")
            model_info_data = result.get("model_info")

        return ModelAnalysisResponse(success=True, message="Analysis retrieved", analysis=analysis_text, model_info=model_info_data)
    except Exception as e:
        return ModelAnalysisResponse(success=False, message=str(e), error=str(e))


@router.post("/get-logs", response_model=GenericResponse)
async def get_logs(request: SessionRequest):
    try:
        session = await session_service.get_session(request.session_id)
        if not session or "agent" not in session:
            return GenericResponse(success=False, message="Session not found", error="Session not found")
        agent = session["agent"]
        logs = agent.get_logs() if hasattr(agent, "get_logs") else []
        return GenericResponse(success=True, message="Logs retrieved", data=make_json_serializable(logs))
    except Exception as e:
        return GenericResponse(success=False, message=str(e), error=str(e))


@router.post("/delete-session", response_model=GenericResponse)
async def delete_session(request: SessionRequest):
    try:
        await session_service.delete_session(request.session_id)
        return GenericResponse(success=True, message="Session deleted")
    except Exception as e:
        return GenericResponse(success=False, message=str(e), error=str(e))


@router.get("/health")
async def health():
    return {"status": "healthy", "agent": "ml_prediction"}
