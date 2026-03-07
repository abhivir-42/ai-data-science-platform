"""
Internal agent client for workflow orchestration.

This replaces the old HTTP-based uAgent client that routed to 6 separate
microservices on ports 8004-8009. Now calls agents directly in-process.
"""

import time
from typing import Optional, Dict, Any

import pandas as pd
from loguru import logger

from app.agents.data_loader_tools_agent import DataLoaderToolsAgent
from app.agents.data_cleaning_agent import DataCleaningAgent
from app.agents.data_visualisation_agent import DataVisualisationAgent
from app.agents.feature_engineering_agent import FeatureEngineeringAgent
from app.agents.ml_agents.h2o_ml_agent import H2OMLAgent
from app.agents.ml_prediction_agent import MLPredictionAgent
from app.api.agent_routes.common import (
    get_llm, decode_csv_content, dataframe_to_json_safe, make_json_serializable,
    session_service,
)


class UAgentClient:
    """Internal client that calls agents directly (no HTTP)."""

    def __init__(self, user_id: Optional[str] = None):
        self.user_id = user_id

    def _get_df_from_session(self, session: dict) -> Optional[pd.DataFrame]:
        """Extract a DataFrame from a session's agent."""
        agent = session.get("agent")
        if agent is None:
            return None
        for method in ["get_data_engineered", "get_data_cleaned", "get_data_raw"]:
            if hasattr(agent, method):
                df = getattr(agent, method)()
                if df is not None and isinstance(df, pd.DataFrame):
                    return df
        if hasattr(agent, "get_artifacts"):
            df = agent.get_artifacts(as_dataframe=True)
            if df is not None and isinstance(df, pd.DataFrame):
                return df
        return None

    async def load_file(self, agent_type: str, filename: str, file_content: str, user_instructions: str = "") -> Dict[str, Any]:
        start = time.time()
        try:
            df = decode_csv_content(file_content, filename)
            llm = get_llm()
            agent = DataLoaderToolsAgent(model=llm, create_react_agent_kwargs={}, invoke_react_agent_kwargs={}, checkpointer=None)
            agent.invoke_agent(user_instructions=user_instructions or "Load and analyze the uploaded file", data_raw=df)

            session_id = await session_service.create_session(agent, "loading", {
                "operation": "load_file", "filename": filename, "execution_time": time.time() - start
            }, self.user_id)

            return {"success": True, "session_id": session_id, "message": "Data loaded", "execution_time_seconds": time.time() - start}
        except Exception as e:
            logger.error(f"load_file failed: {e}")
            return {"success": False, "error": str(e), "session_id": ""}

    async def clean_data_from_session(self, session_id: str, user_instructions: str = "") -> Dict[str, Any]:
        start = time.time()
        try:
            session = await session_service.get_session(session_id)
            if not session:
                return {"success": False, "error": "Session not found", "session_id": ""}

            df = self._get_df_from_session(session)
            if df is None:
                return {"success": False, "error": "No data in session", "session_id": ""}

            llm = get_llm()
            agent = DataCleaningAgent(model=llm, log=True, log_path="./temp", overwrite=True, human_in_the_loop=False, bypass_recommended_steps=False, bypass_explain_code=False, n_samples=30)
            agent.invoke_agent(data_raw=df, user_instructions=user_instructions or "Clean the data", max_retries=3)

            new_session_id = await session_service.create_session(agent, "cleaning", {
                "operation": "clean_from_session", "source_session_id": session_id, "execution_time": time.time() - start
            }, self.user_id)

            return {"success": True, "session_id": new_session_id, "message": "Data cleaned", "execution_time_seconds": time.time() - start}
        except Exception as e:
            logger.error(f"clean_data_from_session failed: {e}")
            return {"success": False, "error": str(e), "session_id": ""}

    async def clean_csv_data(self, filename: str, file_content: str, user_instructions: str = "") -> Dict[str, Any]:
        start = time.time()
        try:
            df = decode_csv_content(file_content, filename)
            llm = get_llm()
            agent = DataCleaningAgent(model=llm, log=True, log_path="./temp", overwrite=True, human_in_the_loop=False, bypass_recommended_steps=False, bypass_explain_code=False, n_samples=30)
            agent.invoke_agent(data_raw=df, user_instructions=user_instructions or "Clean the data", max_retries=3)

            session_id = await session_service.create_session(agent, "cleaning", {
                "operation": "clean_csv", "filename": filename, "execution_time": time.time() - start
            }, self.user_id)

            return {"success": True, "session_id": session_id, "message": "CSV cleaned", "execution_time_seconds": time.time() - start}
        except Exception as e:
            logger.error(f"clean_csv_data failed: {e}")
            return {"success": False, "error": str(e), "session_id": ""}

    async def create_chart_from_session(self, session_id: str, user_instructions: str = "") -> Dict[str, Any]:
        start = time.time()
        try:
            session = await session_service.get_session(session_id)
            if not session:
                return {"success": False, "error": "Session not found", "session_id": ""}

            df = self._get_df_from_session(session)
            if df is None:
                return {"success": False, "error": "No data in session", "session_id": ""}

            llm = get_llm()
            agent = DataVisualisationAgent(model=llm, log=True, log_path="./temp", overwrite=True, human_in_the_loop=False, bypass_recommended_steps=False, bypass_explain_code=False, n_samples=30)
            agent.invoke_agent(data_raw=df, user_instructions=user_instructions or "Create visualizations", max_retries=3)

            new_session_id = await session_service.create_session(agent, "visualization", {
                "operation": "create_chart_from_session", "source_session_id": session_id, "execution_time": time.time() - start
            }, self.user_id)

            return {"success": True, "session_id": new_session_id, "message": "Chart created", "execution_time_seconds": time.time() - start}
        except Exception as e:
            logger.error(f"create_chart_from_session failed: {e}")
            return {"success": False, "error": str(e), "session_id": ""}

    async def create_chart_csv(self, filename: str, file_content: str, user_instructions: str = "") -> Dict[str, Any]:
        start = time.time()
        try:
            df = decode_csv_content(file_content, filename)
            llm = get_llm()
            agent = DataVisualisationAgent(model=llm, log=True, log_path="./temp", overwrite=True, human_in_the_loop=False, bypass_recommended_steps=False, bypass_explain_code=False, n_samples=30)
            agent.invoke_agent(data_raw=df, user_instructions=user_instructions or "Create visualizations", max_retries=3)

            session_id = await session_service.create_session(agent, "visualization", {
                "operation": "create_chart_csv", "filename": filename, "execution_time": time.time() - start
            }, self.user_id)

            return {"success": True, "session_id": session_id, "message": "Chart created", "execution_time_seconds": time.time() - start}
        except Exception as e:
            logger.error(f"create_chart_csv failed: {e}")
            return {"success": False, "error": str(e), "session_id": ""}

    async def engineer_features_from_session(self, session_id: str, target_variable: str, user_instructions: str = "") -> Dict[str, Any]:
        start = time.time()
        try:
            session = await session_service.get_session(session_id)
            if not session:
                return {"success": False, "error": "Session not found", "session_id": ""}

            df = self._get_df_from_session(session)
            if df is None:
                return {"success": False, "error": "No data in session", "session_id": ""}

            llm = get_llm()
            agent = FeatureEngineeringAgent(model=llm, log=True, log_path="./temp", overwrite=True, human_in_the_loop=False, bypass_recommended_steps=False, bypass_explain_code=False, n_samples=30)
            agent.invoke_agent(data_raw=df, user_instructions=user_instructions or "Engineer features", target_variable=target_variable, max_retries=3)

            new_session_id = await session_service.create_session(agent, "engineering", {
                "operation": "engineer_from_session", "source_session_id": session_id, "target_variable": target_variable, "execution_time": time.time() - start
            }, self.user_id)

            return {"success": True, "session_id": new_session_id, "message": "Features engineered", "execution_time_seconds": time.time() - start}
        except Exception as e:
            logger.error(f"engineer_features_from_session failed: {e}")
            return {"success": False, "error": str(e), "session_id": ""}

    async def engineer_features_csv(self, filename: str, file_content: str, target_variable: str, user_instructions: str = "") -> Dict[str, Any]:
        start = time.time()
        try:
            df = decode_csv_content(file_content, filename)
            llm = get_llm()
            agent = FeatureEngineeringAgent(model=llm, log=True, log_path="./temp", overwrite=True, human_in_the_loop=False, bypass_recommended_steps=False, bypass_explain_code=False, n_samples=30)
            agent.invoke_agent(data_raw=df, user_instructions=user_instructions or "Engineer features", target_variable=target_variable, max_retries=3)

            session_id = await session_service.create_session(agent, "engineering", {
                "operation": "engineer_csv", "filename": filename, "target_variable": target_variable, "execution_time": time.time() - start
            }, self.user_id)

            return {"success": True, "session_id": session_id, "message": "Features engineered", "execution_time_seconds": time.time() - start}
        except Exception as e:
            logger.error(f"engineer_features_csv failed: {e}")
            return {"success": False, "error": str(e), "session_id": ""}

    async def train_model_from_session(self, session_id: str, target_variable: str, user_instructions: str = "", max_runtime_secs: int = 300) -> Dict[str, Any]:
        start = time.time()
        try:
            session = await session_service.get_session(session_id)
            if not session:
                return {"success": False, "error": "Session not found", "session_id": ""}

            df = self._get_df_from_session(session)
            if df is None:
                return {"success": False, "error": "No data in session", "session_id": ""}

            llm = get_llm()
            agent = H2OMLAgent(model=llm, log=True, log_path="./temp", model_directory="./temp/models", overwrite=True, human_in_the_loop=False, bypass_recommended_steps=False, bypass_explain_code=False)
            agent.invoke_agent(data_raw=df, user_instructions=user_instructions or "Train models", target_variable=target_variable, max_retries=3)

            new_session_id = await session_service.create_session(agent, "training", {
                "operation": "train_from_session", "source_session_id": session_id, "target_variable": target_variable, "execution_time": time.time() - start
            }, self.user_id)

            return {"success": True, "session_id": new_session_id, "message": "Model trained", "execution_time_seconds": time.time() - start}
        except Exception as e:
            logger.error(f"train_model_from_session failed: {e}")
            return {"success": False, "error": str(e), "session_id": ""}

    async def train_model_csv(self, filename: str, file_content: str, target_variable: str, user_instructions: str = "", max_runtime_secs: int = 300) -> Dict[str, Any]:
        start = time.time()
        try:
            df = decode_csv_content(file_content, filename)
            llm = get_llm()
            agent = H2OMLAgent(model=llm, log=True, log_path="./temp", model_directory="./temp/models", overwrite=True, human_in_the_loop=False, bypass_recommended_steps=False, bypass_explain_code=False)
            agent.invoke_agent(data_raw=df, user_instructions=user_instructions or "Train models", target_variable=target_variable, max_retries=3)

            session_id = await session_service.create_session(agent, "training", {
                "operation": "train_csv", "filename": filename, "target_variable": target_variable, "execution_time": time.time() - start
            }, self.user_id)

            return {"success": True, "session_id": session_id, "message": "Model trained", "execution_time_seconds": time.time() - start}
        except Exception as e:
            logger.error(f"train_model_csv failed: {e}")
            return {"success": False, "error": str(e), "session_id": ""}

    async def predict_batch(self, model_session_id: str, filename: str = None, file_content: str = None) -> Dict[str, Any]:
        start = time.time()
        try:
            session = await session_service.get_session(model_session_id)
            if not session or "agent" not in session:
                return {"success": False, "error": "Model session not found", "session_id": ""}

            training_agent = session["agent"]
            model_path = training_agent.get_model_path() if hasattr(training_agent, "get_model_path") else None

            llm = get_llm()
            agent = MLPredictionAgent(model=llm, log=True, log_path="./temp", model_directory="./temp/models", overwrite=True, human_in_the_loop=False, bypass_recommended_steps=False, bypass_explain_code=False)

            if model_path:
                agent.load_model(model_path)

            result = None
            if file_content:
                df = decode_csv_content(file_content, filename or "batch.csv")
                result = agent.predict_batch(df) if hasattr(agent, "predict_batch") else None

            new_session_id = await session_service.create_session(agent, "prediction", {
                "operation": "predict_batch", "model_session_id": model_session_id, "batch_results": make_json_serializable(result), "execution_time": time.time() - start
            }, self.user_id)

            return {"success": True, "session_id": new_session_id, "message": "Predictions made", "execution_time_seconds": time.time() - start}
        except Exception as e:
            logger.error(f"predict_batch failed: {e}")
            return {"success": False, "error": str(e), "session_id": ""}

    # Session data retrieval methods used by workflow_execution
    async def get_session_data(self, agent_type: str, session_id: str) -> Optional[Dict[str, Any]]:
        try:
            session = await session_service.get_session(session_id)
            if not session:
                return None
            df = self._get_df_from_session(session)
            return dataframe_to_json_safe(df) if df is not None else None
        except Exception as e:
            logger.error(f"get_session_data failed: {e}")
            return None

    async def get_session_code(self, agent_type: str, session_id: str) -> Optional[Dict[str, Any]]:
        try:
            session = await session_service.get_session(session_id)
            if not session or "agent" not in session:
                return None
            agent = session["agent"]
            code = None
            for method in ["get_data_cleaner_function", "get_data_visualization_function", "get_feature_engineer_function", "get_h2o_training_function"]:
                if hasattr(agent, method):
                    code = getattr(agent, method)()
                    if code:
                        break
            return {"success": True, "generated_code": code} if code else None
        except Exception as e:
            logger.error(f"get_session_code failed: {e}")
            return None

    async def get_session_chart(self, agent_type: str, session_id: str) -> Optional[Dict[str, Any]]:
        try:
            session = await session_service.get_session(session_id)
            if not session or "agent" not in session:
                return None
            agent = session["agent"]
            response = agent.response if hasattr(agent, "response") else None
            if response and isinstance(response, dict):
                plotly_graph = response.get("plotly_graph")
                return {"success": True, "plotly_chart": make_json_serializable(plotly_graph)} if plotly_graph else None
            return None
        except Exception as e:
            logger.error(f"get_session_chart failed: {e}")
            return None

    async def get_session_leaderboard(self, agent_type: str, session_id: str) -> Optional[Dict[str, Any]]:
        try:
            session = await session_service.get_session(session_id)
            if not session or "agent" not in session:
                return None
            agent = session["agent"]
            lb = agent.get_leaderboard() if hasattr(agent, "get_leaderboard") else None
            if lb is not None:
                data = lb.to_dict(orient="records") if hasattr(lb, "to_dict") else make_json_serializable(lb)
                return {"success": True, "leaderboard": data}
            return None
        except Exception as e:
            logger.error(f"get_session_leaderboard failed: {e}")
            return None

    async def get_model_path(self, agent_type: str, session_id: str) -> Optional[str]:
        try:
            session = await session_service.get_session(session_id)
            if not session or "agent" not in session:
                return None
            agent = session["agent"]
            return agent.get_model_path() if hasattr(agent, "get_model_path") else None
        except Exception as e:
            logger.error(f"get_model_path failed: {e}")
            return None

    async def check_all_agents_health(self) -> Dict[str, Any]:
        """All agents run in-process now, so health is always OK."""
        return {
            "loading": {"status": "ready", "agent_status": "ready"},
            "cleaning": {"status": "ready", "agent_status": "ready"},
            "visualization": {"status": "ready", "agent_status": "ready"},
            "engineering": {"status": "ready", "agent_status": "ready"},
            "training": {"status": "ready", "agent_status": "ready"},
            "prediction": {"status": "ready", "agent_status": "ready"},
        }
