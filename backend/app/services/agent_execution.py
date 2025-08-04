"""
Agent Execution Service

Handles the execution of different types of AI agents with their specific interfaces
and parameter requirements.
"""

import asyncio
import traceback
from typing import Dict, Any, Optional, Union
from loguru import logger
import pandas as pd

from app.core.agent_registry import agent_registry
from app.core.config import settings


class AgentExecutionError(Exception):
    """Custom exception for agent execution errors"""
    pass


class AgentExecutionService:
    """Service for executing AI agents with proper parameter handling"""
    
    def __init__(self):
        self.registry = agent_registry
    
    async def execute_agent_async(
        self, 
        agent_id: str, 
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute an agent asynchronously
        
        Args:
            agent_id: ID of the agent to execute
            parameters: Parameters for agent execution
            
        Returns:
            Dict containing execution results and metadata
        """
        try:
            # Get agent instance
            agent_instance = self._create_agent_instance(agent_id)
            
            # Execute based on agent type
            result = await self._execute_agent_by_type(agent_instance, agent_id, parameters, async_mode=True)
            
            return {
                "status": "completed",
                "agent_id": agent_id,
                "parameters": parameters,
                "result": result,
                "message": f"Agent {agent_id} executed successfully"
            }
            
        except Exception as e:
            logger.error(f"Async execution failed for agent {agent_id}: {e}")
            logger.error(traceback.format_exc())
            return {
                "status": "failed",
                "agent_id": agent_id,
                "parameters": parameters,
                "error": str(e),
                "message": f"Agent {agent_id} execution failed: {str(e)}"
            }
    
    def execute_agent_sync(
        self, 
        agent_id: str, 
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute an agent synchronously
        
        Args:
            agent_id: ID of the agent to execute
            parameters: Parameters for agent execution
            
        Returns:
            Dict containing execution results and metadata
        """
        try:
            # Get agent instance
            agent_instance = self._create_agent_instance(agent_id)
            
            # Execute based on agent type
            result = self._execute_agent_by_type(agent_instance, agent_id, parameters, async_mode=False)
            
            return {
                "status": "completed",
                "agent_id": agent_id,
                "parameters": parameters,
                "result": result,
                "message": f"Agent {agent_id} executed successfully"
            }
            
        except Exception as e:
            logger.error(f"Sync execution failed for agent {agent_id}: {e}")
            logger.error(traceback.format_exc())
            return {
                "status": "failed",
                "agent_id": agent_id,
                "parameters": parameters,
                "error": str(e),
                "message": f"Agent {agent_id} execution failed: {str(e)}"
            }
    
    def _create_agent_instance(self, agent_id: str):
        """Create an agent instance with proper initialization"""
        agent_class = self.registry.get_agent(agent_id)
        if not agent_class:
            raise AgentExecutionError(f"Agent not found: {agent_id}")
        
        # Initialize agent with default parameters based on agent type
        init_params = self._get_agent_init_params(agent_id)
        
        try:
            return agent_class(**init_params)
        except Exception as e:
            raise AgentExecutionError(f"Failed to create agent instance {agent_id}: {e}")
    
    def _get_agent_init_params(self, agent_id: str) -> Dict[str, Any]:
        """Get initialization parameters for different agent types"""
        
        # Import the LLM for agents that need it
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                api_key=settings.OPENAI_API_KEY
            )
        except Exception as e:
            logger.warning(f"Could not initialize OpenAI LLM: {e}")
            llm = None
        
        # Agent-specific initialization parameters based on actual constructor signatures
        if agent_id == "data_loader":
            # DataLoaderToolsAgent(model, create_react_agent_kwargs, invoke_react_agent_kwargs, checkpointer)
            return {
                "model": llm,
                "create_react_agent_kwargs": {},
                "invoke_react_agent_kwargs": {},
                "checkpointer": None
            }
        elif agent_id in ["data_cleaning", "data_wrangling", "feature_engineering"]:
            # BaseAgent-derived agents that need model, log, log_path, etc.
            return {
                "model": llm,
                "log": True,
                "log_path": "./temp",
                "overwrite": True,
                "human_in_the_loop": False,
                "bypass_recommended_steps": False,
                "bypass_explain_code": False,
                "checkpointer": None
            }
        elif agent_id == "data_visualization":  
            # DataVisualisationAgent - similar to other BaseAgent derivatives
            return {
                "model": llm,
                "log": True,
                "log_path": "./temp",
                "overwrite": True,
                "human_in_the_loop": False,
                "bypass_recommended_steps": False,
                "bypass_explain_code": False,
                "checkpointer": None
            }
        elif agent_id == "ml_prediction":
            # H2OMLAgent - BaseAgent derivative with model_directory
            return {
                "model": llm,
                "log": True,
                "log_path": "./temp",
                "model_directory": "./temp/models",
                "overwrite": True,
                "human_in_the_loop": False,
                "bypass_recommended_steps": False,
                "bypass_explain_code": False,
                "checkpointer": None
            }
        elif agent_id == "data_analysis":
            # DataAnalysisAgent(output_dir, intent_parser_model, enable_async)
            return {
                "output_dir": "./temp/analysis",
                "intent_parser_model": "gpt-4o-mini",
                "enable_async": True
            }
        elif agent_id == "supervisor":
            # SupervisorAgent - check its constructor
            return {
                "openai_api_key": settings.OPENAI_API_KEY,
            }
        else:
            # Default parameters for unknown agents
            return {"model": llm}
    
    def _execute_agent_by_type(
        self, 
        agent_instance, 
        agent_id: str, 
        parameters: Dict[str, Any], 
        async_mode: bool = False
    ) -> Dict[str, Any]:
        """Execute agent based on its type and interface"""
        
        if agent_id == "data_analysis":
            return self._execute_data_analysis_agent(agent_instance, parameters, async_mode)
        elif agent_id == "supervisor":
            return self._execute_supervisor_agent(agent_instance, parameters, async_mode)
        elif agent_id in ["data_loader", "data_cleaning", "data_visualization"]:
            return self._execute_base_agent(agent_instance, parameters, async_mode)
        elif agent_id in ["data_wrangling", "feature_engineering"]:
            return self._execute_data_processing_agent(agent_instance, parameters, async_mode)
        elif agent_id == "ml_prediction":
            return self._execute_ml_agent(agent_instance, parameters, async_mode)
        else:
            # Fallback to generic execution
            return self._execute_generic_agent(agent_instance, parameters, async_mode)
    
    def _execute_data_analysis_agent(self, agent, parameters: Dict[str, Any], async_mode: bool):
        """Execute Data Analysis Agent"""
        csv_url = parameters.get("csv_url", "")
        user_request = parameters.get("user_request", parameters.get("user_instructions", ""))
        
        if not csv_url:
            raise AgentExecutionError("csv_url parameter is required for data analysis agent")
        
        if async_mode:
            result = asyncio.run(agent.analyze_async(csv_url, user_request, **parameters))
        else:
            result = agent.analyze(csv_url, user_request, **parameters)
        
        # Convert DataAnalysisResult to dict if needed
        if hasattr(result, '__dict__'):
            return result.__dict__
        return result
    
    def _execute_supervisor_agent(self, agent, parameters: Dict[str, Any], async_mode: bool):
        """Execute Supervisor Agent"""
        csv_url = parameters.get("csv_url", "")
        natural_language_request = parameters.get("natural_language_request", 
                                                 parameters.get("user_instructions", ""))
        
        if not csv_url:
            raise AgentExecutionError("csv_url parameter is required for supervisor agent")
        
        if async_mode:
            result = asyncio.run(agent.analyze_async(csv_url, natural_language_request))
        else:
            result = agent.analyze(csv_url, natural_language_request)
        
        if hasattr(result, '__dict__'):
            return result.__dict__
        return result
    
    def _execute_base_agent(self, agent, parameters: Dict[str, Any], async_mode: bool):
        """Execute BaseAgent-derived agents (data loader, cleaning, visualization)"""
        user_instructions = parameters.get("user_instructions", "")
        
        # Remove user_instructions from parameters to avoid duplicate argument
        exec_params = {k: v for k, v in parameters.items() if k != "user_instructions"}
        
        if async_mode:
            asyncio.run(agent.ainvoke_agent(user_instructions=user_instructions, **exec_params))
        else:
            agent.invoke_agent(user_instructions=user_instructions, **exec_params)
        
        # Extract results from agent response
        return self._extract_base_agent_results(agent)
    
    def _execute_data_processing_agent(self, agent, parameters: Dict[str, Any], async_mode: bool):
        """Execute data processing agents (wrangling, feature engineering)"""
        data_raw = parameters.get("data_raw")
        user_instructions = parameters.get("user_instructions", "")
        
        # Convert data_raw if it's a file path or needs processing
        if isinstance(data_raw, str):
            # Assume it's a file path or CSV data
            try:
                if data_raw.startswith("http"):
                    data_raw = pd.read_csv(data_raw).to_dict()
                else:
                    # Try to read as CSV from uploads directory
                    import os
                    file_path = os.path.join(settings.UPLOAD_PATH, data_raw)
                    if os.path.exists(file_path):
                        data_raw = pd.read_csv(file_path).to_dict()
            except Exception as e:
                logger.warning(f"Could not process data_raw: {e}")
        
        # Remove processed parameters to avoid duplicates
        exec_params = {k: v for k, v in parameters.items() if k not in ["data_raw", "user_instructions"]}
        
        if async_mode:
            asyncio.run(agent.ainvoke_agent(data_raw=data_raw, user_instructions=user_instructions, **exec_params))
        else:
            agent.invoke_agent(data_raw=data_raw, user_instructions=user_instructions, **exec_params)
        
        return self._extract_base_agent_results(agent)
    
    def _execute_ml_agent(self, agent, parameters: Dict[str, Any], async_mode: bool):
        """Execute ML Prediction Agent"""
        data_raw = parameters.get("data_raw")
        user_instructions = parameters.get("user_instructions", "")
        target_variable = parameters.get("target_variable", "")
        
        # Process data_raw similar to data processing agents
        if isinstance(data_raw, str):
            try:
                if data_raw.startswith("http"):
                    data_raw = pd.read_csv(data_raw)
                else:
                    import os
                    file_path = os.path.join(settings.UPLOAD_PATH, data_raw)
                    if os.path.exists(file_path):
                        data_raw = pd.read_csv(file_path)
            except Exception as e:
                logger.warning(f"Could not process data_raw for ML agent: {e}")
                raise AgentExecutionError(f"Invalid data_raw parameter: {e}")
        
        if not isinstance(data_raw, pd.DataFrame):
            raise AgentExecutionError("data_raw must be a pandas DataFrame for ML agent")
        
        # Remove processed parameters to avoid duplicates
        exec_params = {k: v for k, v in parameters.items() if k not in ["data_raw", "user_instructions", "target_variable"]}
        
        if async_mode:
            asyncio.run(agent.ainvoke_agent(data_raw=data_raw, user_instructions=user_instructions, target_variable=target_variable, **exec_params))
        else:
            agent.invoke_agent(data_raw=data_raw, user_instructions=user_instructions, target_variable=target_variable, **exec_params)
        
        # Extract ML-specific results
        results = self._extract_base_agent_results(agent)
        
        # Add ML-specific outputs
        try:
            if hasattr(agent, 'get_leaderboard'):
                leaderboard = agent.get_leaderboard()
                if leaderboard is not None:
                    results["leaderboard"] = leaderboard.to_dict() if hasattr(leaderboard, 'to_dict') else leaderboard
            
            if hasattr(agent, 'get_best_model_id'):
                results["best_model_id"] = agent.get_best_model_id()
            
            if hasattr(agent, 'get_model_path'):
                results["model_path"] = agent.get_model_path()
        except Exception as e:
            logger.warning(f"Could not extract ML-specific results: {e}")
        
        return results
    
    def _execute_generic_agent(self, agent, parameters: Dict[str, Any], async_mode: bool):
        """Generic execution fallback"""
        user_instructions = parameters.get("user_instructions", "")
        
        if hasattr(agent, 'ainvoke_agent') and async_mode:
            asyncio.run(agent.ainvoke_agent(user_instructions, **parameters))
        elif hasattr(agent, 'invoke_agent'):
            agent.invoke_agent(user_instructions, **parameters)
        elif hasattr(agent, 'invoke'):
            if async_mode:
                asyncio.run(agent.ainvoke(parameters))
            else:
                agent.invoke(parameters)
        else:
            raise AgentExecutionError(f"Agent does not have a recognized execution method")
        
        return self._extract_base_agent_results(agent)
    
    def _extract_base_agent_results(self, agent) -> Dict[str, Any]:
        """Extract results from BaseAgent-derived agents"""
        results = {}
        
        # Get the main response
        if hasattr(agent, 'response') and agent.response:
            results["response"] = agent.response
        
        # Extract common result methods
        try:
            if hasattr(agent, 'get_ai_message'):
                results["ai_message"] = agent.get_ai_message()
            
            if hasattr(agent, 'get_artifacts'):
                artifacts = agent.get_artifacts()
                if artifacts:
                    results["artifacts"] = artifacts
            
            if hasattr(agent, 'get_internal_messages'):
                results["internal_messages"] = agent.get_internal_messages()
            
            if hasattr(agent, 'get_workflow_summary'):
                results["workflow_summary"] = agent.get_workflow_summary()
            
            if hasattr(agent, 'get_tool_calls'):
                results["tool_calls"] = agent.get_tool_calls()
                
        except Exception as e:
            logger.warning(f"Could not extract some agent results: {e}")
        
        return results


# Global service instance
agent_execution_service = AgentExecutionService() 