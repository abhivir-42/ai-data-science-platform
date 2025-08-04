"""
Agent Registry for managing AI Data Science agents
"""

from typing import Dict, Type, List, Any, Optional
from loguru import logger

# Import all agents
from app.agents.data_loader_tools_agent import DataLoaderToolsAgent
from app.agents.data_cleaning_agent import DataCleaningAgent
from app.agents.data_wrangling_agent import DataWranglingAgent
from app.agents.feature_engineering_agent import FeatureEngineeringAgent
from app.agents.data_visualisation_agent import DataVisualisationAgent
from app.agents.supervisor_agent import SupervisorAgent
from app.agents.data_analysis_agent import DataAnalysisAgent
from app.agents.ml_prediction_agent import MLPredictionAgent


class AgentRegistry:
    """Registry for managing all available AI agents"""
    
    def __init__(self):
        self._agents: Dict[str, Type] = {}
        self._agent_metadata: Dict[str, Dict[str, Any]] = {}
        self._register_agents()
    
    def _register_agents(self):
        """Register all available agents"""
        
        # Register each agent with metadata
        agents_config = [
            {
                "id": "data_loader",
                "name": "Data Loader Tools Agent", 
                "class": DataLoaderToolsAgent,
                "description": "Load and process data from various sources (CSV, Excel, JSON, PDF, etc.)",
                "category": "data_ingestion",
                "inputs": ["file_path", "url", "directory"],
                "outputs": ["dataframe", "summary", "metadata"]
            },
            {
                "id": "data_cleaning",
                "name": "Data Cleaning Agent",
                "class": DataCleaningAgent, 
                "description": "Clean and preprocess datasets (handle missing values, outliers, duplicates)",
                "category": "data_preprocessing",
                "inputs": ["dataframe", "cleaning_instructions"],
                "outputs": ["cleaned_dataframe", "cleaning_report"]
            },
            {
                "id": "data_wrangling",
                "name": "Data Wrangling Agent",
                "class": DataWranglingAgent,
                "description": "Transform and reshape data (join, pivot, aggregate, compute features)",
                "category": "data_transformation", 
                "inputs": ["dataframe", "transformation_rules"],
                "outputs": ["transformed_dataframe", "transformation_log"]
            },
            {
                "id": "feature_engineering",
                "name": "Feature Engineering Agent",
                "class": FeatureEngineeringAgent,
                "description": "Create and encode features for machine learning",
                "category": "feature_creation",
                "inputs": ["dataframe", "target_variable", "engineering_instructions"],
                "outputs": ["engineered_features", "encoding_report"]
            },
            {
                "id": "data_visualization",
                "name": "Data Visualization Agent", 
                "class": DataVisualisationAgent,
                "description": "Generate interactive charts and visualizations",
                "category": "visualization",
                "inputs": ["dataframe", "chart_type", "visualization_requirements"],
                "outputs": ["plotly_chart", "visualization_code"]
            },
            {
                "id": "supervisor",
                "name": "Supervisor Agent",
                "class": SupervisorAgent,
                "description": "Orchestrate multi-agent workflows and coordinate analysis tasks",
                "category": "orchestration",
                "inputs": ["natural_language_request", "csv_url"],
                "outputs": ["comprehensive_analysis", "workflow_results"]
            },
            {
                "id": "data_analysis",
                "name": "Data Analysis Agent",
                "class": DataAnalysisAgent,
                "description": "Enhanced workflow management and structured analysis",
                "category": "analysis",
                "inputs": ["analysis_request", "datasets"],
                "outputs": ["structured_results", "analysis_insights"]
            },
            {
                "id": "ml_prediction",
                "name": "ML Prediction Agent",
                "class": MLPredictionAgent,
                "description": "Train machine learning models and make predictions",
                "category": "machine_learning",
                "inputs": ["dataset", "target_variable", "ml_parameters"],
                "outputs": ["trained_model", "predictions", "model_metrics"]
            }
        ]
        
        for agent_config in agents_config:
            agent_id = agent_config["id"]
            agent_class = agent_config["class"]
            
            self._agents[agent_id] = agent_class
            self._agent_metadata[agent_id] = {
                k: v for k, v in agent_config.items() if k != "class"
            }
            
            logger.info(f"Registered agent: {agent_id} ({agent_config['name']})")
    
    def get_agent(self, agent_id: str) -> Optional[Type]:
        """Get agent class by ID"""
        return self._agents.get(agent_id)
    
    def get_agent_metadata(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent metadata by ID"""
        return self._agent_metadata.get(agent_id)
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """List all registered agents with their metadata"""
        return [
            {
                "id": agent_id,
                **metadata
            }
            for agent_id, metadata in self._agent_metadata.items()
        ]
    
    def get_agents_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get agents by category"""
        return [
            {
                "id": agent_id,
                **metadata
            }
            for agent_id, metadata in self._agent_metadata.items()
            if metadata.get("category") == category
        ]
    
    def create_agent_instance(self, agent_id: str, **kwargs) -> Optional[Any]:
        """Create an instance of the specified agent"""
        agent_class = self.get_agent(agent_id)
        if not agent_class:
            logger.error(f"Agent not found: {agent_id}")
            return None
        
        try:
            return agent_class(**kwargs)
        except Exception as e:
            logger.error(f"Failed to create agent instance {agent_id}: {e}")
            return None


# Global agent registry instance
agent_registry = AgentRegistry() 