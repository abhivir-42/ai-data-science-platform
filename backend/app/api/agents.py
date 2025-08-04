"""
Agents API endpoints
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from loguru import logger

from app.core.agent_registry import agent_registry
from app.services.agent_execution import agent_execution_service

router = APIRouter()


class AgentExecutionRequest(BaseModel):
    """Request model for agent execution"""
    parameters: Dict[str, Any] = {}
    background: bool = False


class AgentExecutionResponse(BaseModel):
    """Response model for agent execution"""
    job_id: Optional[str] = None
    status: str
    result: Optional[Dict[str, Any]] = None
    message: str


@router.get("/")
async def list_agents() -> List[Dict[str, Any]]:
    """List all available agents"""
    try:
        agents = agent_registry.list_agents()
        logger.info(f"Listed {len(agents)} agents")
        return agents
    except Exception as e:
        logger.error(f"Failed to list agents: {e}")
        raise HTTPException(status_code=500, detail="Failed to list agents")


@router.get("/{agent_id}")
async def get_agent_details(agent_id: str) -> Dict[str, Any]:
    """Get details for a specific agent"""
    try:
        metadata = agent_registry.get_agent_metadata(agent_id)
        if not metadata:
            raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")
        
        return {
            "id": agent_id,
            **metadata
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agent details for {agent_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get agent details")


@router.get("/{agent_id}/schema")
async def get_agent_schema(agent_id: str) -> Dict[str, Any]:
    """Get parameter schema for a specific agent"""
    try:
        agent_class = agent_registry.get_agent(agent_id)
        if not agent_class:
            raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")
        
        # Get agent metadata
        metadata = agent_registry.get_agent_metadata(agent_id)
        
        # Generate parameter schema based on agent type
        schema = _generate_agent_parameter_schema(agent_id, metadata)
        
        return {
            "agent_id": agent_id,
            "name": metadata.get("name", "Unknown Agent"),
            "description": metadata.get("description", ""),
            "inputs": metadata.get("inputs", []),
            "outputs": metadata.get("outputs", []),
            "parameters": schema
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get schema for agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get agent schema")


@router.post("/{agent_id}/execute")
async def execute_agent(
    agent_id: str,
    request: AgentExecutionRequest,
    background_tasks: BackgroundTasks
) -> AgentExecutionResponse:
    """Execute a specific agent"""
    try:
        # Check if agent exists
        agent_class = agent_registry.get_agent(agent_id)
        if not agent_class:
            raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")
        
        logger.info(f"Executing agent {agent_id} with parameters: {request.parameters}")
        
        if request.background:
            # TODO: Implement background execution with Celery
            # For now, return a mock job ID
            job_id = f"job_{agent_id}_{hash(str(request.parameters))}"
            return AgentExecutionResponse(
                job_id=job_id,
                status="queued",
                message=f"Agent {agent_id} execution queued with job ID: {job_id}"
            )
        else:
            # Synchronous execution using the execution service
            try:
                result = agent_execution_service.execute_agent_sync(agent_id, request.parameters)
                
                return AgentExecutionResponse(
                    status=result["status"],
                    result=result.get("result"),
                    message=result["message"]
                )
                
            except Exception as e:
                logger.error(f"Agent execution failed: {e}")
                return AgentExecutionResponse(
                    status="failed",
                    message=f"Agent execution failed: {str(e)}"
                )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to execute agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to execute agent")


@router.get("/categories/{category}")
async def get_agents_by_category(category: str) -> List[Dict[str, Any]]:
    """Get agents by category"""
    try:
        agents = agent_registry.get_agents_by_category(category)
        logger.info(f"Found {len(agents)} agents in category: {category}")
        return agents
    except Exception as e:
        logger.error(f"Failed to get agents by category {category}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get agents by category")


def _generate_agent_parameter_schema(agent_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Generate parameter schema based on agent type"""
    
    # Base schema structure
    base_schema = {
        "type": "object",
        "properties": {},
        "required": []
    }
    
    # Agent-specific parameter schemas
    if agent_id == "data_analysis":
        base_schema["properties"] = {
            "csv_url": {
                "type": "string",
                "description": "URL or path to CSV file for analysis",
                "format": "uri"
            },
            "user_request": {
                "type": "string", 
                "description": "Natural language analysis request"
            }
        }
        base_schema["required"] = ["csv_url", "user_request"]
        
    elif agent_id == "supervisor":
        base_schema["properties"] = {
            "csv_url": {
                "type": "string",
                "description": "URL or path to CSV file for analysis",
                "format": "uri"
            },
            "natural_language_request": {
                "type": "string",
                "description": "Natural language request for analysis workflow"
            }
        }
        base_schema["required"] = ["csv_url", "natural_language_request"]
        
    elif agent_id in ["data_wrangling", "feature_engineering"]:
        base_schema["properties"] = {
            "data_raw": {
                "type": "string",
                "description": "Raw data (file path, URL, or uploaded file ID)"
            },
            "user_instructions": {
                "type": "string",
                "description": "Instructions for data processing"
            }
        }
        base_schema["required"] = ["data_raw", "user_instructions"]
        
    elif agent_id == "ml_prediction":
        base_schema["properties"] = {
            "data_raw": {
                "type": "string", 
                "description": "Dataset for ML training (file path, URL, or uploaded file ID)"
            },
            "target_variable": {
                "type": "string",
                "description": "Target variable for prediction/classification"
            },
            "user_instructions": {
                "type": "string",
                "description": "ML training instructions and requirements"
            }
        }
        base_schema["required"] = ["data_raw", "target_variable", "user_instructions"]
        
    elif agent_id in ["data_loader", "data_cleaning", "data_visualization"]:
        base_schema["properties"] = {
            "user_instructions": {
                "type": "string",
                "description": "Instructions for the agent"
            }
        }
        base_schema["required"] = ["user_instructions"]
        
    else:
        # Generic schema
        base_schema["properties"] = {
            "user_instructions": {
                "type": "string",
                "description": "Instructions for the agent"
            }
        }
        base_schema["required"] = ["user_instructions"]
    
    return base_schema 