"""
Agents API endpoints
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from loguru import logger

from app.core.agent_registry import agent_registry

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
        
        # TODO: Extract actual schema from agent class
        # This is a placeholder implementation
        metadata = agent_registry.get_agent_metadata(agent_id)
        
        return {
            "agent_id": agent_id,
            "name": metadata.get("name", "Unknown Agent"),
            "description": metadata.get("description", ""),
            "inputs": metadata.get("inputs", []),
            "outputs": metadata.get("outputs", []),
            "parameters": {
                # TODO: Generate actual parameter schema from agent class
                "type": "object",
                "properties": {},
                "required": []
            }
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
            # Synchronous execution
            try:
                # Create agent instance
                agent_instance = agent_registry.create_agent_instance(agent_id)
                if not agent_instance:
                    raise HTTPException(status_code=500, detail=f"Failed to create agent instance: {agent_id}")
                
                # TODO: Execute agent with parameters
                # This is a placeholder - actual execution depends on agent interface
                result = {
                    "agent_id": agent_id,
                    "parameters": request.parameters,
                    "status": "completed",
                    "message": f"Agent {agent_id} executed successfully (placeholder)"
                }
                
                return AgentExecutionResponse(
                    status="completed",
                    result=result,
                    message=f"Agent {agent_id} executed successfully"
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