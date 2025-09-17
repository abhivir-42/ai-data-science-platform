"""
Workflow API endpoints for orchestrating multi-agent workflows
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Form
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
import base64
import asyncio
import os

from app.services.workflow_execution import workflow_execution_service, WorkflowStatus
from app.core.logging import logger


router = APIRouter(prefix="/workflows", tags=["workflows"])


class WorkflowStepRequest(BaseModel):
    agent_type: str
    parameters: Dict[str, Any] = {}


class WorkflowExecutionRequest(BaseModel):
    name: str
    steps: List[WorkflowStepRequest]
    initial_data: Optional[Dict[str, Any]] = None


class WorkflowExecutionResponse(BaseModel):
    success: bool
    workflow_id: str
    message: str
    status: str


class WorkflowStatusResponse(BaseModel):
    id: str
    name: str
    status: str
    current_step: int
    total_steps: int
    progress_percentage: float
    execution_time: float
    steps: List[Dict[str, Any]]


@router.post("/execute", response_model=WorkflowExecutionResponse)
async def execute_workflow(
    request: WorkflowExecutionRequest,
    background_tasks: BackgroundTasks
) -> WorkflowExecutionResponse:
    """
    Execute a multi-agent workflow
    
    This endpoint accepts a workflow definition and executes it by chaining
    multiple AI agents together. Each step's output becomes the next step's input.
    """
    try:
        logger.info(f"Starting workflow execution: {request.name}")
        
        # Convert request to format expected by service
        workflow_steps = [
            {
                'agent_type': step.agent_type,
                'parameters': step.parameters
            }
            for step in request.steps
        ]
        
        # Execute workflow in background for long-running operations
        # TODO: Extract user_id from request context when authentication is implemented
        execution = await workflow_execution_service.execute_workflow(
            workflow_name=request.name,
            steps=workflow_steps,
            initial_data=request.initial_data
        )
        
        return WorkflowExecutionResponse(
            success=True,
            workflow_id=execution.id,
            message=f"Workflow '{request.name}' execution completed",
            status=execution.status.value if hasattr(execution.status, 'value') else str(execution.status)
        )
        
    except ValueError as ve:
        logger.error(f"Workflow validation error: {ve}")
        raise HTTPException(status_code=400, detail=f"Invalid workflow configuration: {str(ve)}")
    except TimeoutError as te:
        logger.error(f"Workflow timeout: {te}")
        raise HTTPException(status_code=408, detail=f"Workflow execution timed out: {str(te)}")
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {str(e)}")


@router.post("/execute-quick-analysis", response_model=WorkflowExecutionResponse)
async def execute_quick_analysis(
    file: UploadFile = File(...),
    user_instructions: str = Form("Perform quick data analysis on the uploaded file")
) -> WorkflowExecutionResponse:
    """
    Execute the Quick Data Analysis workflow with file upload
    
    This is a convenience endpoint that:
    1. Accepts a file upload
    2. Executes the predefined Quick Analysis workflow (Load → Clean → Visualize)
    3. Returns the workflow execution ID for tracking progress
    """
    try:
        # Validate file before processing
        if not file.filename:
            raise HTTPException(status_code=400, detail="File name is required")

        # Check file extension
        allowed_extensions = ['.csv', '.xlsx', '.xls', '.json', '.parquet', '.txt']
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_ext}. Supported types: {', '.join(allowed_extensions)}"
            )

        # Read and validate file content
        file_content = await file.read()

        # Check file size (50MB limit)
        max_size = 50 * 1024 * 1024  # 50MB
        if len(file_content) > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large: {len(file_content)} bytes. Maximum size: {max_size} bytes (50MB)"
            )

        # Check for empty files
        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        # Encode to base64
        try:
            file_content_b64 = base64.b64encode(file_content).decode('utf-8')
        except Exception as encode_error:
            logger.error(f"Base64 encoding failed: {encode_error}")
            raise HTTPException(status_code=500, detail="Failed to encode file content")

        logger.info(f"Starting Quick Analysis workflow for file: {file.filename} ({len(file_content)} bytes, {file_ext})")
        
        # Define the Quick Analysis workflow steps
        workflow_steps = [
            {
                'agent_type': 'loading',
                'parameters': {
                    'filename': file.filename,
                    'file_content': file_content_b64,
                    'user_instructions': 'Load and analyze the uploaded file'
                }
            },
            {
                'agent_type': 'cleaning', 
                'parameters': {
                    'user_instructions': 'Clean the data using recommended steps: handle missing values, remove duplicates, detect outliers'
                }
            },
            {
                'agent_type': 'visualization',
                'parameters': {
                    'user_instructions': 'Create comprehensive visualizations to understand the data: distributions, correlations, and key insights'
                }
            }
        ]
        
        # Execute the workflow asynchronously
        # TODO: Extract user_id from request context when authentication is implemented
        execution = await workflow_execution_service.start_workflow_async(
            workflow_name=f"Quick Analysis - {file.filename}",
            steps=workflow_steps,
            initial_data={}
        )
        
        return WorkflowExecutionResponse(
            success=True,
            workflow_id=execution.id,
            message=f"Quick Analysis workflow for '{file.filename}' started",
            status=execution.status.value if hasattr(execution.status, 'value') else str(execution.status)
        )
        
    except ValueError as ve:
        logger.error(f"Quick Analysis validation error: {ve}")
        raise HTTPException(status_code=400, detail=f"Invalid file or parameters: {str(ve)}")
    except TimeoutError as te:
        logger.error(f"Quick Analysis timeout: {te}")
        raise HTTPException(status_code=408, detail=f"Analysis timed out - file may be too large: {str(te)}")
    except Exception as e:
        logger.error(f"Quick Analysis workflow failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Quick Analysis workflow failed: {str(e)}")


@router.post("/execute-ml-pipeline", response_model=WorkflowExecutionResponse)
async def execute_ml_pipeline(
    file: UploadFile = File(...),
    target_variable: str = Form(...),
    user_instructions: str = Form("Train machine learning models on the uploaded dataset"),
    max_runtime_secs: int = Form(300)
) -> WorkflowExecutionResponse:
    """
    Execute the Complete ML Pipeline workflow with file upload
    
    This executes: Load → Clean → Feature Engineering → ML Training → Prediction
    """
    try:
        # Read and encode the uploaded file
        file_content = await file.read()
        file_content_b64 = base64.b64encode(file_content).decode('utf-8')
        
        logger.info(f"Starting ML Pipeline workflow for file: {file.filename}, target: {target_variable}")
        
        # Define the ML Pipeline workflow steps
        workflow_steps = [
            {
                'agent_type': 'loading',
                'parameters': {
                    'filename': file.filename,
                    'file_content': file_content_b64,
                    'user_instructions': 'Load and analyze the uploaded file for ML training'
                }
            },
            {
                'agent_type': 'cleaning',
                'parameters': {
                    'user_instructions': 'Clean the data for machine learning: handle missing values, remove duplicates, prepare for training'
                }
            },
            {
                'agent_type': 'engineering',
                'parameters': {
                    'target_variable': target_variable,
                    'user_instructions': 'Engineer features for machine learning: create new features, encode categoricals, scale features'
                }
            },
            {
                'agent_type': 'training',
                'parameters': {
                    'target_variable': target_variable,
                    'user_instructions': user_instructions,
                    'max_runtime_secs': max_runtime_secs
                }
            }
        ]
        
        # Execute the workflow
        execution = await workflow_execution_service.execute_workflow(
            workflow_name=f"ML Pipeline - {file.filename}",
            steps=workflow_steps,
            initial_data={}
        )
        
        return WorkflowExecutionResponse(
            success=True,
            workflow_id=execution.id,
            message=f"ML Pipeline workflow for '{file.filename}' completed",
            status=execution.status.value if hasattr(execution.status, 'value') else str(execution.status)
        )
        
    except Exception as e:
        logger.error(f"ML Pipeline workflow failed: {e}")
        raise HTTPException(status_code=500, detail=f"ML Pipeline workflow failed: {str(e)}")


@router.get("/{workflow_id}/status", response_model=WorkflowStatusResponse)
async def get_workflow_status(workflow_id: str) -> WorkflowStatusResponse:
    """Get the current status of a workflow execution"""
    
    try:
        status = await workflow_execution_service.get_execution_status(workflow_id)
        if not status:
            raise HTTPException(status_code=404, detail=f"Workflow not found: {workflow_id}")
        
        return WorkflowStatusResponse(**status)
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Failed to get workflow status for {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get workflow status: {str(e)}")


@router.get("/{workflow_id}/results")
async def get_workflow_results(workflow_id: str) -> Dict[str, Any]:
    """Get the complete results from a workflow execution"""
    
    try:
        execution = await workflow_execution_service.get_execution(workflow_id)
        if not execution:
            raise HTTPException(status_code=404, detail=f"Workflow not found: {workflow_id}")
        
        execution_status = execution.status.value if hasattr(execution.status, 'value') else str(execution.status)
        if execution_status != 'completed':
            status_str = execution.status.value if hasattr(execution.status, 'value') else str(execution.status)
            raise HTTPException(
                status_code=400,
                detail=f"Workflow not completed yet. Current status: {status_str}. Please wait for completion before requesting results."
            )

        status_str = execution.status.value if hasattr(execution.status, 'value') else str(execution.status)
        return {
            'workflow_id': execution.id,
            'name': execution.name,
            'status': status_str,
            'total_execution_time': execution.total_execution_time,
            'results': execution.results,
            'steps': [
                {
                    'id': step.id,
                    'agent_type': step.agent_type,
                    'status': step.status.value if hasattr(step.status, 'value') else str(step.status),
                    'session_id': step.session_id,
                    'execution_time_seconds': step.execution_time_seconds,
                    'error': step.error
                }
                for step in execution.steps
            ]
        }
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Failed to get workflow results for {workflow_id}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to retrieve workflow results: {str(e)}")


@router.get("/", response_model=List[Dict[str, Any]])
async def list_workflows() -> List[Dict[str, Any]]:
    """List all workflow executions"""
    
    executions = workflow_execution_service.list_executions()
    
    return [
        {
            'id': execution.id,
            'name': execution.name,
            'status': execution.status.value if hasattr(execution.status, 'value') else str(execution.status),
            'created_at': execution.created_at,
            'total_execution_time': execution.total_execution_time,
            'total_steps': len(execution.steps),
            'completed_steps': len([s for s in execution.steps if (s.status == WorkflowStatus.COMPLETED if hasattr(s.status, 'value') else str(s.status) == 'completed')])
        }
        for execution in executions
    ]


@router.get("/templates")
async def get_workflow_templates() -> List[Dict[str, Any]]:
    """Get predefined workflow templates"""
    
    return [
        {
            'id': 'quick-analysis',
            'name': 'Quick Data Analysis',
            'description': 'Load → Clean → Visualize workflow for rapid insights',
            'steps': ['loading', 'cleaning', 'visualization'],
            'estimated_time': '5-10 min',
            'difficulty': 'Beginner'
        },
        {
            'id': 'ml-pipeline',
            'name': 'Complete ML Pipeline',
            'description': 'End-to-end machine learning workflow from data to predictions',
            'steps': ['loading', 'cleaning', 'engineering', 'training'],
            'estimated_time': '20-30 min',
            'difficulty': 'Advanced'
        },
        {
            'id': 'data-prep',
            'name': 'Data Preparation',
            'description': 'Clean and prepare your data for analysis with feature engineering',
            'steps': ['loading', 'cleaning', 'engineering'],
            'estimated_time': '10-15 min',
            'difficulty': 'Intermediate'
        }
    ]


@router.get("/health")
async def check_agents_health() -> Dict[str, Any]:
    """Check health of all uAgents required for workflows"""
    
    from app.lib.uagent_client import UAgentClient
    
    client = UAgentClient()
    health_checks = await client.check_all_agents_health()
    
    all_healthy = all(
        agent_health.get('status') == 'ready' or agent_health.get('agent_status') == 'ready'
        for agent_health in health_checks.values()
    )
    
    return {
        'all_agents_healthy': all_healthy,
        'agent_health': health_checks,
        'workflow_service_status': 'ready'
    }
