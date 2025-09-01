#!/usr/bin/env python3
"""
Standalone Workflow Server for testing the Quick Data Analysis workflow
This server doesn't depend on any existing app configuration to avoid import issues.
"""

import asyncio
import time
import uuid
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import uvicorn

# Response models
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

# In-memory storage for testing
workflow_executions = {}

app = FastAPI(
    title="AI Data Science Workflow API",
    description="API for executing Quick Data Analysis workflows",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "AI Data Science Workflow API",
        "version": "0.1.0",
        "docs": "/docs",
        "workflows": "/api/workflows/"
    }

@app.get("/api/workflows/health")
async def health_check():
    """Check health of workflow agents"""
    return {
        "all_agents_healthy": True,
        "agent_health": {
            "loading": {"status": "ready", "port": 8005},
            "cleaning": {"status": "ready", "port": 8004},
            "visualization": {"status": "ready", "port": 8006}
        },
        "workflow_service_status": "ready"
    }

@app.get("/api/workflows/templates")
async def get_workflow_templates():
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

def simulate_agent_execution(agent_type: str, step_num: int, total_steps: int) -> Dict[str, Any]:
    """Simulate agent execution for testing"""
    
    # Simulate different execution times for different agents
    execution_times = {
        'loading': 3,
        'cleaning': 5,
        'visualization': 4,
        'engineering': 6,
        'training': 10
    }
    
    execution_time = execution_times.get(agent_type, 3)
    time.sleep(execution_time)
    
    session_id = f"session_{agent_type}_{int(time.time())}"
    
    return {
        'success': True,
        'session_id': session_id,
        'execution_time_seconds': execution_time,
        'message': f'{agent_type.title()} agent completed successfully'
    }

async def execute_workflow_simulation(workflow_name: str, file_name: str, steps: List[str]) -> str:
    """Execute workflow simulation and return workflow_id"""
    
    workflow_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Initialize workflow execution
    execution = {
        'id': workflow_id,
        'name': workflow_name,
        'status': 'running',
        'current_step': 0,
        'total_steps': len(steps),
        'progress_percentage': 0,
        'execution_time': 0,
        'start_time': start_time,
        'steps': []
    }
    
    workflow_executions[workflow_id] = execution
    
    # Execute each step
    for i, step_type in enumerate(steps):
        execution['current_step'] = i + 1
        execution['progress_percentage'] = ((i + 1) / len(steps)) * 100
        execution['execution_time'] = time.time() - start_time
        
        print(f"Executing step {i+1}/{len(steps)}: {step_type}")
        
        # Simulate agent execution
        step_result = simulate_agent_execution(step_type, i + 1, len(steps))
        
        step_info = {
            'id': str(uuid.uuid4()),
            'agent_type': step_type,
            'status': 'completed',
            'session_id': step_result['session_id'],
            'execution_time_seconds': step_result['execution_time_seconds'],
            'error': None
        }
        
        execution['steps'].append(step_info)
        
        # Update execution in storage
        workflow_executions[workflow_id] = execution
    
    # Mark as completed
    execution['status'] = 'completed'
    execution['execution_time'] = time.time() - start_time
    workflow_executions[workflow_id] = execution
    
    return workflow_id

@app.post("/api/workflows/execute-quick-analysis", response_model=WorkflowExecutionResponse)
async def execute_quick_analysis(
    file: UploadFile = File(...),
    user_instructions: str = Form("Perform quick data analysis on the uploaded file")
):
    """Execute the Quick Data Analysis workflow with file upload"""
    
    try:
        print(f"🚀 Starting Quick Analysis workflow for file: {file.filename}")
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Read file content to validate it's not empty
        file_content = await file.read()
        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
        # Reset file pointer for potential re-reading
        await file.seek(0)
        
        # Define workflow steps for Quick Analysis
        steps = ['loading', 'cleaning', 'visualization']
        workflow_name = f"Quick Analysis - {file.filename}"
        
        # Start workflow execution in background task
        workflow_id = await execute_workflow_simulation(workflow_name, file.filename, steps)
        
        return WorkflowExecutionResponse(
            success=True,
            workflow_id=workflow_id,
            message=f"Quick Analysis workflow for '{file.filename}' completed successfully",
            status="completed"
        )
        
    except Exception as e:
        print(f"❌ Quick Analysis workflow failed: {e}")
        raise HTTPException(status_code=500, detail=f"Quick Analysis workflow failed: {str(e)}")

@app.get("/api/workflows/{workflow_id}/status", response_model=WorkflowStatusResponse)
async def get_workflow_status(workflow_id: str):
    """Get the current status of a workflow execution"""
    
    execution = workflow_executions.get(workflow_id)
    if not execution:
        raise HTTPException(status_code=404, detail=f"Workflow not found: {workflow_id}")
    
    return WorkflowStatusResponse(
        id=execution['id'],
        name=execution['name'],
        status=execution['status'],
        current_step=execution['current_step'],
        total_steps=execution['total_steps'],
        progress_percentage=execution['progress_percentage'],
        execution_time=execution['execution_time'],
        steps=[
            {
                'agent_type': step['agent_type'],
                'status': step['status'],
                'session_id': step.get('session_id'),
                'error': step.get('error')
            }
            for step in execution['steps']
        ]
    )

@app.get("/api/workflows/{workflow_id}/results")
async def get_workflow_results(workflow_id: str):
    """Get the complete results from a workflow execution"""
    
    execution = workflow_executions.get(workflow_id)
    if not execution:
        raise HTTPException(status_code=404, detail=f"Workflow not found: {workflow_id}")
    
    if execution['status'] not in ['completed', 'failed']:
        raise HTTPException(
            status_code=400, 
            detail=f"Workflow not completed yet. Current status: {execution['status']}"
        )
    
    return {
        'workflow_id': execution['id'],
        'name': execution['name'],
        'status': execution['status'],
        'total_execution_time': execution['execution_time'],
        'results': {
            'message': f'Workflow "{execution["name"]}" completed successfully',
            'steps_completed': len([s for s in execution['steps'] if s['status'] == 'completed']),
            'sessions_created': [step['session_id'] for step in execution['steps']],
            'execution_summary': {
                'total_time': f"{execution['execution_time']:.1f}s",
                'steps': [
                    {
                        'agent': step['agent_type'],
                        'time': f"{step['execution_time_seconds']:.1f}s",
                        'session': step['session_id']
                    }
                    for step in execution['steps']
                ]
            }
        },
        'steps': execution['steps']
    }

@app.get("/api/workflows/")
async def list_workflows():
    """List all workflow executions"""
    
    return [
        {
            'id': execution['id'],
            'name': execution['name'],
            'status': execution['status'],
            'created_at': execution['start_time'],
            'total_execution_time': execution.get('execution_time'),
            'total_steps': execution['total_steps'],
            'completed_steps': len([s for s in execution['steps'] if s['status'] == 'completed'])
        }
        for execution in workflow_executions.values()
    ]

if __name__ == "__main__":
    print("🚀 Starting AI Data Science Workflow API Server...")
    print("📡 Available endpoints:")
    print("   GET  http://127.0.0.1:8000/")
    print("   GET  http://127.0.0.1:8000/docs")
    print("   GET  http://127.0.0.1:8000/api/workflows/health")
    print("   GET  http://127.0.0.1:8000/api/workflows/templates")
    print("   POST http://127.0.0.1:8000/api/workflows/execute-quick-analysis")
    print("   GET  http://127.0.0.1:8000/api/workflows/{id}/status")
    print("   GET  http://127.0.0.1:8000/api/workflows/{id}/results")
    print("   GET  http://127.0.0.1:8000/api/workflows/")
    print("")
    print("🎯 Frontend can connect to this API for workflow execution!")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
