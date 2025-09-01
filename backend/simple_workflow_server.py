#!/usr/bin/env python3
"""
Simple Workflow Server for testing the Quick Data Analysis workflow
"""

import asyncio
import time
import uuid
import base64
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import uvicorn

# Simple response models
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

# Simple in-memory storage for testing
workflow_executions = {}

app = FastAPI(
    title="Quick Workflow API Test",
    description="Test API for Quick Data Analysis workflow",
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
    return {"message": "Quick Workflow API Test Server", "docs": "/docs"}

@app.get("/api/workflows/health")
async def health_check():
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
    return [
        {
            'id': 'quick-analysis',
            'name': 'Quick Data Analysis',
            'description': 'Load → Clean → Visualize workflow for rapid insights',
            'steps': ['loading', 'cleaning', 'visualization'],
            'estimated_time': '5-10 min',
            'difficulty': 'Beginner'
        }
    ]

def simulate_agent_execution(agent_type: str, step_num: int) -> Dict[str, Any]:
    """Simulate agent execution for testing"""
    time.sleep(2)  # Simulate processing time
    
    session_id = f"session_{agent_type}_{int(time.time())}"
    
    return {
        'success': True,
        'session_id': session_id,
        'execution_time_seconds': 2.0,
        'message': f'{agent_type.title()} agent completed successfully'
    }

async def simulate_workflow_execution(workflow_name: str, file_name: str) -> Dict[str, Any]:
    """Simulate the Quick Data Analysis workflow execution"""
    
    workflow_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Define workflow steps
    steps = [
        {'agent_type': 'loading', 'name': 'Data Loader'},
        {'agent_type': 'cleaning', 'name': 'Data Cleaning'}, 
        {'agent_type': 'visualization', 'name': 'Data Visualization'}
    ]
    
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
    for i, step in enumerate(steps):
        execution['current_step'] = i + 1
        execution['progress_percentage'] = ((i + 1) / len(steps)) * 100
        execution['execution_time'] = time.time() - start_time
        
        print(f"Executing step {i+1}/{len(steps)}: {step['agent_type']}")
        
        # Simulate agent execution
        step_result = simulate_agent_execution(step['agent_type'], i + 1)
        
        step_info = {
            'id': str(uuid.uuid4()),
            'agent_type': step['agent_type'],
            'status': 'completed',
            'session_id': step_result['session_id'],
            'execution_time_seconds': step_result['execution_time_seconds'],
            'error': None
        }
        
        execution['steps'].append(step_info)
    
    # Mark as completed
    execution['status'] = 'completed'
    execution['execution_time'] = time.time() - start_time
    
    return execution

@app.post("/api/workflows/execute-quick-analysis", response_model=WorkflowExecutionResponse)
async def execute_quick_analysis(
    file: UploadFile = File(...),
    user_instructions: str = Form("Perform quick data analysis on the uploaded file")
):
    """Execute the Quick Data Analysis workflow with file upload"""
    
    try:
        print(f"Starting Quick Analysis workflow for file: {file.filename}")
        
        # Read file content (for demo we'll just validate it exists)
        file_content = await file.read()
        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
        # Start workflow execution in background
        workflow_name = f"Quick Analysis - {file.filename}"
        
        # For demo, simulate the execution
        execution = await simulate_workflow_execution(workflow_name, file.filename)
        
        return WorkflowExecutionResponse(
            success=True,
            workflow_id=execution['id'],
            message=f"Quick Analysis workflow for '{file.filename}' completed successfully",
            status=execution['status']
        )
        
    except Exception as e:
        print(f"Quick Analysis workflow failed: {e}")
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
    
    if execution['status'] != 'completed':
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
            'message': 'Quick Data Analysis workflow completed successfully',
            'steps_completed': len(execution['steps']),
            'sessions_created': [step['session_id'] for step in execution['steps']]
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
    print("🚀 Starting Quick Workflow API Test Server...")
    print("📡 Available endpoints:")
    print("   GET  http://127.0.0.1:8000/")
    print("   GET  http://127.0.0.1:8000/docs")
    print("   GET  http://127.0.0.1:8000/api/workflows/health")
    print("   GET  http://127.0.0.1:8000/api/workflows/templates")
    print("   POST http://127.0.0.1:8000/api/workflows/execute-quick-analysis")
    print("   GET  http://127.0.0.1:8000/api/workflows/{id}/status")
    print("   GET  http://127.0.0.1:8000/api/workflows/{id}/results")
    print("   GET  http://127.0.0.1:8000/api/workflows/")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
