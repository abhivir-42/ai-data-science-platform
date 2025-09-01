"""
Workflow Execution Service for chaining AI Data Science agents
"""

import asyncio
import time
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from loguru import logger

from app.lib.uagent_client import UAgentClient


class WorkflowStatus(Enum):
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowStep:
    """Represents a single step in a workflow"""
    id: str
    agent_type: str  # 'loading', 'cleaning', 'visualization', etc.
    parameters: Dict[str, Any]
    status: WorkflowStatus = WorkflowStatus.PENDING
    session_id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time_seconds: Optional[float] = None


@dataclass
class WorkflowExecution:
    """Represents a complete workflow execution"""
    id: str
    name: str
    steps: List[WorkflowStep]
    status: WorkflowStatus = WorkflowStatus.PENDING
    created_at: float = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    total_execution_time: Optional[float] = None
    current_step_index: int = 0
    results: Dict[str, Any] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if self.results is None:
            self.results = {}


class WorkflowExecutionService:
    """Service for executing multi-agent workflows"""
    
    def __init__(self):
        self.executions: Dict[str, WorkflowExecution] = {}
        self.uagent_client = UAgentClient()
    
    async def execute_workflow(
        self, 
        workflow_name: str,
        steps: List[Dict[str, Any]],
        initial_data: Optional[Dict[str, Any]] = None
    ) -> WorkflowExecution:
        """
        Execute a workflow by chaining multiple agents
        
        Args:
            workflow_name: Name of the workflow
            steps: List of workflow steps with agent_type and parameters
            initial_data: Initial data to pass to first agent (e.g. uploaded file)
            
        Returns:
            WorkflowExecution object with execution status and results
        """
        
        # Create workflow execution
        execution_id = str(uuid.uuid4())
        workflow_steps = [
            WorkflowStep(
                id=str(uuid.uuid4()),
                agent_type=step['agent_type'],
                parameters=step.get('parameters', {})
            ) 
            for step in steps
        ]
        
        execution = WorkflowExecution(
            id=execution_id,
            name=workflow_name,
            steps=workflow_steps
        )
        
        self.executions[execution_id] = execution
        
        logger.info(f"Starting workflow execution: {workflow_name} (ID: {execution_id})")
        
        try:
            execution.status = WorkflowStatus.RUNNING
            execution.started_at = time.time()
            
            # Execute each step in sequence
            current_data = initial_data
            
            for i, step in enumerate(execution.steps):
                execution.current_step_index = i
                logger.info(f"Executing step {i+1}/{len(execution.steps)}: {step.agent_type}")
                
                step.status = WorkflowStatus.RUNNING
                step_start_time = time.time()
                
                try:
                    # Execute the agent step
                    step_result = await self._execute_agent_step(step, current_data)
                    
                    step.result = step_result
                    step.session_id = step_result.get('session_id')
                    step.status = WorkflowStatus.COMPLETED
                    step.execution_time_seconds = time.time() - step_start_time
                    
                    # Update current data for next step
                    current_data = await self._prepare_data_for_next_step(step, step_result)
                    
                    logger.info(f"Step {i+1} completed successfully")
                    
                except Exception as e:
                    step.status = WorkflowStatus.FAILED
                    step.error = str(e)
                    step.execution_time_seconds = time.time() - step_start_time
                    
                    logger.error(f"Step {i+1} failed: {e}")
                    
                    # Fail the entire workflow
                    execution.status = WorkflowStatus.FAILED
                    execution.completed_at = time.time()
                    execution.total_execution_time = execution.completed_at - execution.started_at
                    
                    return execution
            
            # All steps completed successfully
            execution.status = WorkflowStatus.COMPLETED
            execution.completed_at = time.time() 
            execution.total_execution_time = execution.completed_at - execution.started_at
            
            # Gather final results
            execution.results = await self._gather_workflow_results(execution)
            
            logger.info(f"Workflow {workflow_name} completed successfully in {execution.total_execution_time:.2f}s")
            
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.completed_at = time.time()
            if execution.started_at:
                execution.total_execution_time = execution.completed_at - execution.started_at
            
            logger.error(f"Workflow {workflow_name} failed: {e}")
        
        return execution
    
    async def _execute_agent_step(self, step: WorkflowStep, input_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a single agent step"""
        
        agent_type = step.agent_type
        parameters = step.parameters.copy()
        
        # Add input data to parameters
        if input_data:
            parameters.update(input_data)
        
        logger.debug(f"Executing {agent_type} agent with parameters: {parameters}")
        
        # Execute based on agent type
        if agent_type == 'loading':
            return await self._execute_loading_agent(parameters)
        elif agent_type == 'cleaning':
            return await self._execute_cleaning_agent(parameters)
        elif agent_type == 'visualization':
            return await self._execute_visualization_agent(parameters)
        elif agent_type == 'engineering':
            return await self._execute_engineering_agent(parameters)
        elif agent_type == 'training':
            return await self._execute_training_agent(parameters)
        elif agent_type == 'prediction':
            return await self._execute_prediction_agent(parameters)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    async def _execute_loading_agent(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data loading agent"""
        
        if 'filename' in parameters and 'file_content' in parameters:
            # Load from uploaded file
            response = await self.uagent_client.load_file(
                agent_type='loading',
                filename=parameters['filename'],
                file_content=parameters['file_content'],
                user_instructions=parameters.get('user_instructions', 'Load and analyze the uploaded file')
            )
        else:
            raise ValueError("Loading agent requires 'filename' and 'file_content' parameters")
        
        if not response.get('success'):
            raise Exception(f"Data loading failed: {response.get('error', 'Unknown error')}")
        
        return {
            'session_id': response['session_id'],
            'agent_type': 'loading',
            'execution_time_seconds': response.get('execution_time_seconds'),
            'message': response.get('message')
        }
    
    async def _execute_cleaning_agent(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data cleaning agent"""
        
        # Use session_id from previous step if available
        if 'session_id' in parameters:
            response = await self.uagent_client.clean_data_from_session(
                session_id=parameters['session_id'],
                user_instructions=parameters.get('user_instructions', 'Clean the data using recommended steps')
            )
        elif 'filename' in parameters and 'file_content' in parameters:
            response = await self.uagent_client.clean_csv_data(
                filename=parameters['filename'],
                file_content=parameters['file_content'],
                user_instructions=parameters.get('user_instructions', 'Clean the data using recommended steps')
            )
        else:
            raise ValueError("Cleaning agent requires either 'session_id' or 'filename'+'file_content' parameters")
        
        if not response.get('success'):
            raise Exception(f"Data cleaning failed: {response.get('error', 'Unknown error')}")
        
        return {
            'session_id': response['session_id'],
            'agent_type': 'cleaning',
            'execution_time_seconds': response.get('execution_time_seconds'),
            'message': response.get('message')
        }
    
    async def _execute_visualization_agent(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data visualization agent"""
        
        if 'session_id' in parameters:
            response = await self.uagent_client.create_chart_from_session(
                session_id=parameters['session_id'],
                user_instructions=parameters.get('user_instructions', 'Create comprehensive visualizations to understand the data')
            )
        elif 'filename' in parameters and 'file_content' in parameters:
            response = await self.uagent_client.create_chart_csv(
                filename=parameters['filename'],
                file_content=parameters['file_content'],
                user_instructions=parameters.get('user_instructions', 'Create comprehensive visualizations to understand the data')
            )
        else:
            raise ValueError("Visualization agent requires either 'session_id' or 'filename'+'file_content' parameters")
        
        if not response.get('success'):
            raise Exception(f"Data visualization failed: {response.get('error', 'Unknown error')}")
        
        return {
            'session_id': response['session_id'],
            'agent_type': 'visualization',
            'execution_time_seconds': response.get('execution_time_seconds'),
            'message': response.get('message')
        }
    
    async def _execute_engineering_agent(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute feature engineering agent"""
        
        target_variable = parameters.get('target_variable', '')
        if not target_variable:
            raise ValueError("Feature engineering requires 'target_variable' parameter")
        
        if 'session_id' in parameters:
            response = await self.uagent_client.engineer_features_from_session(
                session_id=parameters['session_id'],
                target_variable=target_variable,
                user_instructions=parameters.get('user_instructions', 'Engineer features for machine learning')
            )
        elif 'filename' in parameters and 'file_content' in parameters:
            response = await self.uagent_client.engineer_features_csv(
                filename=parameters['filename'],
                file_content=parameters['file_content'],
                target_variable=target_variable,
                user_instructions=parameters.get('user_instructions', 'Engineer features for machine learning')
            )
        else:
            raise ValueError("Engineering agent requires either 'session_id' or 'filename'+'file_content' parameters")
        
        if not response.get('success'):
            raise Exception(f"Feature engineering failed: {response.get('error', 'Unknown error')}")
        
        return {
            'session_id': response['session_id'],
            'agent_type': 'engineering',
            'execution_time_seconds': response.get('execution_time_seconds'),
            'message': response.get('message')
        }
    
    async def _execute_training_agent(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute ML training agent"""
        
        target_variable = parameters.get('target_variable', '')
        if not target_variable:
            raise ValueError("Training agent requires 'target_variable' parameter")
        
        if 'session_id' in parameters:
            response = await self.uagent_client.train_model_from_session(
                session_id=parameters['session_id'],
                target_variable=target_variable,
                user_instructions=parameters.get('user_instructions', 'Train machine learning models'),
                max_runtime_secs=parameters.get('max_runtime_secs', 120)
            )
        elif 'filename' in parameters and 'file_content' in parameters:
            response = await self.uagent_client.train_model_csv(
                filename=parameters['filename'],
                file_content=parameters['file_content'],
                target_variable=target_variable,
                user_instructions=parameters.get('user_instructions', 'Train machine learning models'),
                max_runtime_secs=parameters.get('max_runtime_secs', 120)
            )
        else:
            raise ValueError("Training agent requires either 'session_id' or 'filename'+'file_content' parameters")
        
        if not response.get('success'):
            raise Exception(f"Model training failed: {response.get('error', 'Unknown error')}")
        
        return {
            'session_id': response['session_id'],
            'agent_type': 'training',
            'execution_time_seconds': response.get('execution_time_seconds'),
            'message': response.get('message')
        }
    
    async def _execute_prediction_agent(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute ML prediction agent"""
        
        if 'model_session_id' not in parameters:
            raise ValueError("Prediction agent requires 'model_session_id' parameter")
        
        response = await self.uagent_client.predict_batch(
            model_session_id=parameters['model_session_id'],
            filename=parameters.get('filename'),
            file_content=parameters.get('file_content')
        )
        
        if not response.get('success'):
            raise Exception(f"Model prediction failed: {response.get('error', 'Unknown error')}")
        
        return {
            'session_id': response['session_id'],
            'agent_type': 'prediction',
            'execution_time_seconds': response.get('execution_time_seconds'),
            'message': response.get('message')
        }
    
    async def _prepare_data_for_next_step(self, completed_step: WorkflowStep, step_result: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data from completed step for the next step"""
        
        # The key insight: pass session_id from previous step to next step
        # This allows agents to chain their work together
        
        return {
            'session_id': step_result.get('session_id'),
            'previous_agent': completed_step.agent_type,
            'previous_result': step_result
        }
    
    async def _gather_workflow_results(self, execution: WorkflowExecution) -> Dict[str, Any]:
        """Gather comprehensive results from all workflow steps"""
        
        results = {
            'workflow_id': execution.id,
            'workflow_name': execution.name,
            'total_execution_time': execution.total_execution_time,
            'steps_completed': len([s for s in execution.steps if s.status == WorkflowStatus.COMPLETED]),
            'total_steps': len(execution.steps),
            'step_results': []
        }
        
        for step in execution.steps:
            step_summary = {
                'step_id': step.id,
                'agent_type': step.agent_type,
                'status': step.status.value,
                'session_id': step.session_id,
                'execution_time_seconds': step.execution_time_seconds,
                'error': step.error
            }
            
            if step.session_id and step.status == WorkflowStatus.COMPLETED:
                try:
                    # Gather additional results from each agent
                    if step.agent_type == 'loading':
                        step_summary['data_info'] = await self.uagent_client.get_session_data('loading', step.session_id)
                    elif step.agent_type == 'cleaning':
                        step_summary['cleaned_data'] = await self.uagent_client.get_session_data('cleaning', step.session_id)
                        step_summary['cleaning_code'] = await self.uagent_client.get_session_code('cleaning', step.session_id)
                    elif step.agent_type == 'visualization':
                        step_summary['chart'] = await self.uagent_client.get_session_chart('visualization', step.session_id)
                        step_summary['viz_code'] = await self.uagent_client.get_session_code('visualization', step.session_id)
                    elif step.agent_type == 'training':
                        step_summary['leaderboard'] = await self.uagent_client.get_session_leaderboard('training', step.session_id)
                        step_summary['model_path'] = await self.uagent_client.get_model_path('training', step.session_id)
                except Exception as e:
                    logger.warning(f"Failed to gather additional results for step {step.agent_type}: {e}")
            
            results['step_results'].append(step_summary)
        
        return results
    
    def get_execution(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get workflow execution by ID"""
        return self.executions.get(execution_id)
    
    def list_executions(self) -> List[WorkflowExecution]:
        """List all workflow executions"""
        return list(self.executions.values())
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get execution status"""
        execution = self.executions.get(execution_id)
        if not execution:
            return None
        
        return {
            'id': execution.id,
            'name': execution.name,
            'status': execution.status.value,
            'current_step': execution.current_step_index,
            'total_steps': len(execution.steps),
            'progress_percentage': (execution.current_step_index / len(execution.steps)) * 100 if execution.steps else 0,
            'execution_time': execution.total_execution_time or (time.time() - execution.started_at if execution.started_at else 0),
            'steps': [
                {
                    'agent_type': step.agent_type,
                    'status': step.status.value,
                    'session_id': step.session_id,
                    'error': step.error
                }
                for step in execution.steps
            ]
        }


# Global service instance
workflow_execution_service = WorkflowExecutionService()
