"""
Workflow Execution Service for chaining AI Data Science agents
"""

import asyncio
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from loguru import logger
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.lib.uagent_client import UAgentClient
from app.core.database import database_manager
from app.models.session import WorkflowExecution as WorkflowExecutionModel, WorkflowStatus


# WorkflowStatus enum imported from models


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
    """Service for executing multi-agent workflows with database persistence"""
    
    def __init__(self, user_id: Optional[str] = None):
        self.user_id = user_id
        self.uagent_client = UAgentClient(user_id=user_id)
        # Note: Removed in-memory executions dict - now using database
    
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
            asdict(WorkflowStep(
                id=str(uuid.uuid4()),
                agent_type=step['agent_type'],
                parameters=step.get('parameters', {})
            ))
            for step in steps
        ]
        
        # Create database record
        execution_model = WorkflowExecutionModel(
            id=execution_id,
            name=workflow_name,
            steps=workflow_steps,
            status=WorkflowStatus.PENDING
        )
        
        async with database_manager.async_session_maker() as db_session:
            db_session.add(execution_model)
            await db_session.commit()
            await db_session.refresh(execution_model)
        
        # Convert to dataclass for compatibility
        execution = WorkflowExecution(
            id=execution_model.id,
            name=execution_model.name,
            steps=[WorkflowStep(**step) for step in execution_model.steps],
            status=WorkflowStatus(execution_model.status),
            created_at=execution_model.created_at.timestamp() if execution_model.created_at else time.time(),
            started_at=execution_model.started_at.timestamp() if execution_model.started_at else None,
            completed_at=execution_model.completed_at.timestamp() if execution_model.completed_at else None,
            total_execution_time=execution_model.total_execution_time,
            current_step_index=execution_model.current_step_index,
            results=execution_model.results or {}
        )
        
        logger.info(f"Starting workflow execution: {workflow_name} (ID: {execution_id})")
        
        # Execute workflow in background
        asyncio.create_task(self._execute_workflow_background(execution))
        
        return execution
    
    async def start_workflow_async(
        self,
        workflow_name: str,
        steps: List[Dict[str, Any]],
        initial_data: Dict[str, Any] = None
    ) -> WorkflowExecution:
        """Start a workflow execution asynchronously and return immediately"""
        execution = await self._create_workflow_execution(workflow_name, steps, initial_data)
        
        # Execute workflow in background
        asyncio.create_task(self._execute_workflow_background(execution))
        
        return execution
    
    async def _create_workflow_execution(
        self,
        workflow_name: str,
        steps: List[Dict[str, Any]],
        initial_data: Dict[str, Any] = None
    ) -> WorkflowExecution:
        """Create a workflow execution record"""
        if initial_data is None:
            initial_data = {}
            
        execution_id = str(uuid.uuid4())
        
        # Convert steps to WorkflowStep objects
        workflow_steps = []
        for step_data in steps:
            step = WorkflowStep(
                id=str(uuid.uuid4()),
                agent_type=step_data['agent_type'],
                parameters=step_data.get('parameters', {}),
                status=WorkflowStatus.PENDING
            )
            workflow_steps.append(step)
        
        # Create execution record in database
        async with database_manager.async_session_maker() as db_session:
            execution_model = WorkflowExecutionModel(
                id=execution_id,
                name=workflow_name,
                status=WorkflowStatus.PENDING,
                steps=[asdict(step) for step in workflow_steps],
                current_step_index=0,
                total_execution_time=0.0,
                results=initial_data
            )
            
            db_session.add(execution_model)
            await db_session.commit()
        
        # Create WorkflowExecution object
        execution = WorkflowExecution(
            id=execution_model.id,
            name=execution_model.name,
            steps=workflow_steps,
            status=WorkflowStatus(execution_model.status),
            created_at=execution_model.created_at.timestamp() if execution_model.created_at else time.time(),
            started_at=None,
            completed_at=None,
            total_execution_time=0.0,
            current_step_index=0,
            results=initial_data
        )
        
        return execution
    
    def _extract_chart_type(self, plotly_graph: Any) -> Optional[str]:
        """Extract chart type from plotly graph data"""
        try:
            if isinstance(plotly_graph, dict) and 'data' in plotly_graph:
                if plotly_graph['data'] and len(plotly_graph['data']) > 0:
                    return plotly_graph['data'][0].get('type', 'unknown')
        except:
            pass
        return None
    
    async def _execute_workflow_background(self, execution: WorkflowExecution):
        """Execute workflow in the background"""
        try:
            execution.status = WorkflowStatus.RUNNING
            execution.started_at = time.time()
            
            # Update database with running status
            async with database_manager.async_session_maker() as db_session:
                await db_session.execute(
                    update(WorkflowExecutionModel)
                    .where(WorkflowExecutionModel.id == execution.id)
                    .values(
                        status=WorkflowStatus.RUNNING,
                        started_at=datetime.utcfromtimestamp(execution.started_at)
                    )
                )
                await db_session.commit()
            
            # Execute each step in sequence
            current_data = execution.results
            
            for i, step in enumerate(execution.steps):
                execution.current_step_index = i
                logger.info(f"Executing step {i+1}/{len(execution.steps)}: {step.agent_type}")
                
                step.status = WorkflowStatus.RUNNING
                step_start_time = time.time()
                
                # Update database with current step status
                execution.steps[i] = step  # Update the step in the execution
                async with database_manager.async_session_maker() as db_session:
                    steps_data = [asdict(s) for s in execution.steps]
                    await db_session.execute(
                        update(WorkflowExecutionModel)
                        .where(WorkflowExecutionModel.id == execution.id)
                        .values(
                            current_step_index=i,
                            steps=steps_data
                        )
                    )
                    await db_session.commit()
                
                try:
                    # Execute the agent step
                    step_result = await self._execute_agent_step(step, current_data)
                    
                    step.result = step_result
                    step.session_id = step_result.get('session_id')
                    step.status = WorkflowStatus.COMPLETED
                    step.execution_time_seconds = time.time() - step_start_time
                    
                    # Update database with completed step
                    execution.steps[i] = step
                    async with database_manager.async_session_maker() as db_session:
                        steps_data = [asdict(s) for s in execution.steps]
                        await db_session.execute(
                            update(WorkflowExecutionModel)
                            .where(WorkflowExecutionModel.id == execution.id)
                            .values(steps=steps_data)
                        )
                        await db_session.commit()
                    
                    # Update current data for next step
                    current_data = await self._prepare_data_for_next_step(step, step_result)
                    
                    logger.info(f"Step {i+1} completed successfully")
                    
                except Exception as e:
                    step.status = WorkflowStatus.FAILED
                    step.error = str(e)
                    step.execution_time_seconds = time.time() - step_start_time
                    
                    logger.error(f"Step {i+1} failed: {e}")
                    
                    # Update database with failed step and workflow
                    execution.status = WorkflowStatus.FAILED
                    execution.completed_at = time.time()
                    execution.total_execution_time = execution.completed_at - execution.started_at
                    execution.steps[i] = step
                    
                    async with database_manager.async_session_maker() as db_session:
                        steps_data = [asdict(s) for s in execution.steps]
                        await db_session.execute(
                            update(WorkflowExecutionModel)
                            .where(WorkflowExecutionModel.id == execution.id)
                            .values(
                                status=WorkflowStatus.FAILED,
                                completed_at=datetime.utcfromtimestamp(execution.completed_at),
                                total_execution_time=execution.total_execution_time,
                                steps=steps_data
                            )
                        )
                        await db_session.commit()
                    
                    return execution
            
            # All steps completed successfully
            execution.status = WorkflowStatus.COMPLETED
            execution.completed_at = time.time() 
            execution.total_execution_time = execution.completed_at - execution.started_at
            
            # Gather final results
            execution.results = await self._gather_workflow_results(execution)
            
            # Update database with completion status
            async with database_manager.async_session_maker() as db_session:
                await db_session.execute(
                    update(WorkflowExecutionModel)
                    .where(WorkflowExecutionModel.id == execution.id)
                    .values(
                        status=WorkflowStatus.COMPLETED,
                        completed_at=datetime.utcfromtimestamp(execution.completed_at),
                        total_execution_time=execution.total_execution_time,
                        results=execution.results
                    )
                )
                await db_session.commit()
            
            logger.info(f"Workflow {execution.name} completed successfully in {execution.total_execution_time:.2f}s")
            
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.completed_at = time.time()
            if execution.started_at:
                execution.total_execution_time = execution.completed_at - execution.started_at
            
            # Update database with failure status
            async with database_manager.async_session_maker() as db_session:
                await db_session.execute(
                    update(WorkflowExecutionModel)
                    .where(WorkflowExecutionModel.id == execution.id)
                    .values(
                        status=WorkflowStatus.FAILED,
                        completed_at=datetime.utcfromtimestamp(execution.completed_at),
                        total_execution_time=execution.total_execution_time
                    )
                )
                await db_session.commit()
            
            logger.error(f"Workflow {execution.name} failed: {e}")
        
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
        
        # Get the chart data and code from the session
        session_id = response['session_id']
        
        # Get chart data directly from session (bypass broken GET endpoints)
        try:
            session_data = await session_service.get_session(session_id)
            if session_data and session_data.get('agent'):
                viz_agent = session_data['agent']
                response_data = viz_agent.get_response()
                
                # Extract chart data
                chart_data = None
                logger.info(f"Response data keys: {list(response_data.keys()) if response_data else 'None'}")
                if response_data and 'plotly_graph' in response_data:
                    plotly_graph = response_data['plotly_graph']
                    logger.info(f"Plotly graph type: {type(plotly_graph)}, not None: {plotly_graph is not None}")
                    if plotly_graph:
                        chart_data = {
                            'success': True,
                            'plotly_chart': plotly_graph,
                            'chart_type': self._extract_chart_type(plotly_graph)
                        }
                        logger.info(f"Chart data created successfully")
                    else:
                        logger.warning("Plotly graph is None or falsy")
                else:
                    logger.warning(f"No plotly_graph in response data. Available keys: {list(response_data.keys()) if response_data else 'None'}")
                
                # Extract visualization code  
                viz_code = None
                if response_data and 'data_visualization_function' in response_data:
                    viz_code = {
                        'success': True,
                        'generated_code': response_data.get('data_visualization_function'),
                        'code_explanation': None
                    }
                    
                logger.info(f"Retrieved chart data directly from session: chart={chart_data is not None}, code={viz_code is not None}")
            else:
                logger.warning(f"Could not retrieve session data for {session_id}")
                chart_data = None
                viz_code = None
        except Exception as e:
            logger.warning(f"Could not retrieve chart data from session: {e}")
            chart_data = None
            viz_code = None
        
        return {
            'session_id': session_id,
            'agent_type': 'visualization',
            'execution_time_seconds': response.get('execution_time_seconds'),
            'message': response.get('message'),
            'chart': {
                'success': True,
                'message': 'Chart retrieved successfully',
                'plotly_chart': chart_data.get('plotly_chart') if chart_data else None,
                'chart_type': chart_data.get('chart_type') if chart_data else None,
                'error': None
            },
            'viz_code': {
                'success': bool(viz_code),
                'message': 'Code retrieved successfully' if viz_code else 'Code not available',
                'generated_code': viz_code.get('generated_code') if viz_code else None,
                'code_explanation': viz_code.get('code_explanation') if viz_code else None,
                'error': None
            }
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
    
    async def get_execution(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get workflow execution by ID from database"""
        try:
            async with database_manager.async_session_maker() as db_session:
                stmt = select(WorkflowExecutionModel).where(WorkflowExecutionModel.id == execution_id)
                result = await db_session.execute(stmt)
                execution_model = result.scalar_one_or_none()

                if not execution_model:
                    return None

                # Convert to dataclass for compatibility
                steps = []
                for step_data in execution_model.steps:
                    try:
                        # Handle different step formats that might exist
                        if isinstance(step_data, dict):
                            steps.append(WorkflowStep(**step_data))
                        else:
                            # If step_data is already a WorkflowStep, convert it
                            steps.append(step_data)
                    except Exception as step_error:
                        logger.warning(f"Failed to parse step data: {step_error}, skipping step")
                        continue

                return WorkflowExecution(
                    id=execution_model.id,
                    name=execution_model.name,
                    steps=steps,
                    status=WorkflowStatus(execution_model.status),
                    created_at=execution_model.created_at.timestamp() if execution_model.created_at else time.time(),
                    started_at=execution_model.started_at.timestamp() if execution_model.started_at else None,
                    completed_at=execution_model.completed_at.timestamp() if execution_model.completed_at else None,
                    total_execution_time=execution_model.total_execution_time,
                    current_step_index=execution_model.current_step_index,
                    results=execution_model.results or {}
                )
        except Exception as e:
            logger.error(f"Failed to get execution {execution_id}: {e}")
            return None
    
    async def list_executions(self) -> List[WorkflowExecution]:
        """List all workflow executions from database"""
        try:
            async with database_manager.async_session_maker() as db_session:
                stmt = select(WorkflowExecutionModel).order_by(WorkflowExecutionModel.created_at.desc())
                result = await db_session.execute(stmt)
                execution_models = result.scalars().all()

                executions = []
                for model in execution_models:
                    try:
                        # Parse steps safely
                        steps = []
                        for step_data in model.steps:
                            try:
                                if isinstance(step_data, dict):
                                    steps.append(WorkflowStep(**step_data))
                                else:
                                    steps.append(step_data)
                            except Exception as step_error:
                                logger.warning(f"Failed to parse step for execution {model.id}: {step_error}")
                                continue

                        executions.append(WorkflowExecution(
                            id=model.id,
                            name=model.name,
                            steps=steps,
                            status=WorkflowStatus(model.status),
                            created_at=model.created_at.timestamp() if model.created_at else time.time(),
                            started_at=model.started_at.timestamp() if model.started_at else None,
                            completed_at=model.completed_at.timestamp() if model.completed_at else None,
                            total_execution_time=model.total_execution_time,
                            current_step_index=model.current_step_index,
                            results=model.results or {}
                        ))
                    except Exception as exec_error:
                        logger.error(f"Failed to parse execution {model.id}: {exec_error}")
                        continue

                return executions
        except Exception as e:
            logger.error(f"Failed to list executions: {e}")
            return []
    
    async def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get execution status from database"""
        execution = await self.get_execution(execution_id)
        if not execution:
            return None
        
        return {
            'id': execution.id,
            'name': execution.name,
            'status': execution.status.value if hasattr(execution.status, 'value') else str(execution.status),
            'current_step': execution.current_step_index,
            'total_steps': len(execution.steps),
            'progress_percentage': (execution.current_step_index / len(execution.steps)) * 100 if execution.steps else 0,
            'execution_time': execution.total_execution_time or (time.time() - execution.started_at if execution.started_at else 0),
            'steps': [
                {
                    'agent_type': step.agent_type,
                    'status': step.status.value if hasattr(step.status, 'value') else str(step.status),
                    'session_id': step.session_id,
                    'error': step.error
                }
                for step in execution.steps
            ]
        }


# Global service instance
workflow_execution_service = WorkflowExecutionService()
