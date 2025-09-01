"""
uAgent Client for backend integration with AI Data Science Platform agents

This client handles communication with 6 uAgents running on ports 8004-8009:
- 8004: Data Cleaning Agent 
- 8005: Data Loader Agent
- 8006: Data Visualization Agent  
- 8007: Feature Engineering Agent
- 8008: H2O ML Training Agent
- 8009: ML Prediction Agent
"""

import aiohttp
import asyncio
from typing import Dict, Any, Optional
from loguru import logger


class UAgentClient:
    """Backend client for communicating with uAgent REST endpoints"""
    
    def __init__(self, host: str = "127.0.0.1"):
        self.host = host
        self.agent_ports = {
            'loading': 8005,
            'cleaning': 8004,
            'visualization': 8006,
            'engineering': 8007,
            'training': 8008,
            'prediction': 8009,
        }
        self.base_urls = {
            agent_type: f"http://{host}:{port}"
            for agent_type, port in self.agent_ports.items()
        }
    
    async def _request(self, agent_type: str, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make HTTP request to uAgent"""
        url = f"{self.base_urls[agent_type]}{endpoint}"
        
        timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes for long operations
        
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                if data is not None:
                    async with session.post(url, json=data) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            raise Exception(f"Request to {url} failed: {response.status} {error_text}")
                        return await response.json()
                else:
                    async with session.get(url) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            raise Exception(f"Request to {url} failed: {response.status} {error_text}")
                        return await response.json()
        except asyncio.TimeoutError:
            raise Exception(f"Request to {url} timed out")
        except Exception as e:
            logger.error(f"Request to {url} failed: {e}")
            raise
    
    # Data Loader Agent (8005)
    async def load_file(self, agent_type: str, filename: str, file_content: str, user_instructions: str = None) -> Dict[str, Any]:
        """Load file using data loader agent"""
        data = {
            'filename': filename,
            'file_content': file_content,
            'user_instructions': user_instructions or 'Load and analyze the uploaded file'
        }
        return await self._request('loading', '/load-file', data)
    
    async def get_session_data(self, agent_type: str, session_id: str) -> Dict[str, Any]:
        """Get session data from any agent"""
        if agent_type == 'loading':
            return await self._request('loading', '/get-artifacts', {'session_id': session_id})
        elif agent_type == 'cleaning':
            return await self._request('cleaning', '/get-cleaned-data', {'session_id': session_id})
        elif agent_type == 'engineering':
            return await self._request('engineering', '/get-session-data', {'session_id': session_id})
        elif agent_type == 'training':
            return await self._request('training', '/get-original-data', {'session_id': session_id})
        elif agent_type == 'prediction':
            return await self._request('prediction', '/get-prediction-results', {'session_id': session_id})
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    # Data Cleaning Agent (8004)
    async def clean_data_from_session(self, session_id: str, user_instructions: str = None) -> Dict[str, Any]:
        """Clean data using existing session"""
        data = {
            'session_id': session_id,
            'user_instructions': user_instructions or 'Clean the data using recommended steps'
        }
        return await self._request('cleaning', '/clean-from-session', data)
    
    async def clean_csv_data(self, filename: str, file_content: str, user_instructions: str = None) -> Dict[str, Any]:
        """Clean CSV data directly"""
        data = {
            'filename': filename,
            'file_content': file_content,
            'user_instructions': user_instructions or 'Clean the data using recommended steps'
        }
        return await self._request('cleaning', '/clean-csv', data)
    
    async def get_session_code(self, agent_type: str, session_id: str) -> Dict[str, Any]:
        """Get generated code from agent session"""
        if agent_type == 'cleaning':
            return await self._request('cleaning', '/get-cleaning-function', {'session_id': session_id})
        elif agent_type == 'visualization':
            return await self._request('visualization', '/get-visualization-function', {'session_id': session_id})
        elif agent_type == 'engineering':
            return await self._request('engineering', '/get-engineering-function', {'session_id': session_id})
        elif agent_type == 'training':
            return await self._request('training', '/get-training-function', {'session_id': session_id})
        else:
            raise ValueError(f"Code not available for agent type: {agent_type}")
    
    # Data Visualization Agent (8006)
    async def create_chart_from_session(self, session_id: str, user_instructions: str = None) -> Dict[str, Any]:
        """Create chart using existing session"""
        data = {
            'session_id': session_id,
            'user_instructions': user_instructions or 'Create comprehensive visualizations to understand the data'
        }
        return await self._request('visualization', '/create-chart', data)
    
    async def create_chart_csv(self, filename: str, file_content: str, user_instructions: str = None) -> Dict[str, Any]:
        """Create chart from CSV data directly"""
        data = {
            'filename': filename,
            'file_content': file_content,
            'user_instructions': user_instructions or 'Create comprehensive visualizations to understand the data'
        }
        return await self._request('visualization', '/create-chart-csv', data)
    
    async def get_session_chart(self, agent_type: str, session_id: str) -> Dict[str, Any]:
        """Get chart from visualization session"""
        if agent_type != 'visualization':
            raise ValueError('Charts only available for visualization agent')
        return await self._request('visualization', '/get-plotly-graph', {'session_id': session_id})
    
    # Feature Engineering Agent (8007)
    async def engineer_features_from_session(self, session_id: str, target_variable: str, user_instructions: str = None) -> Dict[str, Any]:
        """Engineer features using existing session"""
        data = {
            'session_id': session_id,
            'target_variable': target_variable,
            'user_instructions': user_instructions or 'Engineer features for machine learning'
        }
        return await self._request('engineering', '/engineer-features', data)
    
    async def engineer_features_csv(self, filename: str, file_content: str, target_variable: str, user_instructions: str = None) -> Dict[str, Any]:
        """Engineer features from CSV data directly"""
        data = {
            'filename': filename,
            'file_content': file_content,
            'target_variable': target_variable,
            'user_instructions': user_instructions or 'Engineer features for machine learning'
        }
        return await self._request('engineering', '/engineer-features-csv', data)
    
    # ML Training Agent (8008)
    async def train_model_from_session(self, session_id: str, target_variable: str, user_instructions: str = None, max_runtime_secs: int = 120) -> Dict[str, Any]:
        """Train model using existing session"""
        data = {
            'session_id': session_id,
            'target_variable': target_variable,
            'user_instructions': user_instructions or 'Train machine learning models',
            'max_runtime_secs': max_runtime_secs
        }
        return await self._request('training', '/train-model', data)
    
    async def train_model_csv(self, filename: str, file_content: str, target_variable: str, user_instructions: str = None, max_runtime_secs: int = 120) -> Dict[str, Any]:
        """Train model from CSV data directly"""
        data = {
            'filename': filename,
            'file_content': file_content,
            'target_variable': target_variable,
            'user_instructions': user_instructions or 'Train machine learning models',
            'max_runtime_secs': max_runtime_secs
        }
        return await self._request('training', '/train-model-csv', data)
    
    async def get_session_leaderboard(self, agent_type: str, session_id: str) -> Dict[str, Any]:
        """Get leaderboard from training session"""
        if agent_type != 'training':
            raise ValueError('Leaderboard only available for training agent')
        return await self._request('training', '/get-leaderboard', {'session_id': session_id})
    
    async def get_model_path(self, agent_type: str, session_id: str) -> Dict[str, Any]:
        """Get model path from training session"""
        if agent_type != 'training':
            raise ValueError('Model path only available for training agent')
        return await self._request('training', '/get-model-path', {'session_id': session_id})
    
    # ML Prediction Agent (8009)
    async def predict_batch(self, model_session_id: str, filename: str = None, file_content: str = None) -> Dict[str, Any]:
        """Make batch predictions"""
        data = {
            'model_session_id': model_session_id,
            'filename': filename,
            'file_content': file_content
        }
        return await self._request('prediction', '/predict-batch', data)
    
    async def predict_single(self, model_session_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make single prediction"""
        data = {
            'model_session_id': model_session_id,
            'input_data': input_data
        }
        return await self._request('prediction', '/predict-single', data)
    
    # Health checks
    async def check_agent_health(self, agent_type: str) -> Dict[str, Any]:
        """Check health of specific agent"""
        try:
            return await self._request(agent_type, '/health')
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    async def check_all_agents_health(self) -> Dict[str, Dict[str, Any]]:
        """Check health of all agents"""
        health_checks = {}
        
        for agent_type in self.agent_ports.keys():
            health_checks[agent_type] = await self.check_agent_health(agent_type)
        
        return health_checks
