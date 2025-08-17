# 🎯 AI Data Science Platform - uAgent REST Endpoints Implementation Plan

## 🚨 CRITICAL: Fetch.ai uAgent Pattern (NOT FastAPI)

This plan implements **standalone Fetch.ai uAgent REST endpoints** following the established pattern from `rest-endpoint-creation/data_cleaning_endpoint/data_cleaning_rest_agent.py`.

## 🏗️ Architecture: Standalone uAgents

Each agent is a **standalone uAgent** that:
- Uses `from uagents import Agent, Context, Model`
- Uses `@agent.on_rest_get` and `@agent.on_rest_post` decorators  
- Runs independently with `agent.run()`
- Has its own dedicated port
- Follows the session-based pattern for rich result access

## 📁 Directory Structure

```
backend/app/api/uagents/
├── data_cleaning_rest_agent.py      # DataCleaningAgent uAgent (Port 8004) ✅
├── data_loader_rest_agent.py        # DataLoaderToolsAgent uAgent (Port 8005) ✅  
├── data_visualization_rest_agent.py # DataVisualisationAgent uAgent (Port 8006)
├── feature_engineering_rest_agent.py# FeatureEngineeringAgent uAgent (Port 8007)
├── h2o_ml_rest_agent.py             # H2OMLAgent uAgent (Port 8008)
└── ml_prediction_rest_agent.py      # MLPredictionAgent uAgent (Port 8009)
```

## 🔧 Agent Capabilities Analysis

### 1. DataCleaningAgent ✅ COMPLETED
**Port**: 8004
**Key Methods Exposed:**
- `invoke_agent()` → Execute cleaning
- `get_data_cleaned()` → Cleaned DataFrame
- `get_data_raw()` → Original DataFrame  
- `get_data_cleaner_function()` → Generated Python code
- `get_recommended_cleaning_steps()` → Cleaning recommendations
- `get_workflow_summary()` → Process summary
- `get_log_summary()` → Logging details
- `get_response()` → Full response dictionary

**uAgent REST Endpoints:**
- `POST /clean-data` - Clean dataset with instructions
- `POST /clean-csv` - Clean CSV file (base64 encoded)
- `GET /session/{session_id}/cleaned-data` - Get cleaned data from session
- `GET /session/{session_id}/original-data` - Get original data from session
- `GET /session/{session_id}/cleaning-function` - Get generated cleaning code
- `GET /session/{session_id}/cleaning-steps` - Get recommended cleaning steps
- `GET /session/{session_id}/workflow-summary` - Get workflow summary
- `GET /session/{session_id}/logs` - Get execution logs
- `GET /session/{session_id}/full-response` - Get complete response
- `GET /health` - Health check
- `DELETE /session/{session_id}` - Delete session

### 2. DataLoaderToolsAgent ✅ COMPLETED
**Port**: 8005
**Key Methods Exposed:**
- `invoke_agent()` → Execute loading
- `get_artifacts(as_dataframe=True)` → Loaded data
- `get_ai_message()` → AI response about loading
- `get_tool_calls()` → Tools that were used
- `get_internal_messages()` → Detailed execution log

**uAgent REST Endpoints:**
- `POST /load-file` - Load file from path
- `POST /load-directory` - Load multiple files from directory
- `POST /extract-pdf` - Extract data from PDF
- `GET /session/{session_id}/artifacts` - Get loaded data artifacts
- `GET /session/{session_id}/ai-message` - Get AI response about loading
- `GET /session/{session_id}/tool-calls` - Get tools that were used
- `GET /session/{session_id}/internal-messages` - Get detailed execution log
- `GET /session/{session_id}/full-response` - Get complete response
- `GET /supported-formats` - Get supported file formats
- `GET /health` - Health check
- `DELETE /session/{session_id}` - Delete session

### 3. DataVisualisationAgent
**Port**: 8006
**Key Methods to Expose:**
- `invoke_agent()` → Execute visualization
- `get_plotly_graph()` → Plotly chart dictionary
- `get_data_visualization_function()` → Generated visualization code
- `get_recommended_visualization_steps()` → Chart recommendations
- `get_workflow_summary()` → Process summary
- `get_log_summary()` → Logging details
- `get_data_raw()` → Original data used for visualization

**uAgent REST Endpoints to Create:**
- `POST /create-chart` - Generate visualization from data
- `POST /create-chart-csv` - Generate chart from CSV file
- `GET /session/{session_id}/plotly-graph` - Get generated Plotly chart
- `GET /session/{session_id}/visualization-function` - Get generated visualization code
- `GET /session/{session_id}/visualization-steps` - Get chart recommendations
- `GET /session/{session_id}/workflow-summary` - Get workflow summary
- `GET /session/{session_id}/logs` - Get execution logs
- `GET /session/{session_id}/original-data` - Get original data
- `GET /session/{session_id}/full-response` - Get complete response
- `GET /health` - Health check
- `DELETE /session/{session_id}` - Delete session

### 4. FeatureEngineeringAgent
**Port**: 8007
**Key Methods to Expose:**
- `invoke_agent()` → Execute feature engineering
- `get_data_engineered()` → Feature-engineered DataFrame
- `get_data_raw()` → Original DataFrame
- `get_feature_engineer_function()` → Generated feature engineering code
- `get_recommended_feature_engineering_steps()` → Feature engineering recommendations
- `get_workflow_summary()` → Process summary
- `get_log_summary()` → Logging details

**uAgent REST Endpoints to Create:**
- `POST /engineer-features` - Engineer features for dataset
- `POST /engineer-features-csv` - Engineer features from CSV
- `GET /session/{session_id}/engineered-data` - Get feature-engineered data
- `GET /session/{session_id}/original-data` - Get original data
- `GET /session/{session_id}/engineering-function` - Get generated feature engineering code
- `GET /session/{session_id}/engineering-steps` - Get feature engineering recommendations
- `GET /session/{session_id}/workflow-summary` - Get workflow summary
- `GET /session/{session_id}/logs` - Get execution logs
- `GET /session/{session_id}/full-response` - Get complete response
- `GET /health` - Health check
- `DELETE /session/{session_id}` - Delete session

### 5. H2OMLAgent
**Port**: 8008
**Key Methods to Expose:**
- `invoke_agent()` → Execute ML training
- `get_leaderboard()` → H2O AutoML leaderboard
- `get_best_model_id()` → Best model identifier
- `get_model_path()` → Saved model file path # TODO: HOW TO ENABLE END USER TO BE ABLE TO DOWNLOAD IT?
- `get_h2o_train_function()` → Generated H2O training code
- `get_recommended_ml_steps()` → ML recommendations
- `get_workflow_summary()` → Training summary
- `get_log_summary()` → Training logs
- `get_data_raw()` → Original training data

**uAgent REST Endpoints to Create:**
- `POST /train-model` - Train ML model with H2O AutoML
- `POST /train-model-csv` - Train model from CSV file
- `GET /session/{session_id}/leaderboard` - Get H2O AutoML leaderboard
- `GET /session/{session_id}/best-model-id` - Get best model identifier
- `GET /session/{session_id}/model-path` - Get saved model file path
- `GET /session/{session_id}/training-function` - Get generated H2O training code
- `GET /session/{session_id}/ml-steps` - Get ML recommendations
- `GET /session/{session_id}/workflow-summary` - Get training summary
- `GET /session/{session_id}/logs` - Get training logs
- `GET /session/{session_id}/original-data` - Get original training data
- `GET /session/{session_id}/full-response` - Get complete response
- `GET /health` - Health check
- `DELETE /session/{session_id}` - Delete session

### 6. MLPredictionAgent
**Port**: 8009
**Key Methods to Expose:**
- `predict_single()` → Single prediction
- `predict_batch()` → Batch predictions
- `analyze_model()` → Answer questions about trained model
- `load_model()` → Load trained model

**uAgent REST Endpoints to Create:**
- `POST /predict-single` - Make single prediction
- `POST /predict-batch` - Make batch predictions from CSV URL
- `POST /analyze-model` - Ask questions about trained model
- `POST /load-model` - Load model from path or H2O cluster
- `GET /session/{session_id}/prediction-results` - Get prediction results
- `GET /session/{session_id}/batch-results` - Get batch prediction results
- `GET /session/{session_id}/model-analysis` - Get model analysis results
- `GET /health` - Health check
- `DELETE /session/{session_id}` - Delete session

## 🗄️ Session Management Strategy

### Why Session-Based Approach?
- **Rich Functionality**: Each agent execution creates a wealth of data (original data, processed data, generated code, recommendations, logs)
- **Multiple Access Points**: Frontend needs to access different aspects of results at different times
- **Performance**: Avoid re-running expensive operations to get different views of the same result
- **User Experience**: Allow users to explore results step-by-step

### Session Storage Pattern
```python
# In-memory session store (can be replaced with Redis later)
class SessionStore:
    def __init__(self):
        self._sessions = {}
    
    def create_session(self, agent_instance, metadata=None):
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = {
            "agent": agent_instance,
            "created_at": time.time(),
            "metadata": metadata or {}
        }
        return session_id
```

## 🔄 uAgent Implementation Template

```python
#!/usr/bin/env python3
"""
{Agent Name} uAgent REST API.
"""

import os
import sys
import uuid
import time
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
import pandas as pd
import numpy as np

def _ensure_project_root_on_path():
    """Add project root and backend to sys.path for imports"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, "..", "..", "..", "..")
    backend_dir = os.path.join(project_root, "backend")
    
    for path in [project_root, backend_dir]:
        abs_path = os.path.abspath(path)
        if abs_path not in sys.path:
            sys.path.insert(0, abs_path)

def _load_environment():
    """Load environment variables from .env files"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for path in (
        os.path.join(current_dir, ".env"),
        os.path.join(current_dir, "..", "..", "..", "..", ".env"),
    ):
        if os.path.exists(path):
            load_dotenv(dotenv_path=path)
            break

_ensure_project_root_on_path()
_load_environment()

from uagents import Agent, Context, Model
from uagents.setup import fund_agent_if_low
from langchain_openai import ChatOpenAI
from app.agents import YourAgent

# Session management, JSON serialization utilities, etc.

agent = Agent(
    name="{agent_name}_rest_uagent",
    port=800X,  # Unique port
    seed="{agent_name}_rest_uagent_secret_seed",
    endpoint=["http://127.0.0.1:800X/submit"],
)

fund_agent_if_low(agent.wallet.address())

# Pydantic models for requests/responses

@agent.on_rest_post("/your-endpoint", YourRequest, YourResponse)
async def your_endpoint(ctx: Context, req: YourRequest) -> YourResponse:
    # Implementation following session pattern

if __name__ == "__main__":
    print(f"🚀 Starting {agent.name}...")
    print("📡 Available endpoints:")
    print("   GET  http://127.0.0.1:800X/health")
    print("   POST http://127.0.0.1:800X/your-endpoint")
    print("🚀 Agent starting...")
    agent.run()
```

## 🚨 Critical JSON Serialization Rules

### ALWAYS Handle Pandas Types
```python
def make_json_serializable(data):
    """Convert pandas/numpy types to JSON-serializable Python types"""
    if isinstance(data, dict):
        return {k: make_json_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [make_json_serializable(item) for item in data]
    elif pd.isna(data):
        return None
    elif hasattr(data, 'isoformat'):  # datetime/Timestamp
        return data.isoformat()
    elif isinstance(data, (np.integer, np.floating)):
        return data.item() if not np.isnan(data) else None
    elif hasattr(data, 'item') and hasattr(data, 'dtype'):
        return data.item()
    else:
        return data
```

## 🔧 Port Allocation Strategy

| Agent | Port | Status | Description |
|-------|------|--------|-------------|
| Data Cleaning | 8004 | ✅ Implemented | Clean datasets, generate cleaning code |
| Data Loader | 8005 | ✅ Implemented | Load files, directories, PDFs |
| Data Visualization | 8006 | ⏳ Pending | Create charts, generate visualization code |
| Feature Engineering | 8007 | ⏳ Pending | Engineer features, generate feature code |
| H2O ML Training | 8008 | ⏳ Pending | Train ML models, get leaderboards |
| ML Prediction | 8009 | ⏳ Pending | Make predictions, analyze models |

## 🎯 Key Benefits for Frontend Development

### Comprehensive Access to Agent Capabilities
1. **Generated Code Access**: Users can view and download Python functions
2. **Step-by-Step Exploration**: Access different aspects of results progressively
3. **Educational Value**: Users learn data science by seeing AI's approach
4. **Debugging Support**: Full access to logs, tool calls, internal messages
5. **Session Management**: Resume work, compare results, handle multiple datasets

### Rich Frontend Possibilities
```javascript
// Example frontend workflow
const cleaningResponse = await fetch('/api/uagents/data-cleaning/clean-csv', {
    method: 'POST', body: csvData
});
const { session_id } = await cleaningResponse.json();

// Progressive result exploration
const originalData = await fetch(`/session/${session_id}/original-data`);
const recommendations = await fetch(`/session/${session_id}/cleaning-steps`);
const cleanedData = await fetch(`/session/${session_id}/cleaned-data`);
const generatedCode = await fetch(`/session/${session_id}/cleaning-function`);
```

## 🚀 Implementation Order

1. ✅ **DataCleaningAgent** (Port 8004) - COMPLETED
2. ✅ **DataLoaderToolsAgent** (Port 8005) - COMPLETED  
3. ⏳ **DataVisualisationAgent** (Port 8006) - NEXT
4. ⏳ **FeatureEngineeringAgent** (Port 8007)
5. ⏳ **H2OMLAgent** (Port 8008)
6. ⏳ **MLPredictionAgent** (Port 8009)

## 🔄 Testing Strategy

Each uAgent should be tested with:
1. **Health check**: `curl http://127.0.0.1:800X/health`
2. **Main operations**: POST requests for primary functionality using test sample dummy data
3. **Session access**: GET requests for all session-based endpoints
4. **JSON serialization**: Verify pandas/numpy types are handled correctly

Lives of people depend on this being implemented correctly! 🚨