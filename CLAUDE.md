# CLAUDE.md - AI Data Science Platform

## Project Description
Full-stack AI data science platform with 6 LangChain/LangGraph agents for automated data science workflows. Personal portfolio project deployed at ml.abhivir.com.

## Tech Stack
- **Backend**: Python 3.10, FastAPI, LangChain, LangGraph, OpenAI GPT-4o-mini
- **Frontend**: Next.js 14, React 18, TypeScript, Tailwind CSS, Zustand, Plotly.js
- **ML**: H2O AutoML (Java), scikit-learn, XGBoost
- **Infra**: PostgreSQL, Redis, Celery, MLflow, Docker (7 services)
- **Current branch**: feature/dockerization

## Key Commands
```bash
# Backend
cd backend && pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Frontend
cd frontend && npm install && npm run dev

# Docker (full stack - 7 services)
docker-compose up --build

# Tests
cd backend && pytest
cd frontend && npm run type-check
```

## Architecture
Single FastAPI backend serving all agent endpoints. No microservice split.
- `/api/health` - Health checks
- `/api/agents` - Agent CRUD and execution
- `/api/agents/loading/*` - Data loader endpoints
- `/api/agents/cleaning/*` - Data cleaning endpoints
- `/api/agents/visualization/*` - Visualization endpoints
- `/api/agents/engineering/*` - Feature engineering endpoints
- `/api/agents/training/*` - ML training endpoints
- `/api/agents/prediction/*` - ML prediction endpoints
- `/api/data` - File upload/processing
- `/api/workflows` - Multi-agent workflow orchestration
- `/api/auth` - Authentication

## Agents (6 core + supervisor)
1. **Data Loader** - CSV, Excel, JSON, PDF ingestion
2. **Data Cleaning** - Missing values, outliers, duplicates
3. **Feature Engineering** - Feature creation, encoding
4. **Data Visualization** - Plotly chart generation
5. **ML Training** - H2O AutoML training
6. **ML Prediction** - Predictions with trained models
7. **Supervisor** - Orchestrates multi-agent workflows

## Important Notes
- Fetch AI / uagents code has been REMOVED. Do NOT add uagents dependencies.
- The core agent logic in `/backend/app/agents/` uses LangChain/LangGraph.
- Agent routes are in `/backend/app/api/agent_routes/`.
- Frontend agent client is in `/frontend/lib/uagent-client.ts` (renamed to AgentClient internally).
- This is a PUBLIC repo - never commit secrets or API keys.
- .env must be in .gitignore and never committed.

## File Structure Guidelines
- Backend code: `/backend/app/`
- Frontend code: `/frontend/`
- Docker config: root `docker-compose.yml`
- No files in repo root except config files
