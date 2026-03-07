# AI Data Science Platform

A full-stack platform that democratizes data science by providing AI-powered agents for automated data analysis, machine learning, and insights generation. Built with LangChain/LangGraph agents, H2O AutoML, and a modern React frontend.

## Features

- **6 Specialized AI Agents**: Data loading, cleaning, visualization, feature engineering, ML training, and prediction
- **Natural Language Interface**: Describe what you want in plain English
- **Visual Pipeline Builder**: Drag-and-drop workflow orchestration
- **Real-time Processing**: WebSocket-based progress tracking
- **Interactive Visualizations**: Plotly-powered charts
- **AutoML Integration**: H2O AutoML with MLflow experiment tracking
- **Multi-format Support**: CSV, Excel, JSON, Parquet, PDF

## Architecture

```
Frontend (Next.js 14)  <-->  Backend (FastAPI)  <-->  LangChain Agents
        |                          |
        |                    PostgreSQL + Redis
        |                          |
        |                   Celery Workers
        |                          |
        +-- Plotly Charts    MLflow Tracking
```

### Tech Stack

- **Frontend**: Next.js 14, React 18, TypeScript, Tailwind CSS, Zustand, Plotly.js
- **Backend**: Python 3.10, FastAPI, LangChain, LangGraph, OpenAI GPT-4o-mini
- **ML**: H2O AutoML, scikit-learn, XGBoost
- **Infrastructure**: PostgreSQL, Redis, Celery, MLflow, Docker

## Quick Start

### Prerequisites

- Docker & Docker Compose
- OpenAI API key

### Docker Setup (Recommended)

```bash
git clone https://github.com/yourusername/ai-data-science-platform.git
cd ai-data-science-platform

# Configure environment
cp env.example .env
# Edit .env and set your OPENAI_API_KEY

# Start all services (7 containers)
docker-compose up -d

# View logs
docker-compose logs -f
```

Services:
- **Frontend**: http://localhost:8001
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **MLflow**: http://localhost:8002
- **Celery Flower**: http://localhost:8003

### Local Development

#### Backend

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Start PostgreSQL and Redis (or use Docker for just those)
uvicorn app.main:app --reload --port 8000
```

#### Frontend

```bash
cd frontend
npm install
npm run dev
```

## AI Agents

All agents use LangChain/LangGraph with GPT-4o-mini and run inside the single FastAPI backend:

| Agent | Description | Key Endpoints |
|-------|-------------|---------------|
| **Data Loader** | Multi-format data ingestion (CSV, Excel, JSON, PDF) | `/api/agents/loading/*` |
| **Data Cleaning** | Automated preprocessing, missing values, outliers | `/api/agents/cleaning/*` |
| **Data Visualization** | Interactive Plotly chart generation | `/api/agents/visualization/*` |
| **Feature Engineering** | Feature creation, encoding, transformation | `/api/agents/engineering/*` |
| **ML Training** | H2O AutoML model training with leaderboard | `/api/agents/training/*` |
| **ML Prediction** | Single/batch predictions with trained models | `/api/agents/prediction/*` |

## API Endpoints

### Agent Operations
```
POST   /api/agents/loading/load-file              # Upload and load data
POST   /api/agents/cleaning/clean-csv              # Clean data from CSV
POST   /api/agents/cleaning/clean-from-session     # Clean data from previous session
POST   /api/agents/visualization/create-chart-direct  # Generate chart
POST   /api/agents/engineering/engineer-features-csv   # Engineer features
POST   /api/agents/training/train-model-csv        # Train ML model
POST   /api/agents/prediction/predict-single       # Make prediction
```

### Core API
```
GET    /api/health                     # Health check
GET    /api/agents                     # List available agents
POST   /api/agents/{id}/execute        # Execute agent
POST   /api/data/upload                # Upload datasets
GET    /api/workflows/templates        # List workflow templates
POST   /api/workflows/execute          # Execute workflow
```

## Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-proj-...

# Database (defaults work with Docker)
DATABASE_URL=postgresql://postgres:password@localhost:5432/ai_data_science_platform

# Redis (defaults work with Docker)
REDIS_URL=redis://localhost:6379/0

# Optional
MLFLOW_TRACKING_URI=http://localhost:5000
SECRET_KEY=your-secret-key
ENVIRONMENT=development
DEBUG=true
```

## Project Structure

```
ai-data-science-platform/
├── backend/
│   ├── app/
│   │   ├── agents/          # Core LangChain/LangGraph agent classes
│   │   ├── api/
│   │   │   ├── agent_routes/ # Per-agent REST endpoints
│   │   │   ├── agents.py     # Agent CRUD/execute
│   │   │   ├── workflows.py  # Workflow orchestration
│   │   │   └── ...
│   │   ├── core/            # Config, database, logging
│   │   ├── lib/             # Internal agent client
│   │   ├── services/        # Session, workflow execution
│   │   └── main.py          # FastAPI app
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── app/                 # Next.js pages
│   ├── components/          # React components
│   ├── lib/                 # Agent client, store
│   └── Dockerfile
├── docker-compose.yml       # 7 services
└── env.example
```

## License

MIT
