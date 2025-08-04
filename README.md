# AI Data Science Platform

A comprehensive, production-ready platform that democratizes data science by providing both technical and non-technical users with powerful AI agents for data analysis, machine learning, and insights generation.

## 🚀 Features

- **8+ Specialized AI Agents**: Data loading, cleaning, wrangling, feature engineering, visualization, and ML
- **Natural Language Interface**: Ask questions about your data in plain English
- **Visual Pipeline Builder**: Drag-and-drop interface for building complex data workflows
- **Real-time Processing**: WebSocket-based progress tracking and updates
- **Interactive Visualizations**: Plotly-powered charts and dashboards
- **AutoML Integration**: H2O AutoML with MLflow experiment tracking
- **Multi-format Support**: CSV, Excel, JSON, Parquet, PDF data sources

## 🏗️ Architecture

```
Frontend (Next.js)  ←→  Backend (FastAPI)  ←→  Agents (Python)
                              ↕
                         Database (PostgreSQL)
                              ↕
                      Task Queue (Celery + Redis)
                              ↕
                        ML Tracking (MLflow)
```

## 📋 Prerequisites

- Python 3.9+
- Node.js 18+
- Docker & Docker Compose
- Git

## 🚀 Quick Start

### 1. Clone and Setup

```bash
cd ai-data-science-platform
cp env.example .env
# Edit .env with your API keys from fetch/ai-data-science/.env
```

### 2. Development with Docker (Recommended)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

Services will be available at:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Celery Flower**: http://localhost:5555
- **MLflow**: http://localhost:5000

### 3. Manual Development Setup

#### Backend Setup

```bash
# Activate the existing virtual environment
source ../ai-ds-venv/bin/activate

# Install dependencies
cd backend
pip install -r requirements.txt

# Set up database
alembic upgrade head

# Start development server
uvicorn app.main:app --reload --port 8000
```

#### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

## 📚 Agent Integration

The platform integrates all agents from `fetch/ai-data-science`:

- **DataLoaderToolsAgent**: Multi-format data ingestion
- **DataCleaningAgent**: Automated data preprocessing
- **DataWranglingAgent**: Data transformation and reshaping
- **FeatureEngineeringAgent**: Feature creation and encoding
- **DataVisualisationAgent**: Interactive chart generation
- **H2OMLAgent**: AutoML model training
- **SupervisorAgent**: Workflow orchestration
- **DataAnalysisAgent**: Enhanced workflow management

## 🔧 API Endpoints

### Core Endpoints
```
GET    /api/agents                    # List available agents
GET    /api/agents/{id}/schema        # Get agent parameter schema
POST   /api/agents/{id}/execute       # Execute agent with parameters
GET    /api/jobs/{jobId}/status       # Get job execution status
DELETE /api/jobs/{jobId}              # Cancel running job
```

### Data Management
```
POST   /api/data/upload               # Upload datasets
GET    /api/data/preview/{id}         # Preview dataset
POST   /api/data/validate             # Validate data quality
GET    /api/data/summary/{id}         # Get data summary
```

## 🧪 Testing

```bash
# Backend tests
cd backend
pytest

# Frontend tests
cd frontend
npm test

# Integration tests
docker-compose -f docker-compose.test.yml up --abort-on-container-exit
```

## 📦 Production Deployment

### Environment Variables

Copy production values to `.env`:

```bash
# Production database
DATABASE_URL=postgresql://user:pass@host:5432/dbname

# Production Redis
REDIS_URL=redis://host:6379/0

# API Keys
OPENAI_API_KEY=your_production_key
AGENTVERSE_API_TOKEN=your_production_token

# Security
SECRET_KEY=your_secure_secret_key
ENVIRONMENT=production
DEBUG=false
```

### Docker Production Build

```bash
# Build production images
docker-compose -f docker-compose.prod.yml build

# Deploy
docker-compose -f docker-compose.prod.yml up -d
```

## 🔍 Monitoring & Observability

- **Health Checks**: `/health` endpoint
- **Metrics**: Prometheus metrics at `:8001/metrics`
- **Logs**: Structured logging with correlation IDs
- **Tracing**: Request tracing through the system

## 🤝 Contributing

1. Follow the implementation plan in `implementation-plan.md`
2. Use conventional commits
3. Add tests for new features
4. Update documentation

## 📝 Development Notes

### Agent Development

When adding new agents:
1. Create agent class in `backend/app/agents/`
2. Add API endpoint in `backend/app/api/`
3. Register in agent registry
4. Add frontend form generation
5. Update tests and documentation

### Database Migrations

```bash
# Create new migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `ai-ds-venv` is activated
2. **Database Connection**: Check PostgreSQL is running
3. **Redis Connection**: Verify Redis service is up
4. **API Keys**: Confirm keys are set in `.env`

### Logs

```bash
# Backend logs
docker-compose logs backend

# Celery worker logs
docker-compose logs celery-worker

# All services
docker-compose logs -f
```

## 📞 Support

- Check `implementation-plan.md` for development roadmap
- Review existing agents in `fetch/ai-data-science/src/agents/`
- Use existing test files as examples

## 📄 License

This project builds upon the existing `fetch/ai-data-science` codebase and maintains the same licensing terms. 