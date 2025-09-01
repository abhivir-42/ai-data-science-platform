"""
Minimal test script for workflow API endpoints
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Create a minimal FastAPI app just for testing workflow endpoints
app = FastAPI(title="Workflow API Test", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import just the workflow router
try:
    from app.api.workflows import router as workflow_router
    app.include_router(workflow_router, prefix="/api", tags=["workflows"])
    print("✅ Workflow router imported successfully")
except Exception as e:
    print(f"❌ Failed to import workflow router: {e}")
    # Create a simple test endpoint instead
    @app.get("/api/workflows/health")
    async def test_health():
        return {"status": "ok", "message": "Workflow API test endpoint"}

@app.get("/")
async def root():
    return {"message": "Workflow API Test Server", "docs": "/docs"}

if __name__ == "__main__":
    print("🚀 Starting Workflow API Test Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
