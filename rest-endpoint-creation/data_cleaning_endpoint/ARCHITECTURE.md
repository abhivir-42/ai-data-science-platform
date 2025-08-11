# Data Cleaning Endpoint - System Architecture & Connections

## Overview
This document explains the **system connections** and architecture pattern used in the data cleaning endpoint, following the same pattern as `frontend_integration/`.

## System Connection Pattern

### The Universal uAgent REST Pattern
Both `frontend_integration/` and `data_cleaning_endpoint/` follow this identical architecture:

```
Flask Frontend (port 5000) → HTTP POST → uAgent REST Server (port 8xxx) → Business Logic
```

## System 1: `frontend_integration/` (Reference Pattern)

### Connection Flow:
1. **Flask Frontend** (`frontend_app.py:5000`)
   - Renders HTML forms
   - Handles user input
   - Makes HTTP POST requests to uAgents

2. **uAgent REST Servers**
   - `product_search_agent.py:8001` - Product search uAgent
   - `product_info_agent.py:8002` - Product info uAgent

3. **Connection Method:**
   ```python
   # Frontend calls uAgent
   response = requests.post(f"{AGENTS['search']}/search", json=payload)
   
   # uAgent handles request
   @search_agent.on_rest_post("/search", SearchRequest, SearchResponse)
   async def search_products(ctx: Context, req: SearchRequest) -> SearchResponse:
   ```

## System 2: `data_cleaning_endpoint/` (Our Implementation)

### Connection Flow:
1. **Flask Frontend** (`frontend_app.py:5000`)
   - File upload interface
   - Makes HTTP POST requests to data cleaning uAgent

2. **uAgent REST Server**
   - `data_cleaning_rest_agent.py:8003` - Data cleaning uAgent

3. **Connection Method:**
   ```python
   # Frontend calls uAgent (same pattern as System 1)
   response = requests.post(f"{AGENT_URL}/clean-csv", json=payload)
   
   # uAgent handles request (same pattern as System 1)
   @agent.on_rest_post("/clean-csv", CleanCsvRequest, CleanResponse)
   async def handle_clean_csv(ctx: Context, req: CleanCsvRequest) -> CleanResponse:
   ```

## Under the Hood: Complete Flow

### 1. User Interaction
- User opens browser → `http://127.0.0.1:5000`
- Flask serves HTML form from `templates/index.html`

### 2. Frontend Processing
- User uploads CSV file
- Flask `frontend_app.py`:
  - Encodes file to base64
  - Creates JSON payload
  - Makes HTTP POST to uAgent endpoint

### 3. uAgent Processing
- `data_cleaning_rest_agent.py` receives HTTP POST
- uAgent framework deserializes JSON to Pydantic models
- Handler function `handle_clean_csv()` executes:
  - Decodes base64 → CSV text
  - Converts to pandas DataFrame
  - **Calls LangChain agent** (`_invoke_cleaning()`)
  - Returns structured response

### 4. LangChain Integration (Unique to System 2)
- `_invoke_cleaning()` function:
  - Creates `DataCleaningAgent` instance
  - Invokes LangGraph workflow
  - Extracts cleaned data and steps
  - Converts NaN values to JSON-safe format

### 5. Response Flow
- uAgent returns JSON response
- Flask receives response
- Frontend displays results to user

## Key Architecture Benefits

### 1. Separation of Concerns
- **Frontend**: UI/UX, file handling, display
- **uAgent**: REST API, request/response handling
- **LangChain**: Business logic, AI processing

### 2. Standardized Pattern
- Same HTTP POST pattern as `frontend_integration/`
- Same Pydantic model validation
- Same error handling approach

### 3. Scalability
- uAgent can be deployed independently
- Frontend can call multiple uAgents
- Easy to add new endpoints

## How to Follow This Pattern

### For Any New uAgent REST Service:

1. **Create uAgent with REST endpoints:**
   ```python
   from uagents import Agent, Context, Model
   
   agent = Agent(name="your_agent", port=8xxx)
   
   @agent.on_rest_post("/your-endpoint", YourRequest, YourResponse)
   async def handle_request(ctx: Context, req: YourRequest) -> YourResponse:
       # Your business logic here
       return YourResponse(...)
   ```

2. **Create Flask frontend:**
   ```python
   AGENT_URL = "http://127.0.0.1:8xxx"
   
   @app.route('/your-action', methods=['POST'])
   def your_action():
       response = requests.post(f"{AGENT_URL}/your-endpoint", json=payload)
       return jsonify(response.json())
   ```

3. **Define Pydantic models:**
   ```python
   class YourRequest(Model):
       field1: str
       field2: Optional[int] = None
   
   class YourResponse(Model):
       success: bool
       data: Optional[Dict] = None
       error: Optional[str] = None
   ```

## Summary

Both systems use **identical connection patterns**:
- Flask frontend makes HTTP requests
- uAgent REST endpoints handle requests
- Structured request/response with Pydantic models
- Business logic encapsulated in uAgent handlers

The only difference is the **business logic layer**:
- System 1: External API calls (OpenFoodFacts)
- System 2: Internal LangChain agent calls

This pattern is **reusable** for any uAgent-based REST service.
