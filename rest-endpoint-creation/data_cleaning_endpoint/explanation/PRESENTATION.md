# Data Cleaning Endpoint - System Presentation (2 minutes)

## 🏗️ Architecture Overview

### Two-Component System
1. **Flask Frontend** (`frontend_app.py`) - Port 5000
2. **uAgent REST Server** (`data_cleaning_rest_agent.py`) - Port 8003

---

## 🔄 Connection Flow

### Frontend → uAgent Communication
```python
# frontend_app.py
AGENT_URL = "http://127.0.0.1:8003"
response = requests.post(f"{AGENT_URL}/clean-csv", json=payload)
```

### uAgent REST Endpoints
```python
# data_cleaning_rest_agent.py
@agent.on_rest_post("/clean-csv", CleanCsvRequest, CleanResponse)
async def handle_clean_csv(ctx: Context, req: CleanCsvRequest):
    # Process CSV cleaning request
```

---

## 🧠 Core Components

### `data_cleaning_rest_agent.py`
- **uAgent Definition**: Creates agent on port 8003
- **REST Endpoints**: 
  - `GET /health` - Health check
  - `POST /clean-csv` - Clean CSV files
  - `POST /clean-data` - Clean JSON data
- **LangChain Integration**: Calls `DataCleaningAgent` internally
- **JSON Serialization**: Handles pandas NaN values safely

### `frontend_app.py`
- **Web Interface**: Serves HTML upload form
- **File Processing**: Handles CSV uploads, base64 encoding
- **API Communication**: Makes HTTP requests to uAgent
- **Response Handling**: Displays cleaned data and results

---

## 🔧 Key Implementation Details

### Request Processing
1. User uploads CSV → Frontend encodes to base64
2. Frontend sends JSON payload to uAgent
3. uAgent decodes CSV, creates DataFrame
4. **LangChain Agent** processes data cleaning
5. Results returned as structured JSON

### LangChain Integration
```python
def _invoke_cleaning(df, instructions, max_retries):
    agent = DataCleaningAgent(model=llm)
    agent.invoke_agent(data_raw=df, user_instructions=instructions)
    cleaned_df = agent.get_data_cleaned()
    # Convert to JSON-safe format
    return structured_response
```

---

## ✅ How to Verify It's Working

### 1. Start the System
```bash
# Terminal 1: Start uAgent
cd rest-endpoint-creation/data_cleaning_endpoint/ && python data_cleaning_rest_agent.py

# Terminal 2: Start Frontend  
cd rest-endpoint-creation/data_cleaning_endpoint/ && python frontend_app.py
```

### 2. Health Check
```bash
curl http://127.0.0.1:8003/health
# Should return: {"status": "healthy", "agent": "data_cleaning_rest_uagent"}
```

### 3. Test Data Cleaning
```bash
curl -X POST http://127.0.0.1:8003/clean-data \
  -H "Content-Type: application/json" \
  -d '{"data": {"name": ["John", "Jane"], "age": [25, null]}, "user_instructions": "Fill missing values"}'
```

### 4. Web Interface Test
- Open `http://127.0.0.1:5000`
- Upload `test_sample_data.csv`
- Verify cleaning results display

### 5. Expected Success Indicators
- ✅ uAgent starts without errors
- ✅ Health endpoint responds
- ✅ CSV/data cleaning returns `"success": true`
- ✅ Web interface shows cleaned data table
- ✅ No JSON serialization errors in logs

---

## 🎯 Success Criteria

**System is working correctly when:**
- Both servers start successfully
- Health checks pass
- File uploads process without errors
- Cleaned data displays in web interface
- No NaN/serialization errors in console

**Ready for production use when all endpoints respond correctly via both cURL and web interface.**
