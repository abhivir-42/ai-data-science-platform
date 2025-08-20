# uAgent Session Endpoint Troubleshooting Guide

## Problem Description

When implementing uAgent REST endpoints with session-based result access, developers may encounter issues where session endpoints return "not found" errors even when sessions are successfully created and stored.

## Root Cause

The issue stems from **uAgent path parameter handling**. uAgents framework has limitations with GET endpoints that use path parameters like `/session/{session_id}/data`. These endpoints often fail to properly route requests, resulting in "not found" errors.

## Symptoms

1. ✅ Main operation endpoints work (e.g., `POST /clean-data`, `POST /load-file`)
2. ✅ Session creation works and returns valid `session_id`
3. ✅ Session store properly creates and stores sessions
4. ❌ Session-based GET endpoints fail (e.g., `GET /session/{id}/data`)
5. ❌ Frontend tabs show "not available" or buttons are not clickable

## Solution: Convert to POST Endpoints

### Before (Problematic)
```python
@agent.on_rest_get("/session/{session_id}/data", DataResponse)
async def get_session_data(ctx: Context, session_id: str) -> DataResponse:
    # This often fails with "not found" errors
```

### After (Working)
```python
@agent.on_rest_post("/get-session-data", SessionRequest, DataResponse)
async def get_session_data(ctx: Context, req: SessionRequest) -> DataResponse:
    session = session_store.get_session(req.session_id)
    # This works reliably
```

## Implementation Pattern

### 1. Define Session Request Model
```python
class SessionRequest(Model):
    session_id: str
```

### 2. Convert All Session Endpoints
```python
# Instead of GET /session/{id}/cleaning-function
@agent.on_rest_post("/get-cleaning-function", SessionRequest, CodeResponse)

# Instead of GET /session/{id}/cleaning-steps  
@agent.on_rest_post("/get-cleaning-steps", SessionRequest, GenericResponse)

# Instead of GET /session/{id}/logs
@agent.on_rest_post("/get-logs", SessionRequest, GenericResponse)
```

### 3. Update Frontend Client
```typescript
// Instead of GET requests
async getSessionCode(sessionId: string): Promise<CodeResponse> {
  return this.request<CodeResponse>('/get-cleaning-function', { 
    session_id: sessionId 
  });
}
```

## Complete Endpoint Mapping

### Data Cleaning Agent (Port 8004)
- ✅ `POST /clean-data` - Create session
- ✅ `POST /get-cleaned-data` - Get cleaned data
- ✅ `POST /get-cleaning-function` - Get generated code
- ✅ `POST /get-cleaning-steps` - Get recommendations
- ✅ `POST /get-logs` - Get execution logs

### Data Loader Agent (Port 8005) - Needs Fix
- ✅ `POST /load-file` - Create session
- ✅ `POST /get-artifacts` - Get loaded data
- ❌ `GET /session/{id}/data` - Should be `POST /get-session-data`
- ❌ `GET /session/{id}/ai-message` - Should be `POST /get-ai-message`
- ❌ `GET /session/{id}/tool-calls` - Should be `POST /get-tool-calls`

## Frontend Response Format Handling

### Backend Response Format
```json
{
  "success": true,
  "generated_code": "# Python code here...",
  "data": "Recommendations text here...",
  "error": null
}
```

### Frontend Client Handling
```typescript
// Extract data from wrapped response
const response = await this.request<{success: boolean, generated_code?: string}>('/get-cleaning-function', { session_id: sessionId });
return { 
  code: response.generated_code,
  generated_code: response.generated_code
};
```

## Testing Checklist

1. **Session Creation**: Test main operation endpoint
2. **Session Storage**: Verify session exists in store
3. **Data Retrieval**: Test POST session endpoints
4. **Frontend Integration**: Verify tabs are clickable
5. **Content Display**: Check formatting and readability

## Common Pitfalls

1. **Mixed Endpoint Types**: Don't mix GET path parameters with POST body parameters
2. **Response Format Mismatch**: Ensure frontend expects backend response structure
3. **Session Store Issues**: Verify in-memory session persistence
4. **Caching Problems**: Clear browser cache when testing changes

## Debugging Tips

1. **Add Session Store Logging**:
```python
print(f"[SessionStore] Get session {session_id}: {'Found' if session else 'Not found'}")
```

2. **Test Endpoints Directly**:
```bash
curl -X POST http://127.0.0.1:8004/get-cleaning-function \
  -H "Content-Type: application/json" \
  -d '{"session_id": "your-session-id"}'
```

3. **Check Frontend Network Tab**: Look for failed requests in browser dev tools

## Best Practices

1. **Consistent Pattern**: Use POST endpoints with request body for all session operations
2. **Clear Naming**: Use descriptive endpoint names (e.g., `/get-cleaning-function`)
3. **Error Handling**: Return consistent error responses
4. **Documentation**: Update endpoint documentation when changing patterns
5. **Testing**: Test both backend endpoints and frontend integration

## Migration Guide

For existing agents with broken session endpoints:

1. **Identify Broken Endpoints**: Find all `GET /session/{id}/*` endpoints
2. **Convert to POST**: Change to `POST /get-*` pattern
3. **Update Frontend Client**: Modify request methods
4. **Test Thoroughly**: Verify all tabs work correctly
5. **Update Documentation**: Reflect new endpoint patterns

This pattern ensures reliable session-based data access across all uAgent implementations.
