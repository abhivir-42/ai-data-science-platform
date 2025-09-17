# Visualization Chart Data Issue - Debugging Documentation

## 🎯 **Problem Statement**

**Issue**: Charts are not displaying in the frontend "View Results" section. Users see "Chart data not available" and "no chart" instead of the generated Plotly visualizations.

**Status**: 🔧 **IN PROGRESS** - Chart data exists in sessions but doesn't reach frontend results

---

## 🔍 **What We've Confirmed (Working)**

### ✅ **Chart Generation**
- Plotly charts ARE being created successfully by the visualization agent
- Chart data has correct structure: `{'data': [...], 'layout': {...}}`
- Chart data contains valid Plotly traces and layout information

### ✅ **Session Storage** 
- Chart data IS being stored in database sessions correctly
- Sessions can be retrieved and contain the `plotly_graph` field
- Direct session access via `SessionService.get_session()` works perfectly

### ✅ **Data Structure**
- Chart data has correct type: `<class 'dict'>`
- Chart data is truthy: `bool(plotly_graph) = True`
- Chart data contains expected keys: `['data', 'layout']`
- Chart data has valid traces: `len(plotly_graph['data']) = 4`

---

## ❌ **What's Broken**

### 🚨 **Workflow Integration**
- Chart data is NOT making it from sessions to workflow results
- Final API response shows `"plotly_chart": null` in all cases
- There's a disconnect between session data and the final API response

### 🚨 **Data Flow Issue**
- Chart data exists in sessions but gets lost during workflow execution
- The issue is NOT in chart generation or session storage
- The issue is in the workflow execution → API response pipeline

---

## 🔧 **Technical Investigation**

### **Debug Logs Analysis**

From the latest debug run (workflow `85d1304a-bf93-4b34-a241-4650f174d1d4`):

```
2025-09-17 15:59:05 | INFO | [CHART DEBUG] Response data keys: ['messages', 'user_instructions', 'recommended_steps', 'data_raw', 'plotly_graph', 'all_datasets_summary', 'data_visualization_function', 'data_visualization_function_path', 'data_visualization_function_file_name', 'data_visualization_function_name', 'data_visualization_error', 'max_retries', 'retry_count']
2025-09-17 15:59:05 | INFO | [CHART DEBUG] Plotly graph type: <class 'dict'>, not None: True
2025-09-17 15:59:05 | INFO | [CHART DEBUG] bool(plotly_graph): True
2025-09-17 15:59:05 | INFO | [CHART DEBUG] Chart data FORCE created: True
2025-09-17 15:59:05 | INFO | [CHART DEBUG] Chart data plotly_chart is None: False
2025-09-17 15:59:05 | INFO | [CHART DEBUG] About to return - chart_data is None: False
2025-09-17 15:59:05 | INFO | [CHART DEBUG] chart_data keys: ['success', 'plotly_chart', 'chart_type']
2025-09-17 15:59:05 | INFO | [CHART DEBUG] chart_data['plotly_chart'] is None: False
```

**Key Finding**: The debug logs show that chart data is being created correctly in the workflow execution, but the final API response still shows `"plotly_chart": null`.

### **Code Path Analysis**

The issue occurs in `backend/app/services/workflow_execution.py` in the `_execute_visualization_agent` method:

1. ✅ Session data is retrieved successfully
2. ✅ `plotly_graph` is extracted from response data
3. ✅ `chart_data` is created with the plotly data
4. ✅ Debug logs confirm data exists at this point
5. ❌ **BREAK POINT**: Data gets lost somewhere between this point and the final API response

---

## 🛠️ **Attempted Fixes**

### **Fix 1: SessionService Import Error**
- **Issue**: `name 'session_service' is not defined`
- **Fix**: Added `SessionService` import and initialized `self.session_service`
- **Status**: ✅ **FIXED**

### **Fix 2: uAgent GET Endpoints**
- **Issue**: uAgent GET endpoints (`/session/{id}/plotly-graph`) return 404
- **Root Cause**: uAgent framework limitation with path parameters
- **Fix**: Bypassed GET endpoints, used direct session access
- **Status**: ✅ **WORKING**

### **Fix 3: Response Key Mismatch**
- **Issue**: Looking for `chart_response.get('figure')` instead of `chart_response.get('plotly_chart')`
- **Fix**: Corrected key lookup
- **Status**: ✅ **FIXED**

### **Fix 4: Enhanced Debug Logging**
- **Issue**: No visibility into data flow
- **Fix**: Added extensive `[CHART DEBUG]` logging throughout the process
- **Status**: ✅ **IMPLEMENTED** - Reveals data exists but gets lost

---

## 🎯 **Current Status**

### **What's Working**
- Chart generation by visualization agent
- Session storage and retrieval
- Data structure and validation
- Workflow execution up to chart data creation

### **What's Broken**
- Chart data is not making it to the final workflow results
- API response shows `"plotly_chart": null` despite data existing in sessions

### **Next Steps Needed**
1. **Identify the exact break point** where chart data gets lost
2. **Trace the data flow** from `_execute_visualization_agent` to final API response
3. **Fix the data serialization/transmission** issue
4. **Test frontend display** once data reaches the API response

---

## 📁 **Key Files**

### **Primary Files**
- `backend/app/services/workflow_execution.py` - Main workflow execution logic
- `backend/app/services/session_service.py` - Session management
- `backend/app/api/uagents/data_visualization_rest_agent.py` - Visualization agent endpoints

### **Debug Files**
- `UAGENT_SESSION_ENDPOINT_TROUBLESHOOTING.md` - Documents uAgent GET endpoint limitations

### **Test Files**
- `test_final.csv` - Test data used for debugging

---

## 🔧 **Debugging Commands**

### **Test Workflow Execution**
```bash
curl -s -X POST http://localhost:8000/api/workflows/execute-quick-analysis \
  -F "file=@test_final.csv" \
  -F "user_instructions=Test chart data"
```

### **Check Workflow Results**
```bash
curl -s "http://localhost:8000/api/workflows/{workflow_id}/results" | grep -A 3 -B 3 "plotly_chart"
```

### **Direct Session Inspection**
```python
import asyncio
from app.services.session_service import SessionService

async def check_session(session_id):
    session_service = SessionService()
    session = await session_service.get_session(session_id)
    if session and session.get('agent'):
        viz_agent = session['agent']
        response_data = viz_agent.get_response()
        plotly_graph = response_data.get('plotly_graph')
        print(f"plotly_graph exists: {plotly_graph is not None}")
        print(f"plotly_graph type: {type(plotly_graph)}")
        print(f"plotly_graph keys: {list(plotly_graph.keys()) if plotly_graph else 'None'}")
```

---

## 🚨 **Critical Findings**

1. **Chart data EXISTS** in sessions and is accessible
2. **Chart data is CREATED** correctly in workflow execution
3. **Chart data is LOST** somewhere between creation and API response
4. **Debug logs confirm** data exists at the workflow execution level
5. **The issue is NOT** in chart generation, session storage, or data structure

---

## 📝 **Notes for Future Debugging**

- The issue is likely in the data serialization or transmission between workflow execution and API response
- Check if there are any size limits or serialization issues with large plotly data
- Verify that the chart data is being properly included in the final workflow results JSON
- Consider if there are any database field size limitations affecting the storage of plotly data

---

## 🎯 **Success Criteria**

The issue will be resolved when:
1. ✅ Chart data exists in sessions (already working)
2. ✅ Chart data is created in workflow execution (already working)  
3. 🔧 **Chart data appears in final API response** (currently broken)
4. 🔧 **Charts display correctly in frontend** (depends on #3)

---

*Last Updated: 2025-09-17*
*Status: In Progress - Data flow issue identified, break point needs to be located*
