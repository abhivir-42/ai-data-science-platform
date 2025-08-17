# Data Cleaning REST API

This project provides a REST API wrapper around the LangChain Data Cleaning Agent, allowing you to clean datasets via HTTP requests and a web interface.

## 🏗️ Architecture

```
User/Browser/cURL → Flask Frontend (Port 5000) → Local REST Agent (Port 8003) → DataCleaningAgent → LLM
```

## 📋 Prerequisites

1. **Python 3.8+**
2. **OpenAI API Key** - either in a `.env` file at the repo root or exported as env var `OPENAI_API_KEY`
3. **Required Python packages** (install via pip):
   ```bash
   pip install uagents flask pandas langchain-openai requests werkzeug
   ```

## 🚀 Quick Start

### 1. Set up environment
```bash
# Option A: Set your OpenAI API key for this shell
export OPENAI_API_KEY="your-api-key-here"

# Option B: Put it in a .env file at the repo root (auto-loaded)
# echo "OPENAI_API_KEY=your-api-key-here" >> .env

# Navigate to the project directory
cd rest-endpoint-creation/data_cleaning_endpoint
```

### 2. Start the Data Cleaning Agent (Terminal 1)
```bash
python data_cleaning_rest_agent.py
```
You should see:
```
🧹 Starting Data Cleaning REST Agent...
📡 Available endpoints:
   GET  http://127.0.0.1:8003/health
   POST http://127.0.0.1:8003/clean-csv
   POST http://127.0.0.1:8003/clean-data
🚀 Agent starting...
```

### 3. Start the Frontend (Terminal 2)
```bash
python frontend_app.py
```
You should see:
```
🌐 Starting Data Cleaning Frontend...
📱 Web interface: http://127.0.0.1:5000
🔗 Agent endpoint: http://127.0.0.1:8003
💡 Make sure the data cleaning agent is running on port 8003
```

### 4. Access the Web Interface
Open your browser to: **http://127.0.0.1:5000**

## 🌐 Web Interface Features

### Upload & Clean Data
- **File Upload**: Drag & drop or select CSV files
- **Custom Instructions**: Add specific cleaning requirements
- **Real-time Processing**: See progress and results instantly
- **Data Preview**: View cleaned data in a table
- **Download Results**: Get cleaned data as CSV

### Visual Features
- **Health Indicator**: Shows agent connection status
- **Statistics Dashboard**: Compare original vs cleaned data
- **Cleaning Steps**: See what operations were performed
- **Responsive Design**: Works on desktop and mobile

## 🔧 API Endpoints

### 1. Health Check
**GET** `http://127.0.0.1:8003/health`

**Response:**
```json
{
  "status": "healthy",
  "agent": "data_cleaning_rest_agent",
  "message": "Data cleaning agent is running and ready to process data"
}
```

### 2. Clean CSV File
**POST** `http://127.0.0.1:8003/clean-csv`

**Request Body:**
```json
{
  "filename": "data.csv",
  "file_content": "base64_encoded_csv_content",
  "user_instructions": "Don't remove outliers, focus on missing values",
  "max_retries": 3
}
```

**Response:**
```json
{
  "success": true,
  "message": "Data cleaning completed successfully",
  "original_shape": [23, 8],
  "cleaned_shape": [22, 8],
  "cleaning_steps": "# Recommended Data Cleaning Steps:\n1. Handle missing values...",
  "cleaned_data": {
    "records": [...],
    "columns": ["name", "age", "salary", ...]
  },
  "error": null
}
```

### 3. Clean Direct Data
**POST** `http://127.0.0.1:8003/clean-data`

**Request Body:**
```json
{
  "data": {
    "name": ["John", "Jane", "Bob"],
    "age": [25, null, 35],
    "salary": [50000, 60000, 75000]
  },
  "user_instructions": "Fill missing ages with median",
  "max_retries": 3
}
```

## 🧪 Testing with cURL

### 1. Test Health Check
```bash
curl -X GET http://127.0.0.1:8003/health
```

### 2. Test CSV Upload
First, encode your CSV file to base64:
```bash
# Encode the sample CSV file
base64 -i test_sample_data.csv -o encoded_data.txt
```

Then send the request:
```bash
curl -X POST http://127.0.0.1:8003/clean-csv \
  -H "Content-Type: application/json" \
  -d '{
    "filename": "test_data.csv",
    "file_content": "'$(cat encoded_data.txt)'",
    "user_instructions": "Remove duplicates and handle missing values carefully",
    "max_retries": 3
  }'
```

### 3. Test Direct Data Cleaning
```bash
curl -X POST http://127.0.0.1:8003/clean-data \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "name": ["John Doe", "Jane Smith", "", "Bob Johnson"],
      "age": [25, null, 30, 35],
      "salary": [50000, 60000, null, 75000],
      "email": ["john@email.com", "", "invalid-email", "bob@email.com"]
    },
    "user_instructions": "Fill missing values and validate email formats",
    "max_retries": 3
  }'
```

## 📊 Sample Data

The project includes `test_sample_data.csv` with intentionally messy data:
- Missing values in multiple columns
- Duplicate rows
- Invalid/outlier values
- Empty strings and nulls
- Inconsistent data types

This file is perfect for testing the cleaning capabilities.

## 🎯 Data Cleaning Features

The agent automatically performs these cleaning operations:

### Default Cleaning Steps:
1. **Missing Value Handling**
   - Numeric columns: Fill with mean/median
   - Categorical columns: Fill with mode or 'Unknown'
   - Date columns: Forward/backward fill

2. **Duplicate Removal**
   - Remove exact duplicate rows
   - Keep first occurrence

3. **Data Type Optimization**
   - Convert string numbers to numeric
   - Handle date columns properly
   - Optimize memory usage

4. **Outlier Handling** (Conservative)
   - Only if explicitly requested
   - Uses IQR method
   - Considers domain context

5. **Data Validation**
   - Ensures data retention >75%
   - Validates column integrity
   - Provides detailed logging

### Custom Instructions:
You can modify the cleaning process with instructions like:
- "Don't remove outliers"
- "Focus only on missing values"
- "Remove rows with more than 50% missing data"
- "Convert all text to lowercase"

## 🐛 Troubleshooting

### Common Issues:

1. **Agent won't start**
   - Check if OpenAI API key is set: `echo $OPENAI_API_KEY`
   - Ensure all dependencies are installed
   - Check if port 8003 is available

2. **Frontend can't connect to agent**
   - Verify agent is running on port 8003
   - Check firewall settings
   - Look at agent logs for errors

3. **File upload fails**
   - Check file size (max 16MB)
   - Ensure file is valid CSV format
   - Check browser console for errors

4. **Cleaning process fails**
   - Check agent logs for detailed errors
   - Try with smaller dataset first
   - Verify data format is supported

### Logs and Debugging:
- Agent logs appear in the terminal where you started the agent
- Frontend logs appear in browser console (F12)
- Cleaning function logs are saved to `logs/data_cleaning/` directory

## 🔒 Security Notes

- The API runs locally and is not production-ready
- File uploads are processed in memory (16MB limit)
- No authentication or rate limiting implemented
- For production use, add proper security measures

## 🚀 Production Deployment

For production deployment, consider:
1. Add authentication (API keys, OAuth)
2. Implement rate limiting
3. Add input validation and sanitization
4. Use a production WSGI server (Gunicorn, uWSGI)
5. Add monitoring and logging
6. Implement proper error handling
7. Use environment variables for configuration
8. Add database for storing results/history

## 📝 API Response Codes

- **200**: Success
- **400**: Bad Request (invalid input)
- **500**: Internal Server Error
- **413**: Payload Too Large (file too big)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is for educational and development purposes.

---

**Happy Data Cleaning! 🧹✨** 