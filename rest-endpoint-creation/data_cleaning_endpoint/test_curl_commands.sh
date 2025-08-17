#!/bin/bash

# Data Cleaning API - cURL Test Commands
# Make sure the agent is running on port 8003 before running these commands

echo "🧪 Data Cleaning API - cURL Tests"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
AGENT_URL="http://127.0.0.1:8003"
# Resolve CSV file relative to this script's directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CSV_FILE="$SCRIPT_DIR/test_sample_data.csv"

echo -e "${BLUE}Agent URL: $AGENT_URL${NC}"
echo ""

# Test 1: Health Check
echo -e "${YELLOW}1. Testing Health Check Endpoint${NC}"
echo "Command: curl -X GET $AGENT_URL/health"
echo ""

response=$(curl -s -X GET $AGENT_URL/health)
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Health check successful:${NC}"
    echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"
else
    echo -e "${RED}❌ Health check failed${NC}"
fi

echo ""
echo "----------------------------------------"
echo ""

# Test 2: CSV File Upload
echo -e "${YELLOW}2. Testing CSV File Upload Endpoint${NC}"

if [ ! -f "$CSV_FILE" ]; then
    echo -e "${RED}❌ Test file $CSV_FILE not found!${NC}"
    echo "Please make sure the test CSV file exists."
else
    echo "Encoding CSV file to base64..."
base64_content=$(base64 -b 0 "$CSV_FILE" 2>/dev/null || base64 "$CSV_FILE" | tr -d '\n')
    
    echo "Command: curl -X POST $AGENT_URL/clean-csv"
    echo "Payload: CSV file with cleaning instructions"
    echo ""
    
    # Create JSON payload
    json_payload=$(cat <<EOF
{
    "filename": "$CSV_FILE",
    "file_content": "$base64_content",
    "user_instructions": "Remove duplicates and handle missing values carefully. Don't remove outliers unless absolutely necessary.",
    "max_retries": 3
}
EOF
)
    
    echo "Sending request... (this may take a few moments)"
response=$(curl -s -X POST "$AGENT_URL/clean-csv" \
        -H "Content-Type: application/json" \
        -d "$json_payload")
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ CSV cleaning request sent successfully:${NC}"
        echo "$response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if data.get('success'):
        print('Success: ' + data.get('message', 'No message'))
        if data.get('original_shape'):
            print('Original shape: ' + str(data['original_shape']))
        if data.get('cleaned_shape'):
            print('Cleaned shape: ' + str(data['cleaned_shape']))
        if data.get('cleaned_data') and data['cleaned_data'].get('records'):
            print('Records cleaned: ' + str(len(data['cleaned_data']['records'])))
    else:
        print('Error: ' + data.get('error', 'Unknown error'))
except:
    print(sys.stdin.read())
" 2>/dev/null || echo "$response"
    else
        echo -e "${RED}❌ CSV cleaning request failed${NC}"
    fi
fi

echo ""
echo "----------------------------------------"
echo ""

# Test 3: Direct Data Cleaning
echo -e "${YELLOW}3. Testing Direct Data Cleaning Endpoint${NC}"
echo "Command: curl -X POST $AGENT_URL/clean-data"
echo "Payload: JSON data with missing values and duplicates"
echo ""

# Create test data with issues
direct_data_payload=$(cat <<'EOF'
{
    "data": {
        "name": ["John Doe", "Jane Smith", "", "Bob Johnson", "Alice Brown", "John Doe"],
        "age": [25, null, 30, 35, 28, 25],
        "salary": [50000, 60000, null, 75000, 55000, 50000],
        "department": ["Engineering", "Marketing", "Sales", "", "Engineering", "Engineering"],
        "email": ["john@email.com", "jane@email.com", "invalid-email", "bob@email.com", "alice@email.com", "john@email.com"]
    },
    "user_instructions": "Fill missing values appropriately and remove duplicates. Handle invalid email formats.",
    "max_retries": 3
}
EOF
)

echo "Sending request..."
response=$(curl -s -X POST "$AGENT_URL/clean-data" \
    -H "Content-Type: application/json" \
    -d "$direct_data_payload")

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Direct data cleaning request sent successfully:${NC}"
    echo "$response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if data.get('success'):
        print('Success: ' + data.get('message', 'No message'))
        if data.get('original_shape'):
            print('Original shape: ' + str(data['original_shape']))
        if data.get('cleaned_shape'):
            print('Cleaned shape: ' + str(data['cleaned_shape']))
        if data.get('cleaned_data') and data['cleaned_data'].get('records'):
            records = data['cleaned_data']['records']
            print('Records cleaned: ' + str(len(records)))
            print('Sample record: ' + str(records[0]) if records else 'No records')
    else:
        print('Error: ' + data.get('error', 'Unknown error'))
except:
    print(sys.stdin.read())
" 2>/dev/null || echo "$response"
else
    echo -e "${RED}❌ Direct data cleaning request failed${NC}"
fi

echo ""
echo "----------------------------------------"
echo ""

# Test 4: Error Handling
echo -e "${YELLOW}4. Testing Error Handling${NC}"
echo "Testing with invalid CSV content..."

invalid_payload=$(cat <<'EOF'
{
    "filename": "invalid.csv",
    "file_content": "aW52YWxpZCBjc3YgY29udGVudA==",
    "user_instructions": "Clean this invalid data",
    "max_retries": 3
}
EOF
)

response=$(curl -s -X POST "$AGENT_URL/clean-csv" \
    -H "Content-Type: application/json" \
    -d "$invalid_payload")

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Error handling test completed:${NC}"
    echo "$response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if not data.get('success'):
        print('✅ Invalid data properly rejected: ' + data.get('error', 'No error message'))
    else:
        print('⚠️ Invalid data was not rejected (might be handled gracefully)')
except:
    print(sys.stdin.read())
" 2>/dev/null || echo "$response"
else
    echo -e "${RED}❌ Error handling test failed${NC}"
fi

echo ""
echo "========================================"
echo -e "${BLUE}🎉 All cURL tests completed!${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Check the agent logs for detailed processing information"
echo "2. Try the web interface at http://127.0.0.1:5000"
echo "3. Run the Python test script: python3 test_endpoints.py"
echo ""
echo -e "${BLUE}Troubleshooting:${NC}"
echo "- Make sure the agent is running: python3 data_cleaning_rest_agent.py"
echo "- Check your OpenAI API key is set: echo \$OPENAI_API_KEY"
echo "- Verify ports 8003 and 5000 are available" 