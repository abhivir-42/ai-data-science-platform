"""
Flask Frontend for Data Cleaning Agent

This Flask application provides a simple web interface to interact with the
Data Cleaning REST Agent. Users can upload CSV files and get cleaned data back.
"""

from flask import Flask, render_template, request, jsonify, send_file
import requests
import base64
import json
import pandas as pd
import io
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

"""Frontend that calls the local REST agent on port 8003."""
# Agent endpoint (local REST server in this folder)
AGENT_URL = "http://127.0.0.1:8003"

# Ensure upload directory exists
UPLOAD_FOLDER = 'temp_uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    """Main page with file upload form"""
    return render_template('index.html')

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    """Handle CSV file upload and cleaning"""
    try:
        # Check if file was uploaded
        if 'csvFile' not in request.files:
            return jsonify({"error": "No file uploaded"})
        
        file = request.files['csvFile']
        if file.filename == '':
            return jsonify({"error": "No file selected"})
        
        # Get user instructions
        user_instructions = request.form.get('instructions', '').strip()
        if not user_instructions:
            user_instructions = None
        
        # Read file content
        try:
            file_content = file.read()
            file_content_b64 = base64.b64encode(file_content).decode('utf-8')
        except Exception as e:
            return jsonify({"error": f"Failed to read file: {str(e)}"})
        
        # Prepare request for the agent
        payload = {
            "filename": secure_filename(file.filename),
            "file_content": file_content_b64,
            "user_instructions": user_instructions,
            "max_retries": 3
        }
        
        # Send request to the data cleaning agent
        try:
            response = requests.post(f"{AGENT_URL}/clean-csv", json=payload, timeout=300)
            response.raise_for_status()
            result = response.json()
            
            if result.get('success'):
                return jsonify({
                    "success": True,
                    "message": result.get('message'),
                    "original_shape": result.get('original_shape'),
                    "cleaned_shape": result.get('cleaned_shape'),
                    "cleaning_steps": result.get('cleaning_steps'),
                    "cleaned_data": result.get('cleaned_data')
                })
            else:
                return jsonify({
                    "success": False,
                    "error": result.get('error', 'Unknown error occurred')
                })
                
        except requests.RequestException as e:
            return jsonify({"error": f"Failed to connect to cleaning agent: {str(e)}"})
        except Exception as e:
            return jsonify({"error": f"Request failed: {str(e)}"})
            
    except Exception as e:
        return jsonify({"error": f"Upload failed: {str(e)}"})

@app.route('/download_csv', methods=['POST'])
def download_csv():
    """Download cleaned data as CSV file"""
    try:
        data = request.json.get('data')
        if not data or 'records' not in data:
            return jsonify({"error": "No data to download"})
        
        # Convert records back to DataFrame
        df = pd.DataFrame(data['records'])
        
        # Create CSV content
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        
        # Create a file-like object
        csv_file = io.BytesIO(csv_content.encode('utf-8'))
        
        return send_file(
            csv_file,
            mimetype='text/csv',
            as_attachment=True,
            download_name='cleaned_data.csv'
        )
        
    except Exception as e:
        return jsonify({"error": f"Download failed: {str(e)}"})

@app.route('/health')
def health_check():
    """Check health of the data cleaning agent"""
    try:
        response = requests.get(f"{AGENT_URL}/health", timeout=5)
        if response.status_code == 200:
            agent_health = response.json()
            return jsonify({
                "frontend_status": "healthy",
                "agent_status": agent_health.get("status", "unknown"),
                "agent_message": agent_health.get("message", "No message"),
                "connection": "connected"
            })
        else:
            return jsonify({
                "frontend_status": "healthy",
                "agent_status": "unhealthy",
                "connection": "disconnected",
                "error": f"Agent returned status code: {response.status_code}"
            })
    except requests.RequestException as e:
        return jsonify({
            "frontend_status": "healthy",
            "agent_status": "offline",
            "connection": "failed",
            "error": str(e)
        })

if __name__ == '__main__':
    print("🌐 Starting Data Cleaning Frontend...")
    print("📱 Web interface: http://127.0.0.1:5000")
    print("🔗 Agent endpoint: http://127.0.0.1:8003")
    print("💡 Make sure the data cleaning agent is running on port 8003")
    app.run(host='127.0.0.1', port=5000, debug=True) 