#!/usr/bin/env python3
"""
Test script for Data Cleaning REST API endpoints

This script tests all the endpoints to ensure they're working correctly.
Make sure the agent is running on port 8003 before running this script.
"""

import requests
import json
import base64
import time
import os

# Configuration (local REST agent)
AGENT_URL = "http://127.0.0.1:8003"
TEST_CSV_FILE = "test_sample_data.csv"

def test_health_endpoint():
    """Test the health check endpoint"""
    print("🔍 Testing Health Endpoint...")
    
    try:
        response = requests.get(f"{AGENT_URL}/health", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed!")
            print(f"   Status: {data.get('status')}")
            print(f"   Agent: {data.get('agent')}")
            print(f"   Message: {data.get('message')}")
            return True
        else:
            print(f"❌ Health check failed with status code: {response.status_code}")
            return False
            
    except requests.RequestException as e:
        print(f"❌ Health check failed: {str(e)}")
        return False

def test_csv_cleaning_endpoint():
    """Test the CSV cleaning endpoint"""
    print("\n📄 Testing CSV Cleaning Endpoint...")
    
    # Check if test file exists
    if not os.path.exists(TEST_CSV_FILE):
        print(f"❌ Test file {TEST_CSV_FILE} not found!")
        return False
    
    try:
        # Read and encode the CSV file
        with open(TEST_CSV_FILE, 'rb') as f:
            file_content = f.read()
            file_content_b64 = base64.b64encode(file_content).decode('utf-8')
        
        # Prepare the request
        payload = {
            "filename": TEST_CSV_FILE,
            "file_content": file_content_b64,
            "user_instructions": "Remove duplicates and handle missing values carefully. Don't remove outliers.",
            "max_retries": 3
        }
        
        print("   📤 Sending CSV file for cleaning...")
        response = requests.post(
            f"{AGENT_URL}/clean-csv", 
            json=payload, 
            timeout=300  # 5 minutes timeout for processing
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('success'):
                print("✅ CSV cleaning successful!")
                print(f"   Original shape: {data.get('original_shape')}")
                print(f"   Cleaned shape: {data.get('cleaned_shape')}")
                
                # Calculate data retention
                if data.get('original_shape') and data.get('cleaned_shape'):
                    orig_rows = data['original_shape'][0]
                    clean_rows = data['cleaned_shape'][0]
                    retention = (clean_rows / orig_rows) * 100 if orig_rows > 0 else 0
                    print(f"   Data retention: {retention:.1f}%")
                
                # Show cleaning steps
                if data.get('cleaning_steps'):
                    print(f"   Cleaning steps applied: {len(data['cleaning_steps'])} characters")
                
                # Show sample of cleaned data
                if data.get('cleaned_data') and data['cleaned_data'].get('records'):
                    records = data['cleaned_data']['records']
                    print(f"   Sample cleaned records: {min(3, len(records))} of {len(records)}")
                    for i, record in enumerate(records[:3]):
                        print(f"     Record {i+1}: {dict(list(record.items())[:3])}...")
                
                return True
            else:
                print(f"❌ CSV cleaning failed: {data.get('error')}")
                return False
        else:
            print(f"❌ CSV cleaning failed with status code: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data}")
            except:
                print(f"   Response: {response.text}")
            return False
            
    except requests.RequestException as e:
        print(f"❌ CSV cleaning request failed: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ CSV cleaning test failed: {str(e)}")
        return False

def test_direct_data_endpoint():
    """Test the direct data cleaning endpoint"""
    print("\n📊 Testing Direct Data Cleaning Endpoint...")
    
    # Sample data with various issues
    test_data = {
        "name": ["John Doe", "Jane Smith", "", "Bob Johnson", "Alice Brown", "John Doe"],  # Duplicate and empty
        "age": [25, None, 30, 35, 28, 25],  # Missing value
        "salary": [50000, 60000, None, 75000, 55000, 50000],  # Missing value
        "department": ["Engineering", "Marketing", "Sales", "", "Engineering", "Engineering"],  # Empty string
        "email": ["john@email.com", "jane@email.com", "invalid-email", "bob@email.com", "alice@email.com", "john@email.com"]  # Invalid email
    }
    
    payload = {
        "data": test_data,
        "user_instructions": "Fill missing values appropriately and remove duplicates. Fix invalid email formats if possible.",
        "max_retries": 3
    }
    
    try:
        print("   📤 Sending direct data for cleaning...")
        response = requests.post(
            f"{AGENT_URL}/clean-data", 
            json=payload, 
            timeout=300
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('success'):
                print("✅ Direct data cleaning successful!")
                print(f"   Original shape: {data.get('original_shape')}")
                print(f"   Cleaned shape: {data.get('cleaned_shape')}")
                
                # Show sample of cleaned data
                if data.get('cleaned_data') and data['cleaned_data'].get('records'):
                    records = data['cleaned_data']['records']
                    print(f"   Cleaned records: {len(records)}")
                    
                    # Show first few records
                    for i, record in enumerate(records[:3]):
                        print(f"     Record {i+1}: {record}")
                
                return True
            else:
                print(f"❌ Direct data cleaning failed: {data.get('error')}")
                return False
        else:
            print(f"❌ Direct data cleaning failed with status code: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data}")
            except:
                print(f"   Response: {response.text}")
            return False
            
    except requests.RequestException as e:
        print(f"❌ Direct data cleaning request failed: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ Direct data cleaning test failed: {str(e)}")
        return False

def test_error_handling():
    """Test error handling with invalid requests"""
    print("\n🚨 Testing Error Handling...")
    
    # Test 1: Invalid CSV content
    print("   Testing invalid CSV content...")
    try:
        invalid_payload = {
            "filename": "invalid.csv",
            "file_content": "aW52YWxpZCBjc3YgY29udGVudA==",  # "invalid csv content" in base64
            "user_instructions": "Clean this data",
            "max_retries": 3
        }
        
        response = requests.post(f"{AGENT_URL}/clean-csv", json=invalid_payload, timeout=30)
        data = response.json()
        
        if not data.get('success'):
            print("   ✅ Invalid CSV properly rejected")
        else:
            print("   ⚠️ Invalid CSV was not rejected (might be handled gracefully)")
            
    except Exception as e:
        print(f"   ❌ Error testing invalid CSV: {str(e)}")
    
    # Test 2: Empty data
    print("   Testing empty data...")
    try:
        empty_payload = {
            "data": {},
            "user_instructions": "Clean this empty data",
            "max_retries": 3
        }
        
        response = requests.post(f"{AGENT_URL}/clean-data", json=empty_payload, timeout=30)
        data = response.json()
        
        if not data.get('success'):
            print("   ✅ Empty data properly rejected")
        else:
            print("   ⚠️ Empty data was not rejected")
            
    except Exception as e:
        print(f"   ❌ Error testing empty data: {str(e)}")

def main():
    """Run all tests"""
    print("🧪 Starting Data Cleaning API Tests")
    print("=" * 50)
    
    # Track test results
    results = []
    
    # Test 1: Health check
    results.append(("Health Check", test_health_endpoint()))
    
    # Test 2: CSV cleaning
    results.append(("CSV Cleaning", test_csv_cleaning_endpoint()))
    
    # Test 3: Direct data cleaning
    results.append(("Direct Data Cleaning", test_direct_data_endpoint()))
    
    # Test 4: Error handling
    test_error_handling()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Results Summary:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your API is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the agent logs and ensure it's running properly.")
        
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 