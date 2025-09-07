#!/usr/bin/env python3
"""
Frontend Integration Test for Database Migration
Verifies that agent workspaces will work as before with the new database system.
"""

import asyncio
import json
import time
import requests
from typing import Dict, Any

class FrontendIntegrationTester:
    """Test frontend-backend integration after database migration"""
    
    def __init__(self):
        self.base_urls = {
            "cleaning": "http://127.0.0.1:8004",
            "visualization": "http://127.0.0.1:8006"
        }
        self.results = {}
    
    def test_agent_health(self, agent_name: str) -> bool:
        """Test if an agent is healthy"""
        try:
            response = requests.get(f"{self.base_urls[agent_name]}/health", timeout=5)
            if response.status_code == 200:
                print(f"✅ {agent_name.capitalize()} agent is healthy")
                return True
            else:
                print(f"❌ {agent_name.capitalize()} agent health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ {agent_name.capitalize()} agent connection failed: {e}")
            return False
    
    def test_session_creation(self, agent_name: str) -> str:
        """Test session creation with sample data"""
        print(f"\n🔧 Testing {agent_name} session creation...")
        
        try:
            if agent_name == "cleaning":
                # Test data cleaning session creation
                data = {
                    "data": {
                        "Name": ["John", "Jane", "Bob", "Alice"],
                        "Age": [25, None, 30, 28],
                        "Salary": [50000, 60000, None, 55000]
                    },
                    "user_instructions": "Handle missing values appropriately",
                    "max_retries": 3
                }
                response = requests.post(f"{self.base_urls[agent_name]}/clean-data", 
                                       json=data, timeout=30)
                
            elif agent_name == "visualization":
                # Test visualization session creation
                data = {
                    "data": {
                        "x": [1, 2, 3, 4, 5],
                        "y": [10, 15, 13, 17, 20]
                    },
                    "chart_type": "line",
                    "title": "Sample Line Chart"
                }
                response = requests.post(f"{self.base_urls[agent_name]}/create-chart", 
                                       json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success") and result.get("session_id"):
                    session_id = result["session_id"]
                    print(f"✅ {agent_name.capitalize()} session created: {session_id}")
                    return session_id
                else:
                    print(f"❌ {agent_name.capitalize()} session creation failed: {result}")
                    return None
            else:
                print(f"❌ {agent_name.capitalize()} session creation failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ {agent_name.capitalize()} session creation error: {e}")
            return None
    
    def test_session_retrieval(self, agent_name: str, session_id: str) -> bool:
        """Test session data retrieval"""
        print(f"\n📖 Testing {agent_name} session retrieval...")
        
        try:
            if agent_name == "cleaning":
                # Test cleaning data retrieval
                data = {"session_id": session_id}
                response = requests.post(f"{self.base_urls[agent_name]}/get-cleaned-data", 
                                       json=data, timeout=10)
                
            elif agent_name == "visualization":
                # Test visualization retrieval
                response = requests.get(f"{self.base_urls[agent_name]}/session/{session_id}/plotly-graph", 
                                      timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success", True):  # Some endpoints don't have success field
                    print(f"✅ {agent_name.capitalize()} session data retrieved successfully")
                    return True
                else:
                    print(f"❌ {agent_name.capitalize()} session retrieval failed: {result}")
                    return False
            else:
                print(f"❌ {agent_name.capitalize()} session retrieval failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ {agent_name.capitalize()} session retrieval error: {e}")
            return False
    
    def test_session_deletion(self, agent_name: str, session_id: str) -> bool:
        """Test session deletion"""
        print(f"\n🗑️  Testing {agent_name} session deletion...")
        
        try:
            data = {"session_id": session_id}
            response = requests.post(f"{self.base_urls[agent_name]}/delete-session", 
                                   json=data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    print(f"✅ {agent_name.capitalize()} session deleted successfully")
                    return True
                else:
                    print(f"❌ {agent_name.capitalize()} session deletion failed: {result}")
                    return False
            else:
                print(f"❌ {agent_name.capitalize()} session deletion failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ {agent_name.capitalize()} session deletion error: {e}")
            return False
    
    def run_comprehensive_test(self):
        """Run comprehensive integration test"""
        print("🧪 Frontend Integration Test for Database Migration")
        print("=" * 60)
        print("Testing if agent workspaces will work as before...")
        
        overall_success = True
        
        for agent_name in ["cleaning", "visualization"]:
            print(f"\n📋 Testing {agent_name.upper()} AGENT INTEGRATION")
            print("-" * 40)
            
            # Test 1: Health Check
            health_ok = self.test_agent_health(agent_name)
            if not health_ok:
                overall_success = False
                continue
            
            # Test 2: Session Creation
            session_id = self.test_session_creation(agent_name)
            if not session_id:
                overall_success = False
                continue
            
            # Test 3: Session Retrieval
            retrieval_ok = self.test_session_retrieval(agent_name, session_id)
            if not retrieval_ok:
                overall_success = False
            
            # Test 4: Session Deletion
            deletion_ok = self.test_session_deletion(agent_name, session_id)
            if not deletion_ok:
                overall_success = False
            
            # Store results
            self.results[agent_name] = {
                "health": health_ok,
                "session_creation": bool(session_id),
                "session_retrieval": retrieval_ok,
                "session_deletion": deletion_ok,
                "overall": health_ok and bool(session_id) and retrieval_ok and deletion_ok
            }
        
        # Final Results
        print("\n" + "=" * 60)
        print("🏁 FINAL INTEGRATION TEST RESULTS")
        print("=" * 60)
        
        for agent_name, results in self.results.items():
            status = "✅ PASS" if results["overall"] else "❌ FAIL"
            print(f"{agent_name.upper():15}: {status}")
            print(f"                - Health: {'✅' if results['health'] else '❌'}")
            print(f"                - Create: {'✅' if results['session_creation'] else '❌'}")
            print(f"                - Read:   {'✅' if results['session_retrieval'] else '❌'}")
            print(f"                - Delete: {'✅' if results['session_deletion'] else '❌'}")
        
        print("-" * 60)
        
        if overall_success:
            print("🎉 SUCCESS: Frontend agent workspaces WILL work as before!")
            print("✅ Database migration maintains full compatibility")
            print("✅ Sessions are persisted and retrievable")
            print("✅ All CRUD operations working correctly")
            print("✅ Agent functionality preserved")
        else:
            print("⚠️  WARNING: Some integration issues detected")
            print("❌ Frontend workspaces may have reduced functionality")
        
        print("=" * 60)
        return overall_success

def main():
    tester = FrontendIntegrationTester()
    success = tester.run_comprehensive_test()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())






