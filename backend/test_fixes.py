# backend/test_final_fix.py
#!/usr/bin/env python3
import requests
import json

API_BASE = "http://127.0.0.1:5000/api"

def test_final_complexity_fix():
    print("🧪 Testing Final Complexity Fix")
    print("=" * 50)
    
    test_code = """
for i in range(n):
    print(i)
    """
    
    payload = {
        "code": test_code,
        "language": "python"
    }
    
    print(f"📤 Sending request...")
    response = requests.post(f"{API_BASE}/complexity/analyze", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ SUCCESS! Complexity analysis working!")
        complexity = result["complexity_analysis"]
        print(f"📊 Time Complexity: {complexity.get('time_complexity')}")
        print(f"📊 Space Complexity: {complexity.get('space_complexity')}")
        print(f"🔧 RAG Enhanced: {result['ai_processing_details'].get('rag_enhanced', False)}")
    else:
        print(f"❌ Still failing: {response.status_code}")
        print(response.text)

def test_algorithm_solver_fix():
    print("\n🧮 Testing Algorithm Solver Fix")  
    print("=" * 50)
    
    test_problem = "Find the maximum element in an array"
    
    payload = {"problem": test_problem}
    
    print(f"📤 Sending request...")
    response = requests.post(f"{API_BASE}/algorithm-solver/solve", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ SUCCESS! Algorithm solver working!")
        print(f"📊 Problem analyzed successfully")
    else:
        print(f"❌ Still failing: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_final_complexity_fix()
    test_algorithm_solver_fix()
