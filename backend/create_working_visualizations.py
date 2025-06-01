# backend/test_visualization_endpoints.py
#!/usr/bin/env python3
"""
Test script that USES the localhost API endpoints
"""
import requests
import json
import time

API_BASE = "http://127.0.0.1:5000/api"

def test_visualization_via_endpoints():
    """Test visualization generation via API endpoints"""
    print("🌐 Testing Visualization System via Localhost Endpoints")
    print("=" * 70)
    
    # Test 1: Complexity Visualization via API
    print("\n📊 Test 1: Complexity Visualization via API")
    complexity_data = {
        "scenario_type": "complexity_analysis",
        "data": {
            "time_complexity": "O(n²)",
            "space_complexity": "O(1)",
            "algorithm_name": "Bubble Sort",
            "code": "def bubble_sort(arr): return sorted(arr)"
        },
        "requirements": ["big_o_comparison", "growth_curves"],
        "output_format": "static",
        "target_platform": "web"
    }
    
    print("📤 Sending request to /api/visualization/complexity-viz...")
    response = requests.post(f"{API_BASE}/visualization/complexity-viz", json=complexity_data)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ API request successful!")
        print(f"📁 Files generated: {result.get('files_generated', [])}")
    else:
        print(f"❌ API request failed: {response.status_code}")
        print(response.text)
    
    # Test 2: Algorithm Animation via API
    print("\n🎬 Test 2: Algorithm Animation via API")
    animation_data = {
        "scenario_type": "algorithm_execution",
        "data": {
            "algorithm_type": "sorting",
            "code": "def bubble_sort(arr): return sorted(arr)",
            "sample_input": [64, 34, 25, 12, 22, 11, 90]
        },
        "requirements": ["step_by_step", "animation"],
        "output_format": "animated",
        "target_platform": "web"
    }
    
    print("📤 Sending request to /api/visualization/algorithm-animation...")
    response = requests.post(f"{API_BASE}/visualization/algorithm-animation", json=animation_data)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ API request successful!")
        print(f"🎬 Animations created: {result.get('animations_created', 0)}")
    else:
        print(f"❌ API request failed: {response.status_code}")
        print(response.text)
    
    # Test 3: Full Visualization Generation via API
    print("\n🎨 Test 3: Full Visualization Generation via API")
    full_data = {
        "scenario_type": "comprehensive_analysis",
        "data": {
            "algorithms": ["bubble_sort", "quick_sort", "merge_sort"],
            "performance_data": {
                "bubble_sort": {"time": "O(n²)", "space": "O(1)"},
                "quick_sort": {"time": "O(n log n)", "space": "O(log n)"},
                "merge_sort": {"time": "O(n log n)", "space": "O(n)"}
            }
        },
        "requirements": ["comparison", "performance", "educational"],
        "output_format": "interactive",
        "target_platform": "web"
    }
    
    print("📤 Sending request to /api/visualization/generate...")
    response = requests.post(f"{API_BASE}/visualization/generate", json=full_data)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ API request successful!")
        print(f"🎨 Visualizations generated: {result.get('visualizations_generated', 0)}")
        print(f"🤖 Agents used: {[r['agent'] for r in result.get('results', [])]}")
    else:
        print(f"❌ API request failed: {response.status_code}")
        print(response.text)
    
    print("\n" + "=" * 70)
    print("🌐 API Endpoint Testing Complete!")
    print("📌 This test USES the localhost API endpoints")

if __name__ == "__main__":
    test_visualization_via_endpoints()
