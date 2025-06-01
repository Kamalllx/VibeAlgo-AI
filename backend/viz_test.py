# backend/test_visualization_debug.py
#!/usr/bin/env python3
import requests
import json
import os
import time

API_BASE = "http://127.0.0.1:5000/api"

def test_visualization_with_full_debug():
    print("🐛 Testing Visualization System with Full Debug Output")
    print("=" * 80)
    
    # Test 1: Simple Complexity Visualization
    print("\n📊 Test 1: Complexity Visualization with Debug")
    complexity_data = {
        "scenario_type": "complexity_analysis", 
        "data": {
            "time_complexity": "O(n²)",
            "space_complexity": "O(1)",
            "algorithm_name": "Bubble Sort"
        },
        "requirements": ["big_o_comparison"],
        "output_format": "static",
        "target_platform": "debug"
    }
    
    print("📤 Sending request...")
    response = requests.post(f"{API_BASE}/visualization/complexity-viz", json=complexity_data)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Request successful!")
        print(f"📁 Files generated: {result.get('files_generated', [])}")
        
        # Wait for file generation
        time.sleep(3)
        
        # Check if debug files were created
        print("\n🔍 Checking for debug files...")
        debug_path = "generated_visualizations"
        if os.path.exists(debug_path):
            for root, dirs, files in os.walk(debug_path):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        print(f"📝 Found debug file: {file_path}")
                        
                        # Show first few lines of the file
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                lines = f.readlines()[:10]
                                print(f"   First 10 lines:")
                                for i, line in enumerate(lines, 1):
                                    print(f"   {i:2d}: {line.rstrip()}")
                        except Exception as e:
                            print(f"   ❌ Could not read file: {e}")
                        print()
        
    else:
        print(f"❌ Request failed: {response.status_code}")
        try:
            error_data = response.json()
            print(f"📝 Error details: {json.dumps(error_data, indent=2)}")
        except:
            print(f"📝 Raw error: {response.text}")
    
    print("\n" + "=" * 80)
    print("🐛 Debug Testing Complete!")
    print("📁 Check the 'generated_visualizations' folder for debug files")

if __name__ == "__main__":
    test_visualization_with_full_debug()
