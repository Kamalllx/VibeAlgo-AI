
# backend/cli/test_complexity.py
#!/usr/bin/env python3
"""
CLI tool to test complexity analysis functionality
"""

import sys
import os
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.complexity_analyzer import analyze_complexity

def test_complexity_analysis():
    """Test complexity analysis with sample code"""
    
    print("🔍 Testing Algorithm Complexity Analyzer")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        {
            'name': 'Simple Linear Loop',
            'code': '''
for i in range(n):
    print(i)
            '''.strip()
        },
        {
            'name': 'Nested Loops',
            'code': '''
for i in range(n):
    for j in range(n):
        print(i, j)
            '''.strip()
        },
        {
            'name': 'Binary Search',
            'code': '''
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
            '''.strip()
        },
        {
            'name': 'Bubble Sort',
            'code': '''
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
            '''.strip()
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}: {test_case['name']}")
        print("-" * 40)
        print("Code:")
        print(test_case['code'])
        print("\nAnalysis Result:")
        
        try:
            result = analyze_complexity(test_case['code'])
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
            else:
                print(f"⏱️  Time Complexity: {result.get('time_complexity', 'Unknown')}")
                print(f"💾 Space Complexity: {result.get('space_complexity', 'Unknown')}")
                
                loop_analysis = result.get('loop_analysis', {})
                print(f"🔄 Loop Count: {loop_analysis.get('total_loops', 0)}")
                print(f"🔗 Nested Depth: {loop_analysis.get('nested_depth', 0)}")
                
                suggestions = result.get('suggestions', [])
                if suggestions:
                    print("💡 Suggestions:")
                    for suggestion in suggestions:
                        print(f"   • {suggestion}")
                
        except Exception as e:
            print(f"❌ Exception: {str(e)}")
    
    print("\n✅ Complexity Analysis Testing Complete!")

if __name__ == "__main__":
    test_complexity_analysis()
