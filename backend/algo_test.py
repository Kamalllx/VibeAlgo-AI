# backend/test_algorithm_solver.py
#!/usr/bin/env python3
import requests
import json

API_BASE = "http://127.0.0.1:5000/api"

def test_algorithm_solver():
    print("🧮 Testing Algorithm Solver Agent")
    print("=" * 50)
    
    # Test problem
    test_problem = """
    Problem: Two Sum
    
    Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
    
    You may assume that each input would have exactly one solution, and you may not use the same element twice.
    
    Example:
    Input: nums = [2,7,11,15], target = 9
    Output: [0,1]
    Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].
    """
    
    # User's suboptimal solution
    user_solution = """
def twoSum(nums, target):
    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []
    """
    
    payload = {
        "problem": test_problem,
        "user_solution": user_solution
    }
    
    print("📤 Sending problem to algorithm solver...")
    response = requests.post(f"{API_BASE}/algorithm-solver/solve", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Algorithm solver successful!")
        
        solution = result["algorithm_solution"]
        print(f"\n📊 Problem Analysis:")
        print(f"   Type: {solution['problem_analysis'].get('problem_type')}")
        print(f"   Difficulty: {solution['problem_analysis'].get('difficulty')}")
        
        print(f"\n🎯 Algorithmic Approaches: {len(solution['algorithmic_approaches'])}")
        for i, approach in enumerate(solution['algorithmic_approaches'], 1):
            print(f"   {i}. {approach.get('name')} - {approach.get('time_complexity')}")
        
        print(f"\n💻 Generated Solution Available: {'Yes' if solution['optimal_solution'] else 'No'}")
        
        if solution.get('user_solution_comparison'):
            print(f"\n🔍 User Solution Comparison: Available")
        
    else:
        print(f"❌ Request failed: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_algorithm_solver()
