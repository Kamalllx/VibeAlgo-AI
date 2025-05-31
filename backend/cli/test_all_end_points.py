
# backend/cli/test_all_endpoints.py (Corrected version)
#!/usr/bin/env python3
"""
Comprehensive testing script for all backend endpoints
"""

import sys
import os
import json
import requests
import time
from datetime import datetime, timedelta
import random

# Configuration
API_BASE_URL = 'http://127.0.0.1:5000/api'
TEST_USER_ID = 'test_user_123'

def generate_test_data():
    """Generate test data for all endpoints"""
    return {
        'code_samples': [
            {
                'name': 'Simple Linear Loop',
                'code': '''
for i in range(n):
    print(i)
                '''.strip(),
                'expected_complexity': 'O(n)'
            },
            {
                'name': 'Nested Loops (Bubble Sort)',
                'code': '''
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
                '''.strip(),
                'expected_complexity': 'O(n²)'
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
                '''.strip(),
                'expected_complexity': 'O(log n)'
            }
        ],
        'dsa_problems': [
            {'problem_id': 'leetcode_1', 'status': 'solved', 'difficulty': 'easy', 'topic': 'arrays', 'time_spent': 15},
            {'problem_id': 'leetcode_2', 'status': 'attempted', 'difficulty': 'medium', 'topic': 'strings', 'time_spent': 30},
            {'problem_id': 'leetcode_3', 'status': 'solved', 'difficulty': 'hard', 'topic': 'dynamic_programming', 'time_spent': 45},
            {'problem_id': 'leetcode_4', 'status': 'mastered', 'difficulty': 'medium', 'topic': 'trees', 'time_spent': 25},
            {'problem_id': 'leetcode_5', 'status': 'solved', 'difficulty': 'easy', 'topic': 'arrays', 'time_spent': 10}
        ],
        'contest_submissions': [
            {
                'problem_id': 'A',
                'submission_time': (datetime.now() - timedelta(minutes=120)).isoformat(),
                'time_taken': 15,
                'status': 'solved',
                'score': 500.0,
                'attempts': 1
            },
            {
                'problem_id': 'B', 
                'submission_time': (datetime.now() - timedelta(minutes=90)).isoformat(),
                'time_taken': 30,
                'status': 'solved',
                'score': 1000.0,
                'attempts': 2
            },
            {
                'problem_id': 'C',
                'submission_time': (datetime.now() - timedelta(minutes=45)).isoformat(),
                'time_taken': 45,
                'status': 'attempted',
                'score': 0.0,
                'attempts': 3
            }
        ]
    }

def test_health_endpoints():
    """Test health check endpoints"""
    print("\n🏥 Testing Health Endpoints")
    print("-" * 40)
    
    try:
        # Basic health check
        response = requests.get(f"{API_BASE_URL}/health/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Basic Health Check: {data.get('status', 'unknown')}")
        else:
            print(f"❌ Basic Health Check Failed: {response.status_code}")
        
        # Detailed health check
        response = requests.get(f"{API_BASE_URL}/health/detailed", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Detailed Health Check: {data.get('status', 'unknown')}")
            
            # Show system metrics
            metrics = data.get('system_metrics', {})
            print(f"   CPU Usage: {metrics.get('cpu_usage_percent', 0):.1f}%")
            print(f"   Memory Usage: {metrics.get('memory_usage_percent', 0):.1f}%")
            
            # Show service status
            services = data.get('services', {})
            for service, status in services.items():
                status_emoji = "✅" if status.get('status') == 'healthy' else "❌"
                print(f"   {service}: {status_emoji}")
        else:
            print(f"❌ Detailed Health Check Failed: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Health Check Error: {str(e)}")
        return False
    
    return True

def test_complexity_endpoints(code_samples):
    """Test complexity analysis endpoints"""
    print("\n🔍 Testing Complexity Analysis Endpoints")
    print("-" * 40)
    
    success_count = 0
    
    for i, sample in enumerate(code_samples, 1):
        print(f"\n📋 Test {i}: {sample['name']}")
        
        try:
            payload = {
                'code': sample['code'],
                'language': 'python',
                'platform': 'general'
            }
            
            response = requests.post(f"{API_BASE_URL}/complexity/analyze", json=payload, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    result = data.get('data', {})
                    time_complexity = result.get('time_complexity', 'Unknown')
                    space_complexity = result.get('space_complexity', 'Unknown')
                    
                    print(f"✅ Analysis Success")
                    print(f"   Time Complexity: {time_complexity}")
                    print(f"   Space Complexity: {space_complexity}")
                    print(f"   Expected: {sample['expected_complexity']}")
                    
                    # Check suggestions
                    suggestions = result.get('suggestions', [])
                    if suggestions:
                        print(f"   Suggestions: {len(suggestions)} provided")
                    
                    success_count += 1
                else:
                    print(f"❌ Analysis Failed: {data.get('error', 'Unknown error')}")
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request Error: {str(e)}")
    
    print(f"\n📊 Complexity Analysis Summary: {success_count}/{len(code_samples)} successful")
    return success_count == len(code_samples)

def test_dsa_progress_endpoints(dsa_problems):
    """Test DSA progress tracking endpoints"""
    print("\n📊 Testing DSA Progress Endpoints")
    print("-" * 40)
    
    success_count = 0
    
    # Test progress updates
    for i, problem in enumerate(dsa_problems, 1):
        print(f"\n📝 Updating Problem {i}: {problem['problem_id']}")
        
        try:
            payload = {
                'problem_id': problem['problem_id'],
                'status': problem['status'],
                'difficulty': problem['difficulty'],
                'topic': problem['topic'],
                'time_spent': problem['time_spent']
            }
            
            response = requests.post(
                f"{API_BASE_URL}/dsa/progress/{TEST_USER_ID}/update", 
                json=payload, 
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    print(f"✅ Update Success: {problem['status']} ({problem['difficulty']})")
                    success_count += 1
                else:
                    print(f"❌ Update Failed: {data.get('error', 'Unknown error')}")
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request Error: {str(e)}")
    
    # Test progress retrieval
    print(f"\n📈 Retrieving Progress for {TEST_USER_ID}")
    
    try:
        response = requests.get(f"{API_BASE_URL}/dsa/progress/{TEST_USER_ID}", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                progress_data = data.get('data', {})
                stats = progress_data.get('statistics', {})
                
                print("✅ Progress Retrieval Success")
                print(f"   Total Problems: {stats.get('total_problems', 0)}")
                print(f"   Solved Problems: {stats.get('solved_problems', 0)}")
                print(f"   Solve Rate: {stats.get('solve_rate', 0):.1f}%")
                print(f"   Total Time: {stats.get('total_time_spent', 0)} minutes")
                
                # Show skill gaps
                skill_gaps = progress_data.get('skill_gaps', [])
                if skill_gaps:
                    print(f"   Skill Gaps: {', '.join(skill_gaps)}")
                
                success_count += 1
            else:
                print(f"❌ Retrieval Failed: {data.get('error', 'Unknown error')}")
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request Error: {str(e)}")
    
    total_tests = len(dsa_problems) + 1  # +1 for retrieval test
    print(f"\n📊 DSA Progress Summary: {success_count}/{total_tests} successful")
    return success_count == total_tests

def test_contest_endpoints(contest_submissions):
    """Test contest optimization endpoints"""
    print("\n🏆 Testing Contest Optimization Endpoints")
    print("-" * 40)
    
    contest_id = 'test_contest_123'
    
    try:
        payload = {
            'contest_id': contest_id,
            'submissions': contest_submissions
        }
        
        response = requests.post(f"{API_BASE_URL}/contest/optimize", json=payload, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                result = data.get('data', {})
                
                print("✅ Contest Optimization Success")
                
                # Show performance analysis
                perf = result.get('performance_analysis', {})
                print(f"   Success Rate: {perf.get('success_rate', 0):.1f}%")
                print(f"   Total Score: {perf.get('total_score', 0)}")
                print(f"   Efficiency Score: {perf.get('efficiency_score', 0):.1f}/100")
                
                # Show time strategy
                time_strategy = result.get('time_strategy', {})
                recommended_time = time_strategy.get('recommended_time_per_problem', 0)
                print(f"   Recommended Time/Problem: {recommended_time:.1f} min")
                
                # Show recommendations
                recommendations = result.get('recommendations', [])
                print(f"   Recommendations: {len(recommendations)} provided")
                
                return True
            else:
                print(f"❌ Optimization Failed: {data.get('error', 'Unknown error')}")
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request Error: {str(e)}")
    
    return False

def main():
    """Main testing function"""
    print("🚀 Starting Comprehensive Backend Testing")
    print("=" * 50)
    print(f"🎯 API Base URL: {API_BASE_URL}")
    print(f"👤 Test User: {TEST_USER_ID}")
    
    # Generate test data
    test_data = generate_test_data()
    
    # Test results
    results = {
        'health': False,
        'complexity': False,
        'dsa_progress': False,
        'contest': False
    }
    
    # Run tests
    results['health'] = test_health_endpoints()
    time.sleep(1)  # Brief pause between tests
    
    if results['health']:
        results['complexity'] = test_complexity_endpoints(test_data['code_samples'])
        time.sleep(1)
        
        results['dsa_progress'] = test_dsa_progress_endpoints(test_data['dsa_problems'])
        time.sleep(1)
        
        results['contest'] = test_contest_endpoints(test_data['contest_submissions'])
    else:
        print("\n❌ Skipping other tests due to health check failure")
    
    # Final summary
    print("\n" + "=" * 50)
    print("🎯 FINAL TEST SUMMARY")
    print("=" * 50)
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)
    
    for test_name, passed in results.items():
        status_emoji = "✅" if passed else "❌"
        print(f"{status_emoji} {test_name.replace('_', ' ').title()}: {'PASSED' if passed else 'FAILED'}")
    
    print(f"\n📊 Overall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! Backend is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Check the logs above for details.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
