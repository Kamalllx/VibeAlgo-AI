# backend/test_api_endpoints.py
#!/usr/bin/env python3
"""
Comprehensive API Endpoint Testing Script
Tests all endpoints of the Algorithm Intelligence Suite API
"""

import requests
import json
import time
import sys
from datetime import datetime
from typing import Dict, Any

class APITester:
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = []
        
        print(f"🧪 Algorithm Intelligence Suite API Tester")
        print(f"🌐 Base URL: {base_url}")
        print(f"⏰ Test Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
    
    def test_endpoint(self, method: str, endpoint: str, data: Dict[Any, Any] = None, 
                     expected_status: int = 200, description: str = ""):
        """Test a specific API endpoint"""
        
        url = f"{self.base_url}{endpoint}"
        test_name = f"{method} {endpoint}"
        
        print(f"\n🔍 Testing: {test_name}")
        if description:
            print(f"📝 Description: {description}")
        
        try:
            start_time = time.time()
            
            if method.upper() == "GET":
                response = self.session.get(url)
            elif method.upper() == "POST":
                response = self.session.post(url, json=data, headers={'Content-Type': 'application/json'})
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # Check status code
            status_ok = response.status_code == expected_status
            
            # Try to parse JSON response
            try:
                response_json = response.json()
                json_valid = True
            except:
                response_json = response.text
                json_valid = False
            
            # Record results
            result = {
                'test_name': test_name,
                'description': description,
                'status_code': response.status_code,
                'expected_status': expected_status,
                'status_ok': status_ok,
                'response_time': response_time,
                'json_valid': json_valid,
                'response_size': len(response.content),
                'success': status_ok and json_valid
            }
            
            self.test_results.append(result)
            
            # Print results
            status_icon = "✅" if status_ok else "❌"
            json_icon = "📄" if json_valid else "📝"
            
            print(f"   {status_icon} Status: {response.status_code} (expected {expected_status})")
            print(f"   {json_icon} Response: {'Valid JSON' if json_valid else 'Text/HTML'}")
            print(f"   ⏱️ Time: {response_time:.3f}s")
            print(f"   📦 Size: {len(response.content)} bytes")
            
            if status_ok and json_valid and isinstance(response_json, dict):
                # Show key response fields
                key_fields = ['status', 'success', 'session_id', 'algorithm_detected', 'files_generated']
                relevant_fields = {k: v for k, v in response_json.items() if k in key_fields}
                if relevant_fields:
                    print(f"   🔑 Key fields: {relevant_fields}")
            
            if not status_ok:
                print(f"   ⚠️ Error: {response_json if json_valid else response.text[:200]}")
            
            return result
            
        except Exception as e:
            print(f"   💥 Exception: {str(e)}")
            result = {
                'test_name': test_name,
                'description': description,
                'error': str(e),
                'success': False
            }
            self.test_results.append(result)
            return result
    
    def run_all_tests(self):
        """Run comprehensive test suite"""
        
        print("\n🚀 Starting Comprehensive API Test Suite")
        print("=" * 60)
        
        # Test 1: API Status
        self.test_endpoint(
            "GET", "/api/status",
            description="Check API health and system status"
        )
        
        # Test 2: Home Page
        self.test_endpoint(
            "GET", "/",
            description="API documentation homepage"
        )
        
        # Test 3: List Algorithms
        self.test_endpoint(
            "GET", "/api/algorithms",
            description="List all available algorithms in database"
        )
        
        # Test 4: Performance Analysis
        self.test_endpoint(
            "GET", "/api/performance",
            description="Generate comprehensive performance analysis plots"
        )
        
        # Test 5: Simple Algorithm Analysis
        self.test_endpoint(
            "POST", "/api/analyze",
            data={
                "input": "binary search algorithm",
                "input_type": "problem",
                "options": {
                    "include_visualization": True,
                    "include_performance": False
                }
            },
            description="Complete analysis of binary search problem"
        )
        
        # Test 6: Code Complexity Analysis
        self.test_endpoint(
            "POST", "/api/complexity",
            data={
                "code": """
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
""",
                "language": "python"
            },
            description="Detailed complexity analysis of binary search code"
        )
        
        # Test 7: Problem Solving
        self.test_endpoint(
            "POST", "/api/solve",
            data={
                "problem": "Find the maximum element in an array",
                "context": "Given an array of integers, return the maximum value"
            },
            description="Algorithm problem solving with AI"
        )
        
        # Test 8: Visualization Generation
        self.test_endpoint(
            "POST", "/api/visualize",
            data={
                "algorithm": "bubble sort",
                "data": "[5, 2, 8, 1, 9]"
            },
            description="Generate bubble sort visualization"
        )
        
        # Test 9: Complex Algorithm Analysis
        self.test_endpoint(
            "POST", "/api/analyze",
            data={
                "input": """
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
""",
                "input_type": "code",
                "options": {
                    "include_visualization": True,
                    "include_performance": True
                }
            },
            description="Complete analysis of quicksort implementation"
        )
        
        # Test 10: Edge Cases
        
        # Missing required fields
        self.test_endpoint(
            "POST", "/api/analyze",
            data={},
            expected_status=400,
            description="Test missing required fields (should fail)"
        )
        
        # Invalid JSON
        try:
            response = requests.post(f"{self.base_url}/api/analyze", 
                                   data="invalid json", 
                                   headers={'Content-Type': 'application/json'})
            print(f"\n🔍 Testing: POST /api/analyze (Invalid JSON)")
            print(f"   ❌ Status: {response.status_code} (expected 400)")
        except:
            print(f"\n🔍 Testing: POST /api/analyze (Invalid JSON)")
            print(f"   ✅ Request properly rejected")
        
        # Non-existent endpoint
        self.test_endpoint(
            "GET", "/api/nonexistent",
            expected_status=404,
            description="Test non-existent endpoint (should return 404)"
        )
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        
        print("\n" + "=" * 60)
        print("📊 TEST REPORT SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results if result.get('success', False))
        failed_tests = total_tests - successful_tests
        
        print(f"📈 Total Tests: {total_tests}")
        print(f"✅ Successful: {successful_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"📊 Success Rate: {(successful_tests/total_tests*100):.1f}%")
        
        # Average response time
        response_times = [r.get('response_time', 0) for r in self.test_results if 'response_time' in r]
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            print(f"⏱️ Average Response Time: {avg_response_time:.3f}s")
        
        # Failed tests details
        if failed_tests > 0:
            print(f"\n❌ FAILED TESTS:")
            for result in self.test_results:
                if not result.get('success', False):
                    print(f"   • {result['test_name']}: {result.get('error', 'Status/JSON error')}")
        
        # Test performance table
        print(f"\n📋 DETAILED RESULTS:")
        print(f"{'Test Name':<35} {'Status':<8} {'Time(s)':<8} {'Size(B)':<8} {'Result':<8}")
        print("-" * 75)
        
        for result in self.test_results:
            name = result['test_name'][:34]
            status = str(result.get('status_code', 'ERR'))
            time_str = f"{result.get('response_time', 0):.3f}"
            size_str = str(result.get('response_size', 0))
            result_str = "PASS" if result.get('success', False) else "FAIL"
            
            print(f"{name:<35} {status:<8} {time_str:<8} {size_str:<8} {result_str:<8}")
        
        print("\n" + "=" * 60)
        print(f"🎯 Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'failed_tests': failed_tests,
            'success_rate': successful_tests/total_tests*100,
            'average_response_time': avg_response_time if response_times else 0,
            'test_results': self.test_results
        }

def main():
    """Main testing function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Algorithm Intelligence Suite API Tester")
    parser.add_argument('--url', default='http://localhost:5000', help='API base URL')
    parser.add_argument('--save-report', action='store_true', help='Save test report to file')
    
    args = parser.parse_args()
    
    # Check if API is running
    try:
        response = requests.get(f"{args.url}/api/status", timeout=5)
        if response.status_code != 200:
            print(f"❌ API not responding correctly at {args.url}")
            print(f"   Status: {response.status_code}")
            sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to API at {args.url}")
        print(f"   Error: {e}")
        print(f"\n💡 Make sure the API server is running:")
        print(f"   python app.py")
        sys.exit(1)
    
    # Run tests
    tester = APITester(args.url)
    tester.run_all_tests()
    report = tester.generate_test_report()
    
    # Save report if requested
    if args.save_report:
        report_filename = f"api_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n💾 Test report saved to: {report_filename}")
    
    # Exit with appropriate code
    if report['failed_tests'] > 0:
        sys.exit(1)
    else:
        print(f"\n🎉 All tests passed successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()
