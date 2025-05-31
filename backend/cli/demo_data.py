
# backend/cli/demo_data.py
#!/usr/bin/env python3
"""
Demo data generator for testing all backend functionality
"""

import sys
import os
import json
from datetime import datetime, timedelta
import random

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def generate_demo_code_samples():
    """Generate demo code samples for complexity analysis"""
    return [
        {
            'name': 'Simple O(1) Operation',
            'code': 'return arr[0] if arr else None',
            'expected_complexity': 'O(1)'
        },
        {
            'name': 'Linear Search O(n)',
            'code': '''
def linear_search(arr, target):
    for i, val in enumerate(arr):
        if val == target:
            return i
    return -1
            '''.strip(),
            'expected_complexity': 'O(n)'
        },
        {
            'name': 'Binary Search O(log n)',
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
        },
        {
            'name': 'Bubble Sort O(n²)',
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
        }
    ]

def generate_demo_dsa_progress():
    """Generate demo DSA progress data"""
    topics = ['arrays', 'strings', 'trees', 'graphs', 'dynamic_programming', 'sorting', 'searching']
    difficulties = ['easy', 'medium', 'hard']
    statuses = ['attempted', 'solved', 'mastered']
    
    problems = []
    for i in range(20):
        problems.append({
            'problem_id': f'leetcode_{i+1}',
            'status': random.choice(statuses),
            'difficulty': random.choice(difficulties),
            'topic': random.choice(topics),
            'time_spent': random.randint(5, 90)
        })
    
    return problems

def generate_demo_contest_data():
    """Generate demo contest submissions"""
    problems = ['A', 'B', 'C', 'D', 'E', 'F']
    statuses = ['solved', 'attempted', 'solved', 'attempted', 'solved']  # Bias toward solved
    
    base_time = datetime.now() - timedelta(hours=3)
    submissions = []
    cumulative_time = 0
    
    for i, problem in enumerate(problems):
        time_taken = random.randint(10, 60)
        cumulative_time += time_taken
        
        if cumulative_time > 180:  # 3-hour contest
            break
        
        status = random.choice(statuses)
        score = random.randint(500, 2000) if status == 'solved' else random.randint(0, 200)
        
        submissions.append({
            'problem_id': problem,
            'submission_time': (base_time + timedelta(minutes=cumulative_time)).isoformat(),
            'time_taken': time_taken,
            'status': status,
            'score': float(score),
            'attempts': random.randint(1, 3)
        })
    
    return submissions

def save_demo_data():
    """Save all demo data to files"""
    demo_data = {
        'code_samples': generate_demo_code_samples(),
        'dsa_progress': generate_demo_dsa_progress(),
        'contest_submissions': generate_demo_contest_data(),
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'version': '1.0'
        }
    }
    
    # Save to file
    output_file = os.path.join(os.path.dirname(__file__), 'demo_data.json')
    with open(output_file, 'w') as f:
        json.dump(demo_data, f, indent=2)
    
    print(f"📁 Demo data saved to: {output_file}")
    return demo_data

if __name__ == "__main__":
    print("🎲 Generating Demo Data")
    print("=" * 30)
    
    demo_data = save_demo_data()
    
    print(f"✅ Generated {len(demo_data['code_samples'])} code samples")
    print(f"✅ Generated {len(demo_data['dsa_progress'])} DSA problems")
    print(f"✅ Generated {len(demo_data['contest_submissions'])} contest submissions")
    print("\n🎯 Demo data ready for testing!")
