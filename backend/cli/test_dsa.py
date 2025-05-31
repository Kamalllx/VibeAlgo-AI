
# backend/cli/test_dsa.py
#!/usr/bin/env python3
"""
CLI tool to test DSA progress tracking functionality
"""

import sys
import os
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dsa_progress_tracker import update_progress, get_progress

def test_dsa_tracking():
    """Test DSA progress tracking with sample data"""
    
    print("📊 Testing DSA Progress Tracker")
    print("=" * 50)
    
    test_user = "test_user_123"
    
    # Sample problems to track
    sample_problems = [
        {'problem_id': 'leetcode_1', 'status': 'solved', 'difficulty': 'easy', 'topic': 'arrays', 'time_spent': 15},
        {'problem_id': 'leetcode_2', 'status': 'attempted', 'difficulty': 'medium', 'topic': 'strings', 'time_spent': 30},
        {'problem_id': 'leetcode_3', 'status': 'solved', 'difficulty': 'hard', 'topic': 'dynamic_programming', 'time_spent': 45},
        {'problem_id': 'leetcode_4', 'status': 'mastered', 'difficulty': 'medium', 'topic': 'trees', 'time_spent': 25},
        {'problem_id': 'leetcode_5', 'status': 'solved', 'difficulty': 'easy', 'topic': 'arrays', 'time_spent': 10},
        {'problem_id': 'leetcode_6', 'status': 'attempted', 'difficulty': 'hard', 'topic': 'graphs', 'time_spent': 60},
    ]
    
    print(f"👤 Testing with user: {test_user}")
    print(f"📝 Adding {len(sample_problems)} sample problems...")
    
    # Update progress for each problem
    for i, problem in enumerate(sample_problems, 1):
        print(f"\n📋 Adding Problem {i}: {problem['problem_id']}")
        
        try:
            result = update_progress(
                user_id=test_user,
                problem_id=problem['problem_id'],
                status=problem['status'],
                difficulty=problem['difficulty'],
                topic=problem['topic'],
                time_spent=problem['time_spent']
            )
            
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
            else:
                print(f"✅ Updated: {problem['status']} ({problem['difficulty']}) - {problem['time_spent']}min")
                
        except Exception as e:
            print(f"❌ Exception: {str(e)}")
    
    # Get and display progress
    print(f"\n📈 Retrieving Progress for {test_user}")
    print("-" * 40)
    
    try:
        progress = get_progress(test_user)
        
        if 'error' in progress:
            print(f"❌ Error: {progress['error']}")
            return
        
        # Display statistics
        stats = progress.get('statistics', {})
        print("📊 Statistics:")
        print(f"   Total Problems: {stats.get('total_problems', 0)}")
        print(f"   Solved Problems: {stats.get('solved_problems', 0)}")
        print(f"   Mastered Problems: {stats.get('mastered_problems', 0)}")
        print(f"   Solve Rate: {stats.get('solve_rate', 0):.1f}%")
        print(f"   Total Time: {stats.get('total_time_spent', 0)} minutes")
        print(f"   Avg Attempts: {stats.get('average_attempts', 0):.1f}")
        
        # Display by difficulty
        by_difficulty = stats.get('by_difficulty', {})
        if by_difficulty:
            print("\n🎯 By Difficulty:")
            for difficulty, count in by_difficulty.items():
                print(f"   {difficulty.title()}: {count}")
        
        # Display by topic
        by_topic = stats.get('by_topic', {})
        if by_topic:
            print("\n📚 By Topic:")
            for topic, data in by_topic.items():
                solve_rate = (data['solved'] / data['total']) * 100 if data['total'] > 0 else 0
                print(f"   {topic.title()}: {data['solved']}/{data['total']} ({solve_rate:.1f}%)")
        
        # Display skill gaps
        skill_gaps = progress.get('skill_gaps', [])
        if skill_gaps:
            print("\n🎯 Areas for Improvement:")
            for gap in skill_gaps:
                print(f"   • {gap.title()}")
        
        # Display recommendations
        recommendations = progress.get('recommendations', [])
        if recommendations:
            print("\n💡 Recommendations:")
            for rec in recommendations:
                print(f"   • {rec}")
        
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
    
    print("\n✅ DSA Progress Tracking Testing Complete!")

if __name__ == "__main__":
    test_dsa_tracking()
