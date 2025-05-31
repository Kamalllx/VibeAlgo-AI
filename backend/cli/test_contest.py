
# backend/cli/test_contest.py
#!/usr/bin/env python3
"""
CLI tool to test contest optimization functionality
"""

import sys
import os
import json
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.contest_optimizer import optimize_contest

def test_contest_optimization():
    """Test contest optimization with sample data"""
    
    print("🏆 Testing Contest Strategy Optimizer")
    print("=" * 50)
    
    contest_id = "codeforces_round_900"
    
    # Sample contest submissions
    base_time = datetime.now() - timedelta(hours=2)
    sample_submissions = [
        {
            'problem_id': 'A',
            'submission_time': (base_time + timedelta(minutes=10)).isoformat(),
            'time_taken': 15,
            'status': 'solved',
            'score': 500.0,
            'attempts': 1
        },
        {
            'problem_id': 'B',
            'submission_time': (base_time + timedelta(minutes=35)).isoformat(),
            'time_taken': 25,
            'status': 'solved',
            'score': 1000.0,
            'attempts': 2
        },
        {
            'problem_id': 'C',
            'submission_time': (base_time + timedelta(minutes=75)).isoformat(),
            'time_taken': 40,
            'status': 'attempted',
            'score': 0.0,
            'attempts': 3
        },
        {
            'problem_id': 'D',
            'submission_time': (base_time + timedelta(minutes=120)).isoformat(),
            'time_taken': 45,
            'status': 'attempted',
            'score': 200.0,
            'attempts': 2
        },
        {
            'problem_id': 'E',
            'submission_time': (base_time + timedelta(minutes=150)).isoformat(),
            'time_taken': 30,
            'status': 'solved',
            'score': 1500.0,
            'attempts': 1
        }
    ]
    
    print(f"🎯 Contest: {contest_id}")
    print(f"📝 Analyzing {len(sample_submissions)} submissions...")
    
    # Display submission summary
    print("\n📋 Submissions Summary:")
    for i, sub in enumerate(sample_submissions, 1):
        status_emoji = "✅" if sub['status'] == 'solved' else "❌"
        print(f"   {i}. Problem {sub['problem_id']}: {status_emoji} {sub['status']} "
              f"({sub['time_taken']}min, {sub['score']} pts, {sub['attempts']} attempts)")
    
    try:
        print(f"\n🔍 Running Contest Optimization...")
        result = optimize_contest(contest_id, sample_submissions)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            return
        
        # Display performance analysis
        perf = result.get('performance_analysis', {})
        print("\n📊 Performance Analysis:")
        print(f"   Total Problems: {perf.get('total_problems', 0)}")
        print(f"   Solved: {perf.get('solved_problems', 0)}")
        print(f"   Success Rate: {perf.get('success_rate', 0):.1f}%")
        print(f"   Total Score: {perf.get('total_score', 0)}")
        print(f"   Total Time: {perf.get('total_time_spent', 0)} minutes")
        print(f"   Avg Solve Time: {perf.get('average_solve_time', 0):.1f} minutes")
        print(f"   Efficiency Score: {perf.get('efficiency_score', 0):.1f}/100")
        
        # Display time strategy
        time_strategy = result.get('time_strategy', {})
        print("\n⏰ Time Strategy:")
        print(f"   Recommended Time/Problem: {time_strategy.get('recommended_time_per_problem', 0):.1f} min")
        print(f"   Estimated Solvable Problems: {time_strategy.get('estimated_solvable_problems', 0)}")
        print(f"   Strategy Type: {time_strategy.get('strategy_type', 'unknown').title()}")
        
        phase_allocation = time_strategy.get('phase_allocation', {})
        if phase_allocation:
            print("   Phase Allocation:")
            for phase, time in phase_allocation.items():
                print(f"     • {phase.replace('_', ' ').title()}: {time} min")
        
        # Display problem priorities
        priorities = result.get('problem_priority', [])
        if priorities:
            print("\n🎯 Problem Priorities:")
            for i, prob in enumerate(priorities[:5], 1):  # Show top 5
                status_indicator = "✅" if prob['solved'] else "⚠️"
                print(f"   {i}. Problem {prob['problem_id']}: {status_indicator} "
                      f"Priority {prob['priority_score']:.1f}/10 "
                      f"(Score: {prob['score']}, Time: {prob['total_time']}min)")
        
        # Display optimization metrics
        opt_metrics = result.get('optimization_metrics', {})
        if opt_metrics:
            print("\n📈 Optimization Metrics:")
            
            time_dist = opt_metrics.get('time_distribution', {})
            if time_dist:
                print("   Time Distribution:")
                for time_range, count in time_dist.items():
                    print(f"     • {time_range} min: {count} problems")
            
            print(f"   Score Efficiency: {opt_metrics.get('score_efficiency', 0):.2f} pts/min")
            print(f"   Attempt Efficiency: {opt_metrics.get('attempt_efficiency', 0):.1f}%")
            print(f"   Completion Rate: {opt_metrics.get('completion_rate', 0):.1f}%")
        
        # Display recommendations
        recommendations = result.get('recommendations', [])
        if recommendations:
            print("\n💡 Strategic Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        
        # Display strategy recommendations
        strategy_recs = result.get('strategy_recommendations', [])
        if strategy_recs:
            print("\n🎲 Strategy Recommendations:")
            for i, rec in enumerate(strategy_recs, 1):
                print(f"   {i}. {rec.get('type', 'unknown').title()}: {rec.get('suggestion', 'No suggestion')}")
        
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Contest Optimization Testing Complete!")

if __name__ == "__main__":
    test_contest_optimization()
