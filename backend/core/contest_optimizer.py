# backend/core/contest_optimizer.py
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import statistics

@dataclass
class ContestSubmission:
    problem_id: str
    submission_time: datetime
    time_taken: int  # minutes
    status: str  # 'solved', 'attempted', 'partial'
    score: float
    attempts: int = 1

class ContestOptimizer:
    def __init__(self):
        self.strategy_weights = {
            'time_efficiency': 0.4,
            'problem_difficulty': 0.3,
            'success_rate': 0.3
        }
    
    def optimize_contest(self, contest_id: str, submissions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize contest strategy based on submission history"""
        try:
            # Convert submissions to objects
            submission_objects = []
            for sub in submissions:
                submission_objects.append(ContestSubmission(
                    problem_id=sub.get('problem_id', ''),
                    submission_time=datetime.fromisoformat(sub.get('submission_time', datetime.now().isoformat())),
                    time_taken=sub.get('time_taken', 0),
                    status=sub.get('status', 'attempted'),
                    score=sub.get('score', 0.0),
                    attempts=sub.get('attempts', 1)
                ))
            
            # Analyze performance patterns
            performance_analysis = self._analyze_performance(submission_objects)
            
            # Generate time allocation strategy
            time_strategy = self._generate_time_strategy(submission_objects)
            
            # Identify problem prioritization
            problem_priority = self._prioritize_problems(submission_objects)
            
            # Calculate optimization metrics
            optimization_metrics = self._calculate_optimization_metrics(submission_objects)
            
            return {
                'contest_id': contest_id,
                'performance_analysis': performance_analysis,
                'time_strategy': time_strategy,
                'problem_priority': problem_priority,
                'optimization_metrics': optimization_metrics,
                'recommendations': self._generate_recommendations(performance_analysis),
                'visualization_data': self._prepare_contest_visualization(submission_objects)
            }
            
        except Exception as e:
            return {'error': f'Contest optimization failed: {str(e)}'}
    
    def _analyze_performance(self, submissions: List[ContestSubmission]) -> Dict[str, Any]:
        """Analyze contest performance patterns"""
        if not submissions:
            return {'error': 'No submissions to analyze'}
        
        solved_submissions = [s for s in submissions if s.status == 'solved']
        attempted_submissions = [s for s in submissions if s.status == 'attempted']
        
        total_time = sum(s.time_taken for s in submissions)
        total_score = sum(s.score for s in submissions)
        
        success_rate = len(solved_submissions) / len(submissions) * 100 if submissions else 0
        
        # Time analysis
        if solved_submissions:
            avg_solve_time = statistics.mean([s.time_taken for s in solved_submissions])
            time_consistency = statistics.stdev([s.time_taken for s in solved_submissions]) if len(solved_submissions) > 1 else 0
        else:
            avg_solve_time = 0
            time_consistency = 0
        
        return {
            'total_problems': len(submissions),
            'solved_problems': len(solved_submissions),
            'attempted_problems': len(attempted_submissions),
            'success_rate': success_rate,
            'total_time_spent': total_time,
            'total_score': total_score,
            'average_solve_time': avg_solve_time,
            'time_consistency': time_consistency,
            'efficiency_score': self._calculate_efficiency_score(submissions)
        }
    
    def _calculate_efficiency_score(self, submissions: List[ContestSubmission]) -> float:
        """Calculate overall efficiency score"""
        if not submissions:
            return 0.0
        
        solved_submissions = [s for s in submissions if s.status == 'solved']
        if not solved_submissions:
            return 0.0
        
        # Efficiency = (Total Score / Total Time) * Success Rate
        total_score = sum(s.score for s in solved_submissions)
        total_time = sum(s.time_taken for s in solved_submissions)
        success_rate = len(solved_submissions) / len(submissions)
        
        if total_time == 0:
            return 0.0
        
        efficiency = (total_score / total_time) * success_rate * 100
        return min(efficiency, 100.0)  # Cap at 100
    
    def _generate_time_strategy(self, submissions: List[ContestSubmission]) -> Dict[str, Any]:
        """Generate optimal time allocation strategy"""
        if not submissions:
            return {'strategy': 'insufficient_data'}
        
        solved_submissions = [s for s in submissions if s.status == 'solved']
        
        if solved_submissions:
            avg_solve_time = statistics.mean([s.time_taken for s in solved_submissions])
            recommended_time_per_problem = min(avg_solve_time * 1.2, 45)  # 20% buffer, max 45 mins
        else:
            recommended_time_per_problem = 30  # Default 30 minutes
        
        total_contest_time = 180  # Assume 3-hour contest
        estimated_problems = total_contest_time // recommended_time_per_problem
        
        return {
            'recommended_time_per_problem': recommended_time_per_problem,
            'estimated_solvable_problems': int(estimated_problems),
            'time_buffer_percentage': 15,
            'strategy_type': 'balanced',
            'phase_allocation': {
                'reading_problems': 15,  # minutes
                'solving_phase': total_contest_time - 30,
                'review_phase': 15
            }
        }
    
    def _prioritize_problems(self, submissions: List[ContestSubmission]) -> List[Dict[str, Any]]:
        """Prioritize problems based on solving patterns"""
        problem_stats = {}
        
        for submission in submissions:
            pid = submission.problem_id
            if pid not in problem_stats:
                problem_stats[pid] = {
                    'problem_id': pid,
                    'total_attempts': 0,
                    'total_time': 0,
                    'solved': False,
                    'score': 0,
                    'priority_score': 0
                }
            
            stats = problem_stats[pid]
            stats['total_attempts'] += submission.attempts
            stats['total_time'] += submission.time_taken
            stats['score'] = max(stats['score'], submission.score)
            
            if submission.status == 'solved':
                stats['solved'] = True
        
        # Calculate priority scores
        for stats in problem_stats.values():
            if stats['solved']:
                # High priority for solved problems (for review/optimization)
                stats['priority_score'] = 8.0
            elif stats['total_attempts'] > 0:
                # Medium priority for attempted problems
                efficiency = stats['score'] / max(stats['total_time'], 1)
                stats['priority_score'] = min(efficiency * 5, 7.0)
            else:
                # Low priority for unvisited problems
                stats['priority_score'] = 3.0
        
        # Sort by priority score (descending)
        prioritized = sorted(problem_stats.values(), key=lambda x: x['priority_score'], reverse=True)
        
        return prioritized
    
    def _calculate_optimization_metrics(self, submissions: List[ContestSubmission]) -> Dict[str, Any]:
        """Calculate various optimization metrics"""
        if not submissions:
            return {}
        
        # Time distribution analysis
        time_buckets = {'0-15': 0, '15-30': 0, '30-45': 0, '45+': 0}
        for submission in submissions:
            time = submission.time_taken
            if time <= 15:
                time_buckets['0-15'] += 1
            elif time <= 30:
                time_buckets['15-30'] += 1
            elif time <= 45:
                time_buckets['30-45'] += 1
            else:
                time_buckets['45+'] += 1
        
        # Score efficiency
        total_score = sum(s.score for s in submissions)
        total_time = sum(s.time_taken for s in submissions)
        score_efficiency = total_score / max(total_time, 1)
        
        return {
            'time_distribution': time_buckets,
            'score_efficiency': score_efficiency,
            'attempt_efficiency': len([s for s in submissions if s.attempts == 1]) / len(submissions) * 100,
            'completion_rate': len([s for s in submissions if s.status == 'solved']) / len(submissions) * 100
        }
    
    def _generate_recommendations(self, performance_analysis: Dict[str, Any]) -> List[str]:
        """Generate strategic recommendations"""
        recommendations = []
        
        success_rate = performance_analysis.get('success_rate', 0)
        avg_solve_time = performance_analysis.get('average_solve_time', 0)
        efficiency_score = performance_analysis.get('efficiency_score', 0)
        
        if success_rate < 50:
            recommendations.append("Focus on solving easier problems first to build confidence")
            recommendations.append("Practice fundamental concepts before attempting contest problems")
        
        if avg_solve_time > 40:
            recommendations.append("Work on improving problem-solving speed")
            recommendations.append("Practice time-bound problem solving")
        
        if efficiency_score < 30:
            recommendations.append("Balance between speed and accuracy")
            recommendations.append("Consider strategic problem selection")
        
        if not recommendations:
            recommendations.append("Maintain current performance level")
            recommendations.append("Focus on consistency and advanced problem types")
        
        return recommendations
    
    def _prepare_contest_visualization(self, submissions: List[ContestSubmission]) -> Dict[str, Any]:
        """Prepare data for contest visualization"""
        timeline_data = []
        performance_radar = {}
        
        # Timeline data
        for submission in submissions:
            timeline_data.append({
                'time': submission.submission_time.isoformat(),
                'problem': submission.problem_id,
                'status': submission.status,
                'score': submission.score,
                'time_taken': submission.time_taken
            })
        
        # Performance radar data
        metrics = ['speed', 'accuracy', 'consistency', 'problem_variety', 'time_management']
        for metric in metrics:
            performance_radar[metric] = self._calculate_radar_metric(submissions, metric)
        
        return {
            'timeline': timeline_data,
            'performance_radar': performance_radar,
            'score_progression': [s.score for s in submissions],
            'time_efficiency_scatter': [(s.time_taken, s.score) for s in submissions]
        }
    
    def _calculate_radar_metric(self, submissions: List[ContestSubmission], metric: str) -> float:
        """Calculate specific radar chart metrics"""
        if not submissions:
            return 0.0
        
        if metric == 'speed':
            solved = [s for s in submissions if s.status == 'solved']
            if not solved:
                return 0.0
            avg_time = statistics.mean([s.time_taken for s in solved])
            return max(0, 100 - avg_time * 2)  # Inverse of time
        
        elif metric == 'accuracy':
            return len([s for s in submissions if s.status == 'solved']) / len(submissions) * 100
        
        elif metric == 'consistency':
            scores = [s.score for s in submissions]
            if len(scores) < 2:
                return 50.0
            std_dev = statistics.stdev(scores)
            mean_score = statistics.mean(scores)
            return max(0, 100 - (std_dev / max(mean_score, 1)) * 50)
        
        elif metric == 'problem_variety':
            unique_problems = len(set(s.problem_id for s in submissions))
            return min(unique_problems * 20, 100)  # Scale based on variety
        
        elif metric == 'time_management':
            total_time = sum(s.time_taken for s in submissions)
            efficient_submissions = len([s for s in submissions if s.time_taken <= 30])
            return efficient_submissions / len(submissions) * 100
        
        return 50.0  # Default value

# Global optimizer instance
optimizer = ContestOptimizer()

def optimize_contest(contest_id: str, submissions: List[Dict[str, Any]]) -> Dict[str, Any]:
    return optimizer.optimize_contest(contest_id, submissions)
