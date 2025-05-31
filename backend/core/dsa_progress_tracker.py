# backend/core/dsa_progress_tracker.py (Enhanced)
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

class ProblemStatus(Enum):
    ATTEMPTED = "attempted"
    SOLVED = "solved"
    MASTERED = "mastered"

class DifficultyLevel(Enum):
    EASY = "easy"
    MEDIUM = "medium" 
    HARD = "hard"

@dataclass
class ProblemProgress:
    problem_id: str
    user_id: str
    status: ProblemStatus
    difficulty: DifficultyLevel
    topic: str
    attempts: int
    time_spent: int  # in minutes
    last_attempt: str
    solution_efficiency: Optional[float] = None

    def to_dict(self):
        """Convert to JSON-serializable dict"""
        return {
            'problem_id': self.problem_id,
            'user_id': self.user_id,
            'status': self.status.value,  # Convert enum to string
            'difficulty': self.difficulty.value,  # Convert enum to string
            'topic': self.topic,
            'attempts': self.attempts,
            'time_spent': self.time_spent,
            'last_attempt': self.last_attempt,
            'solution_efficiency': self.solution_efficiency
        }

class DSAProgressTracker:
    def __init__(self):
        self.user_progress: Dict[str, List[ProblemProgress]] = {}
        self.skill_matrix = {
            'arrays': ['two_pointers', 'sliding_window', 'prefix_sum'],
            'strings': ['pattern_matching', 'string_manipulation'],
            'trees': ['traversal', 'binary_search_tree', 'balanced_trees'],
            'graphs': ['dfs', 'bfs', 'shortest_path', 'topological_sort'],
            'dynamic_programming': ['memoization', 'tabulation', 'optimization'],
            'sorting': ['comparison_sorts', 'non_comparison_sorts'],
            'searching': ['linear_search', 'binary_search', 'hash_based']
        }
    
    def update_progress(self, user_id: str, problem_id: str, status: str, 
                       difficulty: str = "medium", topic: str = "general",
                       time_spent: int = 0) -> Dict[str, Any]:
        try:
            problem_status = ProblemStatus(status)
            problem_difficulty = DifficultyLevel(difficulty)
            
            if user_id not in self.user_progress:
                self.user_progress[user_id] = []
            
            # Find existing progress or create new
            existing = next((p for p in self.user_progress[user_id] 
                           if p.problem_id == problem_id), None)
            
            if existing:
                existing.status = problem_status
                existing.attempts += 1
                existing.time_spent += time_spent
                existing.last_attempt = datetime.now().isoformat()
            else:
                new_progress = ProblemProgress(
                    problem_id=problem_id,
                    user_id=user_id,
                    status=problem_status,
                    difficulty=problem_difficulty,
                    topic=topic,
                    attempts=1,
                    time_spent=time_spent,
                    last_attempt=datetime.now().isoformat()
                )
                self.user_progress[user_id].append(new_progress)
            
            return {'message': 'Progress updated successfully'}
            
        except ValueError as e:
            return {'error': f'Invalid status or difficulty: {str(e)}'}
    
    def get_progress(self, user_id: str) -> Dict[str, Any]:
        """Get progress with JSON-serializable output"""
        if user_id not in self.user_progress:
            return {
                'problems': [],
                'statistics': {
                    'total_problems': 0,
                    'solved_problems': 0,
                    'mastered_problems': 0,
                    'solve_rate': 0,
                    'by_difficulty': {},
                    'by_topic': {},
                    'average_attempts': 0,
                    'total_time_spent': 0
                },
                'skill_gaps': [],
                'recommendations': []
            }
        user_problems = self.user_progress[user_id]
        problems_dict = [problem.to_dict() for problem in user_problems]
        stats = self._calculate_statistics(user_problems)
        skill_gaps = self._identify_skill_gaps(user_problems)
        return {
            'problems': problems_dict,
            'statistics': stats,
            'skill_gaps': skill_gaps,
            'recommendations': self._get_recommendations(user_problems)
        }
    
    def _calculate_statistics(self, problems: List[ProblemProgress]) -> Dict[str, Any]:
        total = len(problems)
        if total == 0:
            return {}
        
        solved = sum(1 for p in problems if p.status == ProblemStatus.SOLVED)
        mastered = sum(1 for p in problems if p.status == ProblemStatus.MASTERED)
        
        by_difficulty = {}
        for difficulty in DifficultyLevel:
            count = sum(1 for p in problems if p.difficulty == difficulty)
            by_difficulty[difficulty.value] = count
        
        by_topic = {}
        for problem in problems:
            topic = problem.topic
            if topic not in by_topic:
                by_topic[topic] = {'total': 0, 'solved': 0}
            by_topic[topic]['total'] += 1
            if problem.status in [ProblemStatus.SOLVED, ProblemStatus.MASTERED]:
                by_topic[topic]['solved'] += 1
        
        return {
            'total_problems': total,
            'solved_problems': solved,
            'mastered_problems': mastered,
            'solve_rate': solved / total * 100,
            'by_difficulty': by_difficulty,
            'by_topic': by_topic,
            'average_attempts': sum(p.attempts for p in problems) / total,
            'total_time_spent': sum(p.time_spent for p in problems)
        }
    
    def _identify_skill_gaps(self, problems: List[ProblemProgress]) -> List[str]:
        gaps = []
        topic_performance = {}
        
        for problem in problems:
            topic = problem.topic
            if topic not in topic_performance:
                topic_performance[topic] = {'total': 0, 'solved': 0}
            
            topic_performance[topic]['total'] += 1
            if problem.status in [ProblemStatus.SOLVED, ProblemStatus.MASTERED]:
                topic_performance[topic]['solved'] += 1
        
        for topic, performance in topic_performance.items():
            solve_rate = performance['solved'] / performance['total']
            if solve_rate < 0.6:  # Less than 60% solve rate
                gaps.append(topic)
        
        return gaps
    
    def _prepare_progress_visualization(self, problems: List[ProblemProgress]) -> Dict[str, Any]:
        # Prepare data for charts
        timeline_data = []
        skill_radar_data = {}
        
        for problem in problems:
            timeline_data.append({
                'date': problem.last_attempt,
                'status': problem.status.value,
                'difficulty': problem.difficulty.value
            })
        
        # Calculate skill levels for radar chart
        for topic in self.skill_matrix.keys():
            topic_problems = [p for p in problems if p.topic == topic]
            if topic_problems:
                solved = sum(1 for p in topic_problems 
                           if p.status in [ProblemStatus.SOLVED, ProblemStatus.MASTERED])
                skill_radar_data[topic] = (solved / len(topic_problems)) * 100
            else:
                skill_radar_data[topic] = 0
        
        return {
            'timeline': timeline_data,
            'skill_radar': skill_radar_data,
            'difficulty_distribution': {},
            'topic_heatmap': {}
        }
    
    def _get_recommendations(self, problems: List[ProblemProgress]) -> List[str]:
        recommendations = []
        
        # Analyze recent performance
        recent_failures = [p for p in problems[-10:] 
                          if p.status == ProblemStatus.ATTEMPTED]
        
        if len(recent_failures) > 5:
            recommendations.append("Consider reviewing fundamental concepts")
            recommendations.append("Practice easier problems to build confidence")
        
        # Check for topic gaps
        weak_topics = self._identify_skill_gaps(problems)
        if weak_topics:
            recommendations.append(f"Focus on improving: {', '.join(weak_topics[:3])}")
        
        return recommendations

# Global tracker instance
tracker = DSAProgressTracker()

def update_progress(user_id: str, problem_id: str, status: str, **kwargs) -> Dict[str, Any]:
    return tracker.update_progress(user_id, problem_id, status, **kwargs)

def get_progress(user_id: str) -> Dict[str, Any]:
    return tracker.get_progress(user_id)