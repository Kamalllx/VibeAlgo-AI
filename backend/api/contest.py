# backend/api/contest.py
from flask import Blueprint, request, jsonify
from core.contest_optimizer import optimize_contest
from utils.validators import validate_contest_input

bp = Blueprint('contest', __name__)

@bp.route('/optimize', methods=['POST'])
def optimize_contest_strategy():
    """Optimize contest strategy based on submissions"""
    try:
        data = request.get_json()
        
        # Validate input
        validation_result = validate_contest_input(data)
        if not validation_result['valid']:
            return jsonify({'error': validation_result['message']}), 400
        
        contest_id = data.get('contest_id')
        submissions = data.get('submissions', [])
        
        result = optimize_contest(contest_id, submissions)
        
        # Add strategy recommendations
        enhanced_result = {
            **result,
            'strategy_recommendations': _generate_strategy_recommendations(result),
            'time_allocation': _suggest_time_allocation(submissions),
            'risk_assessment': _assess_contest_risk(result)
        }
        
        return jsonify({
            'success': True,
            'data': enhanced_result
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/analyze-pattern', methods=['POST'])
def analyze_contest_pattern():
    """Analyze user's contest performance patterns"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        contest_history = data.get('contest_history', [])
        
        if not contest_history:
            return jsonify({'error': 'No contest history provided'}), 400
        
        pattern_analysis = {
            'performance_trend': _analyze_performance_trend(contest_history),
            'problem_type_performance': _analyze_problem_types(contest_history),
            'time_management': _analyze_time_management(contest_history),
            'consistency_score': _calculate_consistency_score(contest_history)
        }
        
        return jsonify({
            'success': True,
            'data': pattern_analysis
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def _generate_strategy_recommendations(contest_result):
    """Generate strategic recommendations based on contest analysis"""
    recommendations = []
    problem_stats = contest_result.get('problem_stats', {})
    
    for problem_id, stats in problem_stats.items():
        if not stats['solved']:
            recommendations.append({
                'type': 'unsolved_problem',
                'problem': problem_id,
                'suggestion': 'Focus on conceptual understanding before attempting again'
            })
        elif stats['total_time'] > 45:
            recommendations.append({
                'type': 'time_optimization',
                'problem': problem_id,
                'suggestion': 'Practice similar problems to improve speed'
            })
    
    return recommendations

def _suggest_time_allocation(submissions):
    """Suggest optimal time allocation strategy"""
    total_time = sum(s.get('time_taken', 0) for s in submissions)
    problem_count = len(submissions)
    
    if problem_count == 0:
        return {'strategy': 'insufficient_data'}
    
    avg_time = total_time / problem_count
    
    return {
        'strategy': 'balanced' if avg_time < 30 else 'speed_focus',
        'suggested_time_per_problem': min(avg_time * 0.8, 25),
        'buffer_time_percentage': 20
    }

def _assess_contest_risk(contest_result):
    """Assess risk factors in contest performance"""
    problem_stats = contest_result.get('problem_stats', {})
    
    unsolved_count = sum(1 for stats in problem_stats.values() if not stats['solved'])
    total_problems = len(problem_stats)
    
    if total_problems == 0:
        return {'risk_level': 'unknown'}
    
    unsolved_rate = unsolved_count / total_problems
    
    if unsolved_rate > 0.6:
        return {
            'risk_level': 'high',
            'factors': ['high_unsolved_rate', 'conceptual_gaps']
        }
    elif unsolved_rate > 0.3:
        return {
            'risk_level': 'medium',
            'factors': ['moderate_unsolved_rate']
        }
    else:
        return {
            'risk_level': 'low',
            'factors': ['good_problem_solving']
        }

def _analyze_performance_trend(contest_history):
    """Analyze performance trend over time"""
    if len(contest_history) < 2:
        return {'trend': 'insufficient_data'}
    
    scores = [contest.get('score', 0) for contest in contest_history]
    recent_avg = sum(scores[-3:]) / min(len(scores), 3)
    overall_avg = sum(scores) / len(scores)
    
    return {
        'trend': 'improving' if recent_avg > overall_avg else 'declining',
        'recent_average': recent_avg,
        'overall_average': overall_avg
    }

def _analyze_problem_types(contest_history):
    """Analyze performance by problem types"""
    type_performance = {}
    
    for contest in contest_history:
        problems = contest.get('problems', [])
        for problem in problems:
            prob_type = problem.get('type', 'unknown')
            solved = problem.get('solved', False)
            
            if prob_type not in type_performance:
                type_performance[prob_type] = {'total': 0, 'solved': 0}
            
            type_performance[prob_type]['total'] += 1
            if solved:
                type_performance[prob_type]['solved'] += 1
    
    return type_performance

def _analyze_time_management(contest_history):
    """Analyze time management patterns"""
    time_data = []
    
    for contest in contest_history:
        problems = contest.get('problems', [])
        for problem in problems:
            time_spent = problem.get('time_spent', 0)
            solved = problem.get('solved', False)
            time_data.append({'time': time_spent, 'solved': solved})
    
    if not time_data:
        return {'pattern': 'insufficient_data'}
    
    avg_time_solved = sum(d['time'] for d in time_data if d['solved']) / max(1, sum(1 for d in time_data if d['solved']))
    avg_time_unsolved = sum(d['time'] for d in time_data if not d['solved']) / max(1, sum(1 for d in time_data if not d['solved']))
    
    return {
        'average_time_solved': avg_time_solved,
        'average_time_unsolved': avg_time_unsolved,
        'efficiency_ratio': avg_time_solved / max(avg_time_unsolved, 1)
    }

def _calculate_consistency_score(contest_history):
    """Calculate consistency score based on performance variance"""
    scores = [contest.get('score', 0) for contest in contest_history]
    
    if len(scores) < 2:
        return {'score': 0, 'level': 'insufficient_data'}
    
    import statistics
    mean_score = statistics.mean(scores)
    std_dev = statistics.stdev(scores) if len(scores) > 1 else 0
    
    # Consistency score (lower standard deviation = higher consistency)
    consistency = max(0, 100 - (std_dev / max(mean_score, 1)) * 100)
    
    return {
        'score': consistency,
        'level': 'high' if consistency > 80 else 'medium' if consistency > 60 else 'low'
    }
