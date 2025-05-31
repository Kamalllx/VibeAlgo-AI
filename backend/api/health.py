# backend/api/health.py
from flask import Blueprint, jsonify
import psutil
import time
from datetime import datetime

bp = Blueprint('health', __name__)

@bp.route('/', methods=['GET'])
def health_check():
    """Basic health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'Algorithm Intelligence Backend'
    })

@bp.route('/detailed', methods=['GET'])
def detailed_health():
    """Detailed health check with system metrics"""
    try:
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Service status checks
        services_status = {
            'complexity_analyzer': _check_complexity_service(),
            'dsa_tracker': _check_dsa_service(),
            'contest_optimizer': _check_contest_service(),
            'ai_pipeline': _check_ai_services()
        }
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'system_metrics': {
                'cpu_usage_percent': cpu_percent,
                'memory_usage_percent': memory.percent,
                'disk_usage_percent': disk.percent,
                'available_memory_gb': memory.available / (1024**3)
            },
            'services': services_status,
            'uptime_seconds': _get_uptime()
        })
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

def _check_complexity_service():
    """Check if complexity analysis service is working"""
    try:
        from core.complexity_analyzer import analyze_complexity
        test_result = analyze_complexity("print('hello')")
        return {'status': 'healthy', 'last_check': datetime.now().isoformat()}
    except Exception as e:
        return {'status': 'unhealthy', 'error': str(e)}

def _check_dsa_service():
    """Check if DSA tracking service is working"""
    try:
        from core.dsa_progress_tracker import get_progress
        test_result = get_progress('health_check_user')
        return {'status': 'healthy', 'last_check': datetime.now().isoformat()}
    except Exception as e:
        return {'status': 'unhealthy', 'error': str(e)}

def _check_contest_service():
    """Check if contest optimization service is working"""
    try:
        from core.contest_optimizer import optimize_contest
        test_submissions = [{'problem_id': 'test', 'time_taken': 10, 'status': 'solved'}]
        test_result = optimize_contest('health_check', test_submissions)
        return {'status': 'healthy', 'last_check': datetime.now().isoformat()}
    except Exception as e:
        return {'status': 'unhealthy', 'error': str(e)}

def _check_ai_services():
    """Check if AI services are accessible"""
    try:
        # Basic AI service connectivity check
        from ai.groq_client import GroqClient
        # Don't make actual API call in health check, just check initialization
        client = GroqClient()
        return {'status': 'healthy', 'last_check': datetime.now().isoformat()}
    except Exception as e:
        return {'status': 'unhealthy', 'error': str(e)}

_start_time = time.time()

def _get_uptime():
    """Get service uptime in seconds"""
    return time.time() - _start_time