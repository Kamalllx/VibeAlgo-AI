
# backend/utils/validators.py
import re
from typing import Dict, Any

def validate_code_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate code analysis input"""
    if not data:
        return {'valid': False, 'message': 'No data provided'}
    
    code = data.get('code', '')
    if not code or not code.strip():
        return {'valid': False, 'message': 'Code cannot be empty'}
    
    if len(code) > 10000:
        return {'valid': False, 'message': 'Code too long (max 10000 characters)'}
    
    # Check for potentially dangerous patterns
    dangerous_patterns = ['import os', 'import subprocess', 'exec(', 'eval(']
    for pattern in dangerous_patterns:
        if pattern in code.lower():
            return {'valid': False, 'message': f'Potentially dangerous code pattern detected: {pattern}'}
    
    language = data.get('language', 'python')
    if language not in ['python', 'javascript', 'java', 'cpp', 'c']:
        return {'valid': False, 'message': 'Unsupported language'}
    
    return {'valid': True, 'message': 'Valid input'}

def validate_progress_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate DSA progress input"""
    if not data:
        return {'valid': False, 'message': 'No data provided'}
    
    problem_id = data.get('problem_id', '')
    if not problem_id or not problem_id.strip():
        return {'valid': False, 'message': 'Problem ID cannot be empty'}
    
    status = data.get('status', '')
    valid_statuses = ['attempted', 'solved', 'mastered']
    if status not in valid_statuses:
        return {'valid': False, 'message': f'Invalid status. Must be one of: {valid_statuses}'}
    
    difficulty = data.get('difficulty', 'medium')
    valid_difficulties = ['easy', 'medium', 'hard']
    if difficulty not in valid_difficulties:
        return {'valid': False, 'message': f'Invalid difficulty. Must be one of: {valid_difficulties}'}
    
    time_spent = data.get('time_spent', 0)
    if not isinstance(time_spent, int) or time_spent < 0:
        return {'valid': False, 'message': 'Time spent must be a non-negative integer'}
    
    return {'valid': True, 'message': 'Valid input'}

def validate_contest_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate contest optimization input"""
    if not data:
        return {'valid': False, 'message': 'No data provided'}
    
    contest_id = data.get('contest_id', '')
    if not contest_id or not contest_id.strip():
        return {'valid': False, 'message': 'Contest ID cannot be empty'}
    
    submissions = data.get('submissions', [])
    if not isinstance(submissions, list):
        return {'valid': False, 'message': 'Submissions must be a list'}
    
    if len(submissions) > 50:
        return {'valid': False, 'message': 'Too many submissions (max 50)'}
    
    for i, submission in enumerate(submissions):
        if not isinstance(submission, dict):
            return {'valid': False, 'message': f'Submission {i} must be an object'}
        
        if 'problem_id' not in submission:
            return {'valid': False, 'message': f'Submission {i} missing problem_id'}
    
    return {'valid': True, 'message': 'Valid input'}

def sanitize_user_input(text: str) -> str:
    """Sanitize user input to prevent injection attacks"""
    if not text:
        return ""
    
    # Remove potentially dangerous characters
    sanitized = re.sub(r'[<>"\']', '', text)
    
    # Limit length
    sanitized = sanitized[:1000]
    
    return sanitized.strip()
