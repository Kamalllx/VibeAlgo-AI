
# backend/utils/security.py
import hashlib
import secrets
import re
from typing import Dict, Any, Optional

class SecurityUtils:
    @staticmethod
    def hash_code(code: str) -> str:
        """Generate hash for code content"""
        return hashlib.sha256(code.encode()).hexdigest()
    
    @staticmethod
    def generate_session_token() -> str:
        """Generate secure session token"""
        return secrets.token_urlsafe(32)
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename to prevent path traversal"""
        # Remove path traversal attempts
        filename = re.sub(r'\.\./', '', filename)
        filename = re.sub(r'\.\.\\', '', filename)
        
        # Keep only alphanumeric, dots, hyphens, and underscores
        filename = re.sub(r'[^a-zA-Z0-9._-]', '', filename)
        
        return filename[:100]  # Limit length
    
    @staticmethod
    def validate_user_id(user_id: str) -> bool:
        """Validate user ID format"""
        if not user_id or len(user_id) > 50:
            return False
        
        # Allow alphanumeric and common special characters
        pattern = r'^[a-zA-Z0-9._-]+$'
        return bool(re.match(pattern, user_id))
    
    @staticmethod
    def rate_limit_key(user_id: str, endpoint: str) -> str:
        """Generate rate limiting key"""
        return f"rate_limit:{user_id}:{endpoint}"
    
    @staticmethod
    def check_code_safety(code: str) -> Dict[str, Any]:
        """Check code for potentially dangerous patterns"""
        dangerous_patterns = [
            r'import\s+os',
            r'import\s+subprocess',
            r'import\s+sys',
            r'exec\s*\(',
            r'eval\s*\(',
            r'__import__',
            r'open\s*\(',
            r'file\s*\(',
            r'input\s*\(',
            r'raw_input\s*\('
        ]
        
        detected_patterns = []
        for pattern in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                detected_patterns.append(pattern)
        
        return {
            'is_safe': len(detected_patterns) == 0,
            'detected_patterns': detected_patterns,
            'risk_level': 'high' if detected_patterns else 'low'
        }

# Global security instance
security = SecurityUtils()
