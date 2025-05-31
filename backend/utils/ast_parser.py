
# backend/utils/ast_parser.py
import ast
import re
from typing import Dict, List, Any, Optional

class CodeParser:
    def __init__(self):
        self.supported_languages = ['python', 'javascript', 'java', 'cpp']
    
    def parse_code(self, code: str, language: str = 'python') -> Dict[str, Any]:
        """Parse code and extract structural information"""
        try:
            if language == 'python':
                return self._parse_python(code)
            elif language == 'javascript':
                return self._parse_javascript(code)
            else:
                return self._parse_generic(code)
        except Exception as e:
            return {'error': f'Parse error: {str(e)}'}
    
    def _parse_python(self, code: str) -> Dict[str, Any]:
        """Parse Python code using AST"""
        try:
            tree = ast.parse(code)
            
            functions = []
            classes = []
            loops = []
            conditions = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append({
                        'name': node.name,
                        'line': node.lineno,
                        'args': [arg.arg for arg in node.args.args]
                    })
                elif isinstance(node, ast.ClassDef):
                    classes.append({
                        'name': node.name,
                        'line': node.lineno
                    })
                elif isinstance(node, (ast.For, ast.While)):
                    loops.append({
                        'type': type(node).__name__,
                        'line': node.lineno
                    })
                elif isinstance(node, ast.If):
                    conditions.append({
                        'line': node.lineno
                    })
            
            return {
                'functions': functions,
                'classes': classes,
                'loops': loops,
                'conditions': conditions,
                'total_lines': len(code.splitlines()),
                'language': 'python'
            }
            
        except SyntaxError as e:
            return {'error': f'Python syntax error: {str(e)}'}
    
    def _parse_javascript(self, code: str) -> Dict[str, Any]:
        """Parse JavaScript code using regex patterns"""
        # Simple regex-based parsing for JavaScript
        function_pattern = r'function\s+(\w+)\s*\('
        functions = re.findall(function_pattern, code)
        
        loop_patterns = [
            r'for\s*\(',
            r'while\s*\(',
            r'do\s*{.*?}\s*while'
        ]
        loops = sum(len(re.findall(pattern, code)) for pattern in loop_patterns)
        
        if_pattern = r'if\s*\('
        conditions = len(re.findall(if_pattern, code))
        
        return {
            'functions': [{'name': func, 'language': 'javascript'} for func in functions],
            'loops': [{'type': 'loop'} for _ in range(loops)],
            'conditions': [{'type': 'condition'} for _ in range(conditions)],
            'total_lines': len(code.splitlines()),
            'language': 'javascript'
        }
    
    def _parse_generic(self, code: str) -> Dict[str, Any]:
        """Generic parsing for unsupported languages"""
        lines = code.splitlines()
        
        # Count basic structures using simple patterns
        loop_keywords = ['for', 'while', 'do']
        condition_keywords = ['if', 'else', 'switch', 'case']
        
        loops = sum(1 for line in lines for keyword in loop_keywords if keyword in line.lower())
        conditions = sum(1 for line in lines for keyword in condition_keywords if keyword in line.lower())
        
        return {
            'loops': [{'type': 'generic_loop'} for _ in range(loops)],
            'conditions': [{'type': 'generic_condition'} for _ in range(conditions)],
            'total_lines': len(lines),
            'language': 'generic'
        }
    
    def extract_complexity_hints(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract complexity hints from parsed code"""
        loops = parsed_data.get('loops', [])
        loop_count = len(loops)
        
        # Simple complexity estimation
        if loop_count == 0:
            complexity_hint = "O(1)"
        elif loop_count == 1:
            complexity_hint = "O(n)"
        elif loop_count == 2:
            complexity_hint = "O(n²)"
        else:
            complexity_hint = f"O(n^{loop_count})"
        
        return {
            'estimated_complexity': complexity_hint,
            'loop_count': loop_count,
            'confidence': 'low' if parsed_data.get('language') == 'generic' else 'medium'
        }

# Global parser instance
parser = CodeParser()

def parse_code(code: str, language: str = 'python') -> Dict[str, Any]:
    return parser.parse_code(code, language)
