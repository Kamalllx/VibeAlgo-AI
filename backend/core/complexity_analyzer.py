# backend/core/complexity_analyzer.py (Enhanced)
import ast
import re
import json
from typing import Dict, List, Any
from utils.ast_parser import CodeParser
from data.knowledge_base import AlgorithmKnowledgeBase

class ComplexityAnalyzer:
    def __init__(self):
        self.loop_patterns = {
            'for': r'for\s+\w+\s+in\s+range\(',
            'while': r'while\s+.*:',
            'nested_for': r'for\s+.*:\s*\n\s*for\s+.*:'
        }
        
    def analyze_complexity(self, code: str) -> Dict[str, Any]:
        try:
            # Parse AST
            tree = ast.parse(code)
            
            # Analyze loops
            loop_count = self._count_loops(tree)
            nested_loops = self._detect_nested_loops(tree)
            
            # Determine complexity
            time_complexity = self._determine_time_complexity(loop_count, nested_loops)
            space_complexity = self._determine_space_complexity(tree)
            
            # Get algorithm suggestions
            suggestions = self._get_optimization_suggestions(code, time_complexity)
            
            return {
                'time_complexity': time_complexity,
                'space_complexity': space_complexity,
                'loop_analysis': {
                    'total_loops': loop_count,
                    'nested_depth': nested_loops
                },
                'suggestions': suggestions,
                'visualization_data': self._prepare_visualization_data(tree)
            }
            
        except Exception as e:
            return {'error': f'Analysis failed: {str(e)}'}
    
    def _count_loops(self, tree: ast.AST) -> int:
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                count += 1
        return count
    
    def _detect_nested_loops(self, tree: ast.AST) -> int:
        max_depth = 0
        
        def get_depth(node, current_depth=0):
            nonlocal max_depth
            if isinstance(node, (ast.For, ast.While)):
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            
            for child in ast.iter_child_nodes(node):
                get_depth(child, current_depth)
        
        get_depth(tree)
        return max_depth
    
    def _determine_time_complexity(self, loop_count: int, nested_depth: int) -> str:
        if nested_depth == 0:
            return "O(1)"
        elif nested_depth == 1:
            return "O(n)"
        elif nested_depth == 2:
            return "O(n²)"
        elif nested_depth == 3:
            return "O(n³)"
        else:
            return f"O(n^{nested_depth})"
    
    def _determine_space_complexity(self, tree: ast.AST) -> str:
        # Simple heuristic - count variable assignments
        assignments = sum(1 for node in ast.walk(tree) if isinstance(node, ast.Assign))
        if assignments == 0:
            return "O(1)"
        elif assignments < 5:
            return "O(1)"
        else:
            return "O(n)"
    
    def _get_optimization_suggestions(self, code: str, complexity: str) -> List[str]:
        suggestions = []
        
        if "O(n²)" in complexity:
            suggestions.append("Consider using more efficient algorithms like merge sort or quick sort")
            suggestions.append("Look for opportunities to reduce nested loops")
        
        if "for" in code and "range" in code:
            suggestions.append("Consider using list comprehensions for better performance")
        
        return suggestions
    
    def _prepare_visualization_data(self, tree: ast.AST) -> Dict[str, Any]:
        return {
            'nodes': [],
            'edges': [],
            'metrics': {
                'cyclomatic_complexity': self._calculate_cyclomatic_complexity(tree)
            }
        }
    
    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        complexity = 1  # Base complexity
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.And, ast.Or)):
                complexity += 1
        return complexity

# Global analyzer instance
analyzer = ComplexityAnalyzer()

def analyze_complexity(code: str) -> Dict[str, Any]:
    return analyzer.analyze_complexity(code)
