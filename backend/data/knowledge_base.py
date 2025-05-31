
# backend/data/knowledge_base.py
import json
import os
from typing import Dict, List, Any

class AlgorithmKnowledgeBase:
    def __init__(self):
        self.algorithms_data = self._load_algorithms_data()
        self.patterns_data = self._load_patterns_data()
        
    def _load_algorithms_data(self) -> Dict[str, Any]:
        """Load algorithm metadata from config"""
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'algorithms.json')
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self._get_default_algorithms()
    
    def _load_patterns_data(self) -> Dict[str, Any]:
        """Load common algorithm patterns"""
        return {
            'sorting_patterns': {
                'bubble_sort': [
                    'for i in range(len(arr))',
                    'for j in range(0, len(arr)-i-1)',
                    'if arr[j] > arr[j+1]'
                ],
                'merge_sort': [
                    'def merge_sort(arr)',
                    'if len(arr) > 1',
                    'mid = len(arr)//2'
                ],
                'quick_sort': [
                    'def partition(arr, low, high)',
                    'pivot = arr[high]',
                    'while low < high'
                ]
            },
            'search_patterns': {
                'binary_search': [
                    'while left <= right',
                    'mid = (left + right) // 2',
                    'if arr[mid] == target'
                ],
                'linear_search': [
                    'for i in range(len(arr))',
                    'if arr[i] == target'
                ]
            },
            'dynamic_programming': {
                'memoization': [
                    'memo = {}',
                    'if n in memo',
                    'return memo[n]'
                ],
                'tabulation': [
                    'dp = [0] * (n+1)',
                    'for i in range(1, n+1)',
                    'dp[i] = dp[i-1] + dp[i-2]'
                ]
            }
        }
    
    def _get_default_algorithms(self) -> Dict[str, Any]:
        """Default algorithm data if config file not found"""
        return {
            "sorting": {
                "bubble_sort": {
                    "name": "Bubble Sort",
                    "time_complexity": "O(n²)",
                    "space_complexity": "O(1)",
                    "category": "sorting",
                    "difficulty": "easy"
                }
            }
        }
    
    def get_algorithm_info(self, algorithm_name: str) -> Dict[str, Any]:
        """Get information about a specific algorithm"""
        for category, algorithms in self.algorithms_data.items():
            if algorithm_name in algorithms:
                return algorithms[algorithm_name]
        return {}
    
    def identify_algorithm_pattern(self, code: str) -> List[str]:
        """Identify algorithm patterns in code"""
        identified_patterns = []
        code_lower = code.lower()
        
        for category, patterns in self.patterns_data.items():
            for algorithm, pattern_list in patterns.items():
                matches = sum(1 for pattern in pattern_list if pattern.lower() in code_lower)
                if matches >= len(pattern_list) * 0.6:  # 60% pattern match
                    identified_patterns.append(algorithm)
        
        return identified_patterns
    
    def get_optimization_suggestions(self, complexity: str, algorithm_patterns: List[str]) -> List[str]:
        """Get optimization suggestions based on complexity and patterns"""
        suggestions = []
        
        if "O(n²)" in complexity and "bubble_sort" in algorithm_patterns:
            suggestions.append("Consider using merge sort or quick sort for better O(n log n) complexity")
        
        if "O(n)" in complexity and "linear_search" in algorithm_patterns:
            suggestions.append("For sorted arrays, consider binary search for O(log n) complexity")
        
        if len(algorithm_patterns) == 0:
            suggestions.append("Consider using well-known algorithms for better performance")
        
        return suggestions

# Global knowledge base instance
knowledge_base = AlgorithmKnowledgeBase()
