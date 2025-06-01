# backend/visualization_database/visualization_manager.py
#!/usr/bin/env python3
"""
Visualization Manager
Handles algorithm detection, visualization selection and execution
"""

import json
import os
import sys
import subprocess
import re
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

class VisualizationManager:
    def __init__(self, db_file="visualization_database/algorithms_db.json"):
        self.db_file = db_file
        self.base_dir = Path("visualization_database/visualizations")
        self.db = self.load_database()
        
        print(f" Visualization Manager initialized")
        print(f" Database: {self.db_file}")
        print(f" Visualizations: {self.base_dir}")
    
    def load_database(self) -> Dict[str, Any]:
        """Load the algorithm visualization database"""
        try:
            with open(self.db_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Database file not found: {self.db_file}")
            return {}
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in database: {e}")
            return {}
    
    def detect_algorithm(self, user_input: str, generated_code: str = "") -> List[Tuple[str, str, float]]:
        """
        Detect which algorithm(s) the user is asking about
        Returns list of (category, algorithm, confidence) tuples
        """
        matches = []
        input_lower = user_input.lower()
        code_lower = generated_code.lower() if generated_code else ""
        
        # Search through all algorithms
        for category, algorithms in self.db.items():
            for algo_key, algo_data in algorithms.items():
                confidence = self._calculate_confidence(
                    input_lower, code_lower, algo_key, algo_data
                )
                
                if confidence > 0.3:  # Threshold for relevance
                    matches.append((category, algo_key, confidence))
        
        # Sort by confidence (highest first)
        matches.sort(key=lambda x: x[2], reverse=True)
        return matches[:5]  # Return top 5 matches
    
    def _calculate_confidence(self, input_text: str, code_text: str, algo_key: str, algo_data: Dict) -> float:
        """Calculate confidence score for algorithm match"""
        score = 0.0
        
        # Check algorithm name
        algo_name = algo_data["name"].lower()
        if algo_name in input_text:
            score += 0.8
        
        # Check algorithm key
        if algo_key.replace("_", " ") in input_text:
            score += 0.7
        
        # Check description keywords
        description = algo_data["description"].lower()
        desc_words = description.split()
        for word in desc_words:
            if len(word) > 3 and word in input_text:
                score += 0.1
        
        # Check code patterns
        if code_text:
            code_patterns = {
                "dijkstra": ["dijkstra", "priority_queue", "shortest", "distance"],
                "binary_search": ["left", "right", "mid", "binary"],
                "bubble_sort": ["bubble", "swap", "adjacent"],
                "quick_sort": ["pivot", "partition", "quick"],
                "merge_sort": ["merge", "divide", "conquer"],
                "bfs": ["queue", "bfs", "breadth"],
                "dfs": ["stack", "dfs", "depth", "recursive"],
                "fibonacci": ["fibonacci", "fib", "memo"],
                "knapsack": ["knapsack", "weight", "value", "capacity"]
            }
            
            if algo_key in code_patterns:
                for pattern in code_patterns[algo_key]:
                    if pattern in code_text:
                        score += 0.2
        
        return min(score, 1.0)  # Cap at 1.0
    
    def get_algorithm_info(self, category: str, algorithm: str) -> Optional[Dict]:
        """Get detailed information about an algorithm"""
        try:
            return self.db[category][algorithm]
        except KeyError:
            return None
    
    def list_visualizations(self, category: str, algorithm: str) -> List[str]:
        """List available visualization files for an algorithm"""
        algo_info = self.get_algorithm_info(category, algorithm)
        if not algo_info:
            return []
        return algo_info.get("visualization_files", [])
    
    def execute_visualization(self, category: str, algorithm: str, viz_type: str = "animation") -> bool:
        """Execute a specific visualization"""
        algo_info = self.get_algorithm_info(category, algorithm)
        if not algo_info:
            print(f"❌ Algorithm not found: {category}/{algorithm}")
            return False
        
        # Find appropriate visualization file
        viz_files = algo_info["visualization_files"]
        selected_file = None
        
        # Try to find file matching viz_type
        for file_path in viz_files:
            if viz_type in file_path:
                selected_file = file_path
                break
        
        # Fallback to first available file
        if not selected_file and viz_files:
            selected_file = viz_files[0]
        
        if not selected_file:
            print(f"❌ No visualization files found for {algorithm}")
            return False
        
        # Execute the visualization
        full_path = self.base_dir / selected_file
        
        if not full_path.exists():
            print(f" Visualization file not found: {full_path}")
            return False
        
        try:
            print(f"Executing visualization: {selected_file}")
            result = subprocess.run([sys.executable, str(full_path)], 
                                  capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print(f" Visualization completed successfully")
                if result.stdout:
                    print(f"📄 Output: {result.stdout}")
                return True
            else:
                print(f" Visualization failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f" Visualization timed out")
            return False
        except Exception as e:
            print(f" Error executing visualization: {e}")
            return False
    
    def get_best_visualization(self, user_input: str, generated_code: str = "") -> Optional[Tuple[str, str, str]]:
        """Get the best matching visualization for user input"""
        matches = self.detect_algorithm(user_input, generated_code)
        
        if not matches:
            print(" No matching algorithms found")
            return None
        
        # Return best match
        category, algorithm, confidence = matches[0]
        print(f"🎯 Best match: {category}/{algorithm} (confidence: {confidence:.2f})")
        
        # Select appropriate visualization type
        algo_info = self.get_algorithm_info(category, algorithm)
        viz_types = algo_info.get("visualization_types", ["animation"])
        
        # Prefer animation for most cases
        if "animation" in viz_types:
            viz_type = "animation"
        else:
            viz_type = viz_types[0]
        
        return category, algorithm, viz_type
    
    def auto_visualize(self, user_input: str, generated_code: str = "") -> bool:
        """Automatically detect and run best visualization"""
        result = self.get_best_visualization(user_input, generated_code)
         
        if not result:
            return False
        
        category, algorithm, viz_type = result
        return self.execute_visualization(category, algorithm, viz_type)
    
    def list_all_algorithms(self) -> Dict[str, List[str]]:
        """List all available algorithms by category"""
        return {
            category: list(algorithms.keys()) 
            for category, algorithms in self.db.items()
        }
    
    def search_algorithms(self, query: str) -> List[Tuple[str, str, str]]:
        """Search for algorithms matching query"""
        results = []
        query_lower = query.lower()
        
        for category, algorithms in self.db.items():
            for algo_key, algo_data in algorithms.items():
                # Search in name and description
                if (query_lower in algo_data["name"].lower() or 
                    query_lower in algo_data["description"].lower() or
                    query_lower in algo_key):
                    results.append((category, algo_key, algo_data["name"]))
        
        return results

# Global instance
visualization_manager = VisualizationManager()
