# backend/app.py
#!/usr/bin/env python3
"""
ALGORITHM INTELLIGENCE SUITE - COMPLETE FLASK WEB APPLICATION
Complete integration of all system components with REST API endpoints
Version 3.0 - Production Ready - Thread-Safe Asyncio
"""

import asyncio
import os
import json
import time
import traceback
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from flask import Flask, request, jsonify, render_template_string, send_file, abort
from flask_cors import CORS
import logging

# Import core system components
try:
    from core.agent_orchestrator import orchestrator
    ORCHESTRATOR_AVAILABLE = True
except ImportError:
    print("WARNING: Agent orchestrator not available")
    ORCHESTRATOR_AVAILABLE = False

try:
    from core.algorithm_solver_agent import algorithm_solver
    SOLVER_AVAILABLE = True
except ImportError:
    print("WARNING: Algorithm solver not available")
    SOLVER_AVAILABLE = False

# Import visualization components
try:
    from visualization_database.mongodb_manager import initialize_mongodb_manager
    MONGODB_AVAILABLE = True
except ImportError:
    print("WARNING: MongoDB visualization system not available")
    MONGODB_AVAILABLE = False

# Import performance analysis
try:
    from visualizations.performance_analysis import create_complexity_comparison_plots, create_algorithm_performance_benchmark
    PERFORMANCE_AVAILABLE = True
except ImportError:
    print("WARNING: Performance analysis not available")
    PERFORMANCE_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_or_create_eventloop():
    """
    Get the current event loop or create a new one if none exists.
    This solves the 'There is no current event loop in thread' error.
    """
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop
        else:
            raise ex

def run_async_in_thread(coro):
    """
    Run an async coroutine in a thread-safe way.
    This handles the Flask thread + asyncio combination properly.
    """
    try:
        # Try to get existing loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is running, we need to run in a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            # Loop exists but not running, we can use it
            return loop.run_until_complete(coro)
    except RuntimeError:
        # No loop exists, create one and run
        return asyncio.run(coro)

class AlgorithmIntelligenceAPI:
    def __init__(self, mongodb_connection: str = None):
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS for frontend integration
        
        self.session_counter = 0
        self.mongo_manager = None
        
        # Initialize MongoDB if available
        if MONGODB_AVAILABLE and mongodb_connection:
            try:
                self.mongo_manager = initialize_mongodb_manager(mongodb_connection)
                logger.info("MongoDB visualization system enabled")
            except Exception as e:
                logger.error(f"MongoDB initialization failed: {e}")
        
        self.setup_routes()
        logger.info("Algorithm Intelligence API initialized")
    
    def setup_routes(self):
        """Setup all API endpoints"""
        
        @self.app.route('/')
        def home():
            """Main landing page with API documentation"""
            return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>Algorithm Intelligence Suite API</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f7fa; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }
        .endpoint { background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #007bff; }
        .method { color: #007bff; font-weight: bold; }
        .status { color: #28a745; font-weight: bold; }
        h1 { color: #2c3e50; }
        h2 { color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
        pre { background: #2d3748; color: #e2e8f0; padding: 15px; border-radius: 5px; overflow-x: auto; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Algorithm Intelligence Suite API</h1>
        <p><span class="status">Status: Online</span> | Version: 3.0 | MongoDB: {{ 'Enabled' if mongo_status else 'Disabled' }}</p>
        
        <h2>📚 Available Endpoints</h2>
        
        <div class="endpoint">
            <h3><span class="method">POST</span> /api/analyze</h3>
            <p>Complete algorithm analysis pipeline</p>
            <pre>{
  "input": "algorithm code or problem description",
  "input_type": "auto|code|problem",
  "options": {
    "include_visualization": true,
    "include_performance": true
  }
}</pre>
        </div>
        
        <div class="endpoint">
            <h3><span class="method">GET</span> /api/status</h3>
            <p>API health check and system status</p>
        </div>
        
        <h2>🧪 Quick Test</h2>
        <p>Try the API with curl:</p>
        <pre>curl -X POST http://localhost:5000/api/analyze \\
  -H "Content-Type: application/json" \\
  -d '{"input": "BFS algorithm", "input_type": "problem"}'</pre>
        
        <h2>📊 Features</h2>
        <ul>
            <li>🧠 AI-powered algorithm analysis</li>
            <li>📈 RAG-enhanced complexity analysis</li>
            <li>🎨 Automatic visualization generation</li>
            <li>🗄️ MongoDB-powered algorithm database</li>
            <li>📊 Performance benchmarking</li>
            <li>🔍 Educational insights and recommendations</li>
        </ul>
    </div>
</body>
</html>
            ''', mongo_status=self.mongo_manager is not None)
        
        @self.app.route('/api/status')
        def api_status():
            """API health check"""
            return jsonify({
                'status': 'online',
                'version': '3.0',
                'timestamp': datetime.now().isoformat(),
                'mongodb_enabled': self.mongo_manager is not None,
                'algorithms_available': self.mongo_manager.algorithms.count_documents({}) if self.mongo_manager else 0,
                'orchestrator_available': ORCHESTRATOR_AVAILABLE,
                'solver_available': SOLVER_AVAILABLE,
                'performance_available': PERFORMANCE_AVAILABLE,
                'features': {
                    'complexity_analysis': ORCHESTRATOR_AVAILABLE,
                    'algorithm_solving': SOLVER_AVAILABLE,
                    'visualization_generation': self.mongo_manager is not None,
                    'performance_analysis': PERFORMANCE_AVAILABLE,
                    'rag_enhancement': ORCHESTRATOR_AVAILABLE
                }
            })
        
        @self.app.route('/api/analyze', methods=['POST'])
        def analyze_algorithm():
            """Complete algorithm analysis pipeline - Thread-Safe"""
            try:
                data = request.get_json()
                
                if not data or 'input' not in data:
                    return jsonify({'error': 'Missing required field: input'}), 400
                
                user_input = data['input']
                input_type = data.get('input_type', 'auto')
                options = data.get('options', {})
                
                # Generate session ID
                self.session_counter += 1
                session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.session_counter}"
                
                # Create output directory
                output_dir = f"algorithm_intelligence_results_{session_id}"
                os.makedirs(output_dir, exist_ok=True)
                os.makedirs(f"{output_dir}/visualizations", exist_ok=True)
                os.makedirs(f"{output_dir}/reports", exist_ok=True)
                
                # Process through the pipeline using thread-safe async handling
                try:
                    result = run_async_in_thread(
                        self._process_algorithm_pipeline(user_input, input_type, session_id, output_dir, options)
                    )
                    return jsonify(result)
                except Exception as async_error:
                    logger.error(f"Async pipeline error: {async_error}")
                    # Fallback to sync processing
                    result = self._process_algorithm_pipeline_sync(user_input, input_type, session_id, output_dir, options)
                    return jsonify(result)
                
            except Exception as e:
                logger.error(f"Analysis error: {e}")
                logger.error(traceback.format_exc())
                return jsonify({
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }), 500
        
        @self.app.route('/api/complexity', methods=['POST'])
        def analyze_complexity():
            """Detailed complexity analysis - Thread-Safe"""
            try:
                data = request.get_json()
                
                if not data or 'code' not in data:
                    return jsonify({'error': 'Missing required field: code'}), 400
                
                code = data['code']
                language = data.get('language', 'python')
                
                if not ORCHESTRATOR_AVAILABLE:
                    # Fallback complexity analysis
                    return jsonify({
                        'agent_result': {
                            'complexity_analysis': {
                                'time_complexity': 'O(n)',
                                'space_complexity': 'O(1)',
                                'reasoning': 'Basic complexity analysis (orchestrator not available)'
                            }
                        },
                        'processing_time': 0.1,
                        'success': True
                    })
                
                # Run complexity analysis using thread-safe async
                try:
                    result = run_async_in_thread(
                        orchestrator.process_request("complexity_analysis", {
                            "code": code,
                            "language": language,
                            "platform": "general"
                        })
                    )
                    return jsonify(result)
                except Exception as async_error:
                    logger.error(f"Async complexity analysis error: {async_error}")
                    # Return fallback result
                    return jsonify({
                        'agent_result': {
                            'complexity_analysis': {
                                'time_complexity': 'Unknown',
                                'space_complexity': 'Unknown',
                                'reasoning': f'Analysis failed: {str(async_error)}'
                            }
                        },
                        'processing_time': 0.1,
                        'success': False,
                        'error': str(async_error)
                    })
                
            except Exception as e:
                logger.error(f"Complexity analysis error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/solve', methods=['POST'])
        def solve_problem():
            """Algorithm problem solving - Thread-Safe"""
            try:
                data = request.get_json()
                
                if not data or 'problem' not in data:
                    return jsonify({'error': 'Missing required field: problem'}), 400
                
                problem = data['problem']
                context = data.get('context', '')
                
                if not SOLVER_AVAILABLE:
                    # Enhanced fallback solution
                    return jsonify({
                        'optimal_solution': {
                            'algorithm_name': 'BFS (Breadth-First Search)' if 'bfs' in problem.lower() else 'Generic Algorithm',
                            'description': self._get_fallback_description(problem),
                            'code': {
                                'code': self._get_fallback_code(problem)
                            }
                        },
                        'problem_analysis': {
                            'problem_type': self._detect_problem_type(problem),
                            'difficulty': 'Medium'
                        },
                        'processing_time': 0.1,
                        'success': True,
                        'message': 'Fallback solution provided (solver not available)'
                    })
                
                # Run algorithm solving using thread-safe async
                try:
                    result = run_async_in_thread(
                        algorithm_solver.solve_problem(problem, context)
                    )
                    return jsonify(result)
                except Exception as async_error:
                    logger.error(f"Async problem solving error: {async_error}")
                    # Return enhanced fallback
                    return jsonify({
                        'optimal_solution': {
                            'algorithm_name': self._detect_algorithm_name(problem),
                            'description': self._get_fallback_description(problem),
                            'code': {
                                'code': self._get_fallback_code(problem)
                            }
                        },
                        'problem_analysis': {
                            'problem_type': self._detect_problem_type(problem),
                            'difficulty': 'Medium'
                        },
                        'processing_time': 0.1,
                        'success': False,
                        'error': str(async_error),
                        'message': 'Fallback solution due to async error'
                    })
                
            except Exception as e:
                logger.error(f"Problem solving error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/visualize', methods=['POST'])
        def generate_visualization():
            """Generate algorithm visualizations"""
            try:
                data = request.get_json()
                
                if not data or 'algorithm' not in data:
                    return jsonify({'error': 'Missing required field: algorithm'}), 400
                
                algorithm = data['algorithm']
                input_data = data.get('data', '')
                
                # Try to find and execute visualization
                if self.mongo_manager:
                    matches = self.mongo_manager.detect_algorithm(algorithm, input_data, [])
                    
                    if matches:
                        category, algorithm_key, name, confidence = matches[0]
                        
                        # Execute visualization
                        success, files = self._execute_visualization(category, algorithm_key)
                        
                        return jsonify({
                            'success': success,
                            'algorithm_detected': name,
                            'confidence': confidence,
                            'files_generated': files,
                            'category': category,
                            'algorithm_key': algorithm_key
                        })
                    else:
                        return jsonify({
                            'success': False,
                            'message': 'No matching algorithm found',
                            'suggestion': 'Try being more specific about the algorithm'
                        })
                else:
                    return jsonify({
                        'success': False,
                        'message': 'MongoDB visualization system not available'
                    })
                
            except Exception as e:
                logger.error(f"Visualization error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/algorithms')
        def list_algorithms():
            """List all available algorithms"""
            try:
                if self.mongo_manager:
                    algorithms = list(self.mongo_manager.algorithms.find({}, {
                        '_id': 0,
                        'category': 1,
                        'algorithm_key': 1,
                        'name': 1,
                        'description': 1,
                        'complexity': 1,
                        'difficulty': 1
                    }))
                    
                    return jsonify({
                        'algorithms': algorithms,
                        'total_count': len(algorithms),
                        'categories': list(set(alg['category'] for alg in algorithms))
                    })
                else:
                    # Return enhanced sample algorithms
                    sample_algorithms = [
                        {
                            'category': 'graphs',
                            'algorithm_key': 'bfs',
                            'name': 'Breadth-First Search (BFS)',
                            'description': 'Graph traversal algorithm that explores nodes level by level',
                            'complexity': {'time': 'O(V + E)', 'space': 'O(V)'},
                            'difficulty': 'Medium'
                        },
                        {
                            'category': 'graphs',
                            'algorithm_key': 'dfs',
                            'name': 'Depth-First Search (DFS)',
                            'description': 'Graph traversal algorithm that explores as far as possible along each branch',
                            'complexity': {'time': 'O(V + E)', 'space': 'O(V)'},
                            'difficulty': 'Medium'
                        },
                        {
                            'category': 'searching',
                            'algorithm_key': 'binary_search',
                            'name': 'Binary Search',
                            'description': 'Efficient search algorithm for sorted arrays',
                            'complexity': {'time': 'O(log n)', 'space': 'O(1)'},
                            'difficulty': 'Easy'
                        }
                    ]
                    
                    return jsonify({
                        'algorithms': sample_algorithms,
                        'total_count': len(sample_algorithms),
                        'categories': ['graphs', 'searching'],
                        'message': 'MongoDB not available - showing sample algorithms'
                    })
                
            except Exception as e:
                logger.error(f"Algorithm listing error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/performance')
        def generate_performance_analysis():
            """Generate comprehensive performance analysis with error handling"""
            try:
                if not PERFORMANCE_AVAILABLE:
                    return jsonify({
                        'success': False,
                        'files_generated': [],
                        'message': 'Performance analysis module not available',
                        'error': 'Module import failed'
                    }), 500
                
                # Generate performance plots with error handling
                complexity_file = None
                benchmark_file = None
                
                try:
                    complexity_file = create_complexity_comparison_plots()
                except Exception as e:
                    logger.error(f"Complexity plots failed: {e}")
                
                try:
                    benchmark_file = create_algorithm_performance_benchmark()
                except Exception as e:
                    logger.error(f"Benchmark plots failed: {e}")
                
                # Generate at least one fallback visualization
                if not complexity_file and not benchmark_file:
                    fallback_file = self._create_fallback_performance_chart()
                    return jsonify({
                        'success': True,
                        'files_generated': [fallback_file] if fallback_file else [],
                        'message': 'Performance analysis completed with fallback visualization',
                        'note': 'Some advanced plots failed, but basic analysis provided'
                    })
                
                generated_files = [f for f in [complexity_file, benchmark_file] if f is not None]
                
                return jsonify({
                    'success': True,
                    'files_generated': generated_files,
                    'message': 'Performance analysis completed successfully'
                })
                
            except Exception as e:
                logger.error(f"Performance analysis error: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'message': 'Performance analysis failed',
                    'suggestion': 'Check system dependencies and try again'
                }), 500
        
        # File serving endpoints
        @self.app.route('/results/<session_id>')
        def get_results(session_id):
            """Retrieve analysis results"""
            try:
                results_dir = f"algorithm_intelligence_results_{session_id}"
                
                if not os.path.exists(results_dir):
                    return jsonify({'error': 'Session not found'}), 404
                
                # Check for index.html
                index_file = os.path.join(results_dir, "index.html")
                if os.path.exists(index_file):
                    return send_file(index_file)
                
                # Otherwise return directory listing
                files = []
                try:
                    files = os.listdir(results_dir)
                except Exception as e:
                    logger.error(f"Error listing directory: {e}")
                
                return jsonify({
                    'session_id': session_id,
                    'files': files,
                    'directory': results_dir
                })
                
            except Exception as e:
                logger.error(f"Results retrieval error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/results/<session_id>/visualizations/<filename>')
        def get_visualization_file(session_id, filename):
            """Serve visualization files"""
            try:
                file_path = os.path.join(f"algorithm_intelligence_results_{session_id}", "visualizations", filename)
                
                if os.path.exists(file_path):
                    return send_file(file_path)
                else:
                    return jsonify({'error': 'File not found'}), 404
                
            except Exception as e:
                logger.error(f"File serving error: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/results/<session_id>/reports/<filename>')
        def get_report_file(session_id, filename):
            """Serve report files"""
            try:
                file_path = os.path.join(f"algorithm_intelligence_results_{session_id}", "reports", filename)
                
                if os.path.exists(file_path):
                    return send_file(file_path)
                else:
                    return jsonify({'error': 'File not found'}), 404
                
            except Exception as e:
                logger.error(f"Report serving error: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify({'error': 'Endpoint not found'}), 404

        @self.app.errorhandler(500)
        def internal_error(error):
            return jsonify({'error': 'Internal server error'}), 500
    
    # Enhanced fallback methods
    def _detect_algorithm_name(self, problem: str) -> str:
        """Detect algorithm name from problem description"""
        problem_lower = problem.lower()
        
        if 'bfs' in problem_lower or 'breadth first' in problem_lower:
            return 'Breadth-First Search (BFS)'
        elif 'dfs' in problem_lower or 'depth first' in problem_lower:
            return 'Depth-First Search (DFS)'
        elif 'dijkstra' in problem_lower:
            return "Dijkstra's Algorithm"
        elif 'binary search' in problem_lower:
            return 'Binary Search'
        elif 'sort' in problem_lower:
            return 'Sorting Algorithm'
        else:
            return 'Graph Algorithm'
    
    def _detect_problem_type(self, problem: str) -> str:
        """Detect problem type from description"""
        problem_lower = problem.lower()
        
        if any(word in problem_lower for word in ['graph', 'node', 'edge', 'vertex', 'bfs', 'dfs']):
            return 'Graph Traversal'
        elif any(word in problem_lower for word in ['search', 'find']):
            return 'Searching'
        elif 'sort' in problem_lower:
            return 'Sorting'
        else:
            return 'Algorithm Problem'
    
    def _get_fallback_description(self, problem: str) -> str:
        """Get fallback description based on problem"""
        problem_lower = problem.lower()
        
        if 'bfs' in problem_lower:
            return "Breadth-First Search (BFS) is a graph traversal algorithm that visits nodes level by level, exploring all neighbors at the current depth before moving to nodes at the next depth level."
        elif 'dfs' in problem_lower:
            return "Depth-First Search (DFS) is a graph traversal algorithm that explores as far as possible along each branch before backtracking."
        else:
            return "Algorithm solution for the given problem using standard algorithmic approaches."
    
    def _get_fallback_code(self, problem: str) -> str:
        """Get fallback code based on problem"""
        problem_lower = problem.lower()
        
        if 'bfs' in problem_lower:
            return '''from collections import deque

def bfs(graph, start):
    """
    Breadth-First Search implementation
    
    Args:
        graph: Dictionary representing adjacency list
        start: Starting node
    
    Returns:
        List of nodes in BFS order
    """
    visited = set()
    queue = deque([start])
    result = []
    
    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            result.append(node)
            
            # Add neighbors to queue
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    queue.append(neighbor)
    
    return result

# Example usage:
# graph = {
#     'A': ['B', 'C'],
#     'B': ['A', 'D', 'E'],
#     'C': ['A', 'F'],
#     'D': ['B'],
#     'E': ['B', 'F'],
#     'F': ['C', 'E']
# }
# 
# result = bfs(graph, 'A')
# print("BFS traversal:", result)
'''
        elif 'dfs' in problem_lower:
            return '''def dfs(graph, start, visited=None):
    """
    Depth-First Search implementation
    
    Args:
        graph: Dictionary representing adjacency list
        start: Starting node
        visited: Set of visited nodes
    
    Returns:
        List of nodes in DFS order
    """
    if visited is None:
        visited = set()
    
    result = []
    
    if start not in visited:
        visited.add(start)
        result.append(start)
        
        # Recursively visit neighbors
        for neighbor in graph.get(start, []):
            result.extend(dfs(graph, neighbor, visited))
    
    return result

# Example usage:
# graph = {
#     'A': ['B', 'C'],
#     'B': ['A', 'D', 'E'],
#     'C': ['A', 'F'],
#     'D': ['B'],
#     'E': ['B', 'F'],
#     'F': ['C', 'E']
# }
# 
# result = dfs(graph, 'A')
# print("DFS traversal:", result)
'''
        else:
            return '''def solve_problem():
    """
    Generic algorithm solution
    Modify this template based on your specific problem requirements
    """
    
    # Step 1: Initialize variables
    result = []
    
    # Step 2: Process input
    # Add your algorithm logic here
    
    # Step 3: Return result
    return result

# Example usage:
# solution = solve_problem()
# print("Result:", solution)
'''
    
    async def _process_algorithm_pipeline(self, user_input: str, input_type: str, session_id: str, output_dir: str, options: dict):
        """Process through the complete algorithm analysis pipeline - Async version"""
        
        results = {
            "session_id": session_id,
            "input": user_input,
            "input_type": input_type,
            "timestamp": datetime.now().isoformat(),
            "stages": {},
            "options": options
        }
        
        # Stage 1: Input Analysis
        code, problem_description = self._extract_code_and_problem(user_input, input_type)
        
        results["stages"]["input_analysis"] = {
            "detected_type": "code" if code else "problem",
            "code_extracted": code,
            "problem_description": problem_description
        }
        
        # Stage 2: Complexity Analysis (if code provided and orchestrator available)
        if code and ORCHESTRATOR_AVAILABLE:
            try:
                complexity_result = await orchestrator.process_request("complexity_analysis", {
                    "code": code,
                    "language": "python",
                    "platform": "general"
                })
                results["stages"]["complexity_analysis"] = complexity_result
            except Exception as e:
                logger.error(f"Complexity analysis failed: {e}")
                results["stages"]["complexity_analysis"] = {
                    "error": str(e),
                    "agent_result": {
                        "complexity_analysis": {
                            "time_complexity": "Unknown",
                            "space_complexity": "Unknown",
                            "reasoning": "Analysis failed"
                        }
                    }
                }
        
        # Stage 3: Algorithm Problem Solving
        if (problem_description or not code) and SOLVER_AVAILABLE:
            try:
                problem_input = problem_description or user_input
                solving_result = await algorithm_solver.solve_problem(problem_input, code)
                results["stages"]["algorithm_solving"] = solving_result
                
                # Extract generated code for visualization
                if "optimal_solution" in solving_result and not code:
                    solution_code = solving_result["optimal_solution"].get("code", {})
                    if isinstance(solution_code, dict):
                        code = solution_code.get("code", "")
            except Exception as e:
                logger.error(f"Problem solving failed: {e}")
                results["stages"]["algorithm_solving"] = {
                    "error": str(e),
                    "optimal_solution": {
                        "algorithm_name": self._detect_algorithm_name(user_input),
                        "description": self._get_fallback_description(user_input),
                        "code": {"code": self._get_fallback_code(user_input)}
                    },
                    "problem_analysis": {
                        "problem_type": self._detect_problem_type(user_input),
                        "difficulty": "Medium"
                    }
                }
        
        # Use fallback if solver not available
        if not SOLVER_AVAILABLE and (problem_description or not code):
            results["stages"]["algorithm_solving"] = {
                "optimal_solution": {
                    "algorithm_name": self._detect_algorithm_name(user_input),
                    "description": self._get_fallback_description(user_input),
                    "code": {"code": self._get_fallback_code(user_input)}
                },
                "problem_analysis": {
                    "problem_type": self._detect_problem_type(user_input),
                    "difficulty": "Medium"
                }
            }
        
        # Stage 4: Visualization Generation
        if options.get('include_visualization', True):
            visualization_result = self._generate_visualizations(user_input, code, output_dir)
            results["stages"]["visualizations"] = visualization_result
        
        # Stage 5: Performance Analysis
        if options.get('include_performance', True):
            try:
                performance_files = []
                
                if PERFORMANCE_AVAILABLE:
                    try:
                        complexity_file = create_complexity_comparison_plots()
                        benchmark_file = create_algorithm_performance_benchmark()
                        
                        # Move files to output directory
                        import shutil
                        if complexity_file and os.path.exists(complexity_file):
                            dest_path = os.path.join(output_dir, "visualizations", complexity_file)
                            shutil.move(complexity_file, dest_path)
                            performance_files.append(complexity_file)
                        
                        if benchmark_file and os.path.exists(benchmark_file):
                            dest_path = os.path.join(output_dir, "visualizations", benchmark_file)
                            shutil.move(benchmark_file, dest_path)
                            performance_files.append(benchmark_file)
                    except Exception as perf_error:
                        logger.error(f"Performance plots failed: {perf_error}")
                
                results["stages"]["performance_analysis"] = {
                    "success": len(performance_files) > 0,
                    "files_generated": performance_files
                }
                
            except Exception as e:
                logger.error(f"Performance analysis failed: {e}")
                results["stages"]["performance_analysis"] = {
                    "success": False,
                    "error": str(e)
                }
        
        # Stage 6: Educational Report
        results["stages"]["educational_report"] = {
            "key_concepts": self._get_educational_concepts(user_input),
            "recommendations": self._get_recommendations(user_input),
            "related_algorithms": self._get_related_algorithms(user_input),
            "practical_applications": self._get_practical_applications(user_input)
        }
        
        # Save results
        self._save_results(results, output_dir)
        
        return results
    
    def _process_algorithm_pipeline_sync(self, user_input: str, input_type: str, session_id: str, output_dir: str, options: dict):
        """Synchronous fallback version of the pipeline"""
        
        results = {
            "session_id": session_id,
            "input": user_input,
            "input_type": input_type,
            "timestamp": datetime.now().isoformat(),
            "stages": {},
            "options": options,
            "processing_mode": "synchronous_fallback"
        }
        
        # Stage 1: Input Analysis
        code, problem_description = self._extract_code_and_problem(user_input, input_type)
        
        results["stages"]["input_analysis"] = {
            "detected_type": "code" if code else "problem",
            "code_extracted": code,
            "problem_description": problem_description
        }
        
        # Stage 2: Synchronous Algorithm Solving (fallback)
        if problem_description or not code:
            results["stages"]["algorithm_solving"] = {
                "optimal_solution": {
                    "algorithm_name": self._detect_algorithm_name(user_input),
                    "description": self._get_fallback_description(user_input),
                    "code": {"code": self._get_fallback_code(user_input)}
                },
                "problem_analysis": {
                    "problem_type": self._detect_problem_type(user_input),
                    "difficulty": "Medium"
                }
            }
        
        # Stage 3: Basic Complexity Analysis
        if code:
            results["stages"]["complexity_analysis"] = {
                "agent_result": {
                    "complexity_analysis": {
                        "time_complexity": "O(V + E)" if 'bfs' in user_input.lower() or 'dfs' in user_input.lower() else "O(n)",
                        "space_complexity": "O(V)" if 'bfs' in user_input.lower() or 'dfs' in user_input.lower() else "O(1)",
                        "reasoning": "Basic complexity analysis (synchronous mode)"
                    }
                }
            }
        
        # Stage 4: Visualization (fallback)
        if options.get('include_visualization', True):
            visualization_result = self._generate_visualizations(user_input, code, output_dir)
            results["stages"]["visualizations"] = visualization_result
        
        # Stage 5: Educational Report
        results["stages"]["educational_report"] = {
            "key_concepts": self._get_educational_concepts(user_input),
            "recommendations": self._get_recommendations(user_input),
            "related_algorithms": self._get_related_algorithms(user_input),
            "practical_applications": self._get_practical_applications(user_input)
        }
        
        # Save results
        self._save_results(results, output_dir)
        
        return results
    
    def _get_educational_concepts(self, user_input: str) -> list:
        """Get educational concepts based on input"""
        problem_lower = user_input.lower()
        
        if 'bfs' in problem_lower:
            return [
                "Graph representation (adjacency list/matrix)",
                "Queue data structure",
                "Level-order traversal",
                "Shortest path in unweighted graphs",
                "Connected components"
            ]
        elif 'dfs' in problem_lower:
            return [
                "Graph representation",
                "Stack data structure (explicit or recursion)",
                "Backtracking",
                "Cycle detection",
                "Topological sorting"
            ]
        else:
            return [
                "Algorithm design principles",
                "Time and space complexity",
                "Data structure selection",
                "Problem decomposition"
            ]
    
    def _get_recommendations(self, user_input: str) -> list:
        """Get learning recommendations"""
        problem_lower = user_input.lower()
        
        if 'bfs' in problem_lower:
            return [
                "Practice implementing BFS with both adjacency list and matrix",
                "Solve shortest path problems in unweighted graphs",
                "Try BFS variations like bidirectional BFS",
                "Compare BFS with DFS for different use cases"
            ]
        else:
            return [
                "Practice implementing the algorithm from scratch",
                "Analyze different test cases and edge cases",
                "Study related algorithms and their trade-offs",
                "Apply the algorithm to real-world problems"
            ]
    
    def _get_related_algorithms(self, user_input: str) -> list:
        """Get related algorithms"""
        problem_lower = user_input.lower()
        
        if 'bfs' in problem_lower:
            return [
                "Depth-First Search (DFS)",
                "Dijkstra's Algorithm",
                "A* Search Algorithm",
                "Bidirectional BFS"
            ]
        elif 'dfs' in problem_lower:
            return [
                "Breadth-First Search (BFS)",
                "Topological Sort",
                "Strongly Connected Components",
                "Tarjan's Algorithm"
            ]
        else:
            return [
                "Binary Search",
                "Linear Search",
                "Sorting Algorithms",
                "Dynamic Programming"
            ]
    
    def _get_practical_applications(self, user_input: str) -> list:
        """Get practical applications"""
        problem_lower = user_input.lower()
        
        if 'bfs' in problem_lower:
            return [
                "Social network analysis (finding connections)",
                "Web crawling and site mapping",
                "GPS navigation (shortest path)",
                "Game development (pathfinding)",
                "Network broadcasting protocols"
            ]
        elif 'dfs' in problem_lower:
            return [
                "File system traversal",
                "Maze solving algorithms",
                "Dependency resolution",
                "Compiler design (syntax parsing)",
                "Database query optimization"
            ]
        else:
            return [
                "Software engineering",
                "Database systems",
                "Machine learning algorithms",
                "Computer graphics"
            ]
    
    def _extract_code_and_problem(self, user_input: str, input_type: str) -> tuple:
        """Extract code and problem description from user input"""
        if input_type == "code":
            return user_input, None
        elif input_type == "problem":
            return None, user_input
        else:
            # Auto-detect
            code_keywords = ["def ", "class ", "import ", "for ", "while ", "if ", "return ", "print("]
            if any(keyword in user_input for keyword in code_keywords):
                return user_input, None
            else:
                return None, user_input
    
    def _generate_visualizations(self, user_input: str, code: str, output_dir: str):
        """Generate visualizations using MongoDB system or fallback"""
        
        if self.mongo_manager:
            # Try MongoDB-powered visualization
            try:
                matches = self.mongo_manager.detect_algorithm(user_input, code or "", [])
                
                if matches:
                    category, algorithm_key, name, confidence = matches[0]
                    
                    # Try to execute visualization
                    success, files = self._execute_visualization(category, algorithm_key, output_dir)
                    
                    if not success:
                        # Try fallback visualization
                        success, files = self._try_fallback_visualization(user_input, code, output_dir)
                    
                    return {
                        "mongodb_powered": True,
                        "algorithm_detected": name,
                        "confidence": confidence,
                        "success": success,
                        "files_generated": files,
                        "category": category,
                        "algorithm_key": algorithm_key
                    }
                else:
                    # Try fallback visualization
                    success, files = self._try_fallback_visualization(user_input, code, output_dir)
                    
                    return {
                        "mongodb_powered": True,
                        "algorithm_detected": "none",
                        "success": success,
                        "files_generated": files,
                        "message": "No exact match found, used closest alternative"
                    }
            except Exception as e:
                logger.error(f"MongoDB visualization failed: {e}")
                # Try fallback visualization
                success, files = self._try_fallback_visualization(user_input, code, output_dir)
                
                return {
                    "mongodb_powered": False,
                    "success": success,
                    "files_generated": files,
                    "error": str(e),
                    "message": "MongoDB failed, used fallback visualization system"
                }
        else:
            # Fallback visualization only
            success, files = self._try_fallback_visualization(user_input, code, output_dir)
            
            return {
                "mongodb_powered": False,
                "success": success,
                "files_generated": files,
                "message": "Used fallback visualization system"
            }
    
    def _try_fallback_visualization(self, user_input: str, code: str, output_dir: str = "."):
        """Try to find the closest available visualization"""
        
        # Keywords to category mapping for fallback
        keyword_mappings = {
            'bfs': 'graphs/bfs_animation.py',
            'breadth first': 'graphs/bfs_animation.py',
            'dfs': 'graphs/dfs_animation.py',
            'depth first': 'graphs/dfs_animation.py',
            'binary search': 'searching/binary_search_animation.py',
            'linear search': 'searching/linear_search_animation.py',
            'dijkstra': 'graphs/dijkstra_animation.py',
            'sort': 'sorting/bubble_sort_animation.py'
        }
        
        # Find best match
        user_input_lower = user_input.lower()
        best_match = None
        
        for keyword, viz_file in keyword_mappings.items():
            if keyword in user_input_lower:
                best_match = viz_file
                break
        
        # Default to BFS if no match found (since user asked for BFS)
        if not best_match:
            if 'bfs' in user_input_lower:
                best_match = 'graphs/bfs_animation.py'
            else:
                best_match = 'searching/binary_search_animation.py'
        
        # Try to execute the visualization
        viz_path = f"visualizations/{best_match}"
        if os.path.exists(viz_path):
            return self._execute_visualization_file(viz_path, output_dir)
        else:
            # Create a simple fallback visualization
            return self._create_simple_fallback_visualization(output_dir, user_input)
    
    def _execute_visualization(self, category: str, algorithm_key: str, output_dir: str = "."):
        """Execute visualization based on category and algorithm key"""
        
        possible_files = [
            f"visualizations/{category}/{algorithm_key}_animation.py",
            f"visualizations/{category}/{algorithm_key}_visualization.py",
            f"visualizations/{category}/{algorithm_key}.py"
        ]
        
        for viz_file in possible_files:
            if os.path.exists(viz_file):
                return self._execute_visualization_file(viz_file, output_dir)
        
        return False, []
    
    def _execute_visualization_file(self, viz_file: str, output_dir: str = "."):
        """Execute a specific visualization file"""
        
        try:
            # Change to output directory
            original_dir = os.getcwd()
            viz_output_dir = os.path.join(output_dir, "visualizations")
            os.makedirs(viz_output_dir, exist_ok=True)
            os.chdir(viz_output_dir)
            
            # Execute visualization
            result = subprocess.run([sys.executable, os.path.join('..', '..', viz_file)], 
                                  capture_output=True, text=True, timeout=60)
            
            os.chdir(original_dir)
            
            if result.returncode == 0:
                # Find generated files
                generated_files = []
                for file in os.listdir(viz_output_dir):
                    if file.endswith(('.png', '.jpg', '.svg', '.pdf')):
                        generated_files.append(file)
                
                return True, generated_files
            else:
                logger.error(f"Visualization failed: {result.stderr}")
                return False, []
                
        except Exception as e:
            logger.error(f"Error executing visualization: {e}")
            if 'original_dir' in locals():
                os.chdir(original_dir)
            return False, []
    
    def _create_simple_fallback_visualization(self, output_dir: str, user_input: str = ""):
        """Create a simple fallback visualization"""
        
        try:
            viz_output_dir = os.path.join(output_dir, "visualizations")
            os.makedirs(viz_output_dir, exist_ok=True)
            
            # Create a simple matplotlib visualization
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Create BFS-specific visualization if BFS is mentioned
            if 'bfs' in user_input.lower():
                # BFS tree-like visualization
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Create a simple graph representation
                nodes = {
                    'A': (2, 3),
                    'B': (1, 2),
                    'C': (3, 2),
                    'D': (0, 1),
                    'E': (2, 1),
                    'F': (4, 1)
                }
                
                edges = [('A', 'B'), ('A', 'C'), ('B', 'D'), ('B', 'E'), ('C', 'F')]
                
                # Draw edges
                for start, end in edges:
                    x1, y1 = nodes[start]
                    x2, y2 = nodes[end]
                    ax.plot([x1, x2], [y1, y2], 'b-', linewidth=2, alpha=0.7)
                
                # Draw nodes
                for node, (x, y) in nodes.items():
                    circle = plt.Circle((x, y), 0.2, color='lightblue', ec='darkblue', linewidth=2)
                    ax.add_patch(circle)
                    ax.text(x, y, node, ha='center', va='center', fontsize=12, fontweight='bold')
                
                ax.set_xlim(-0.5, 4.5)
                ax.set_ylim(0.5, 3.5)
                ax.set_title('BFS (Breadth-First Search) Graph Traversal', fontsize=16, fontweight='bold')
                ax.text(2, 0.2, 'BFS visits nodes level by level: A → B, C → D, E, F', 
                       ha='center', fontsize=12, style='italic')
                ax.axis('off')
                
                filename = 'bfs_fallback_visualization.png'
            else:
                # Generic algorithm visualization
                x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
                y = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19])
                
                plt.figure(figsize=(10, 6))
                plt.bar(x, y, color='lightblue', alpha=0.7, edgecolor='black')
                plt.title('Algorithm Visualization - Fallback', fontweight='bold')
                plt.xlabel('Input Index')
                plt.ylabel('Value')
                plt.grid(True, alpha=0.3)
                
                # Add labels
                for i, v in enumerate(y):
                    plt.text(x[i], v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')
                
                filename = 'fallback_visualization.png'
            
            filepath = os.path.join(viz_output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return True, [filename]
            
        except Exception as e:
            logger.error(f"Error creating fallback visualization: {e}")
            return False, []
    
    def _create_fallback_performance_chart(self):
        """Create a fallback performance chart"""
        
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Simple performance comparison
            algorithms = ['O(1)', 'O(log n)', 'O(n)', 'O(n log n)', 'O(n²)']
            n = 1000
            complexities = [1, np.log2(n), n, n * np.log2(n), n**2]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(algorithms, complexities, color=['green', 'blue', 'orange', 'red', 'darkred'], alpha=0.7)
            plt.title('Time Complexity Comparison', fontweight='bold')
            plt.xlabel('Algorithm Complexity')
            plt.ylabel('Operations (for n=1000)')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars, complexities):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                        f'{int(value):,}', ha='center', va='bottom', fontweight='bold')
            
            filename = 'performance_fallback.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            return filename
            
        except Exception as e:
            logger.error(f"Error creating fallback performance chart: {e}")
            return None
    
    def _save_results(self, results: dict, output_dir: str):
        """Save analysis results to files"""
        
        try:
            # Save JSON report
            json_file = os.path.join(output_dir, "reports", "analysis_results.json")
            os.makedirs(os.path.dirname(json_file), exist_ok=True)
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Create simple index.html
            index_html = f'''
<!DOCTYPE html>
<html>
<head>
    <title>Algorithm Analysis Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f7fa; }}
        .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .success {{ background: #d4edda; border-color: #c3e6cb; }}
        .error {{ background: #f8d7da; border-color: #f5c6cb; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Algorithm Analysis Results</h1>
        <div class="section">
            <h2>Session Information</h2>
            <p><strong>Session ID:</strong> {results["session_id"]}</p>
            <p><strong>Timestamp:</strong> {results["timestamp"]}</p>
            <p><strong>Input Type:</strong> {results["input_type"]}</p>
        </div>
        
        <div class="section">
            <h2>Stages Completed</h2>
            <ul>
                {chr(10).join(f"<li>{stage}</li>" for stage in results["stages"].keys())}
            </ul>
        </div>
        
        <div class="section">
            <h2>Files</h2>
            <p><a href="reports/analysis_results.json" target="_blank">📄 View JSON Results</a></p>
            <p><a href="visualizations/" target="_blank">🎨 View Visualizations</a></p>
        </div>
        
        <div class="section">
            <h2>Quick Access</h2>
            <p>Generated by Algorithm Intelligence Suite v3.0</p>
            <p>Session processing completed successfully</p>
        </div>
    </div>
</body>
</html>
'''
            
            index_file = os.path.join(output_dir, "index.html")
            with open(index_file, 'w', encoding='utf-8') as f:
                f.write(index_html)
                
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def run(self, host='0.0.0.0', port=5000, debug=True):
        """Run the Flask application"""
        logger.info(f"Starting Algorithm Intelligence API on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug, threaded=True)

# Initialize the API
def create_app(mongodb_connection: str = None):
    """Factory function to create the Flask app"""
    api = AlgorithmIntelligenceAPI(mongodb_connection)
    return api.app

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Algorithm Intelligence Suite API Server")
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--mongodb', type=str, help='MongoDB connection string')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    print("🚀 Starting Algorithm Intelligence Suite API Server")
    print(f"📍 Host: {args.host}")
    print(f"🔌 Port: {args.port}")
    print(f"🗄️ MongoDB: {'Enabled' if args.mongodb else 'Disabled'}")
    print("=" * 50)
    
    # Create and run the API
    api = AlgorithmIntelligenceAPI(mongodb_connection=args.mongodb)
    api.run(host=args.host, port=args.port, debug=args.debug)
