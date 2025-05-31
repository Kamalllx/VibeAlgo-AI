# backend/api/algorithm_solver.py
from flask import Blueprint, request, jsonify
import asyncio
from core.algorithm_solver_agent import algorithm_solver

bp = Blueprint('algorithm_solver', __name__)

@bp.route('/solve', methods=['POST'])
def solve_algorithm_problem():
    """Solve algorithm problem with optimal approach and code generation"""
    
    def run_solver():
        return asyncio.run(algorithm_solver.solve_problem(
            request.json.get("problem"),
            request.json.get("user_solution")
        ))
    
    try:
        print(f"\n🧮 API ENDPOINT: /algorithm-solver/solve")
        
        data = request.get_json()
        problem = data.get("problem", "")
        user_solution = data.get("user_solution")
        
        if not problem:
            return jsonify({"success": False, "error": "Problem statement required"}), 400
        
        print(f"📝 Problem length: {len(problem)} characters")
        print(f"👤 User solution provided: {'Yes' if user_solution else 'No'}")
        
        # Run algorithm solver
        result = run_solver()
        
        return jsonify({
            "success": True,
            "algorithm_solution": result
        })
        
    except Exception as e:
        print(f"❌ Algorithm solver error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@bp.route('/analyze-solution', methods=['POST'])  
def analyze_user_solution():
    """Analyze user's solution for a problem"""
    
    def run_analysis():
        return asyncio.run(algorithm_solver.solve_problem(
            request.json.get("problem"),
            request.json.get("solution")
        ))
    
    try:
        data = request.get_json()
        problem = data.get("problem", "")
        solution = data.get("solution", "")
        
        if not problem or not solution:
            return jsonify({"success": False, "error": "Both problem and solution required"}), 400
        
        result = run_analysis()
        
        return jsonify({
            "success": True,
            "analysis": result
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
