#!/usr/bin/env python3
"""
ALGORITHM INTELLIGENCE SUITE - COMPREHENSIVE CLI
Complete pipeline demonstrating all project objectives
"""

import asyncio
import os
import json
import time
import argparse
from datetime import datetime
from typing import Dict, Any, List

# Import all the agents and systems
from core.agent_orchestrator import orchestrator
from core.algorithm_solver_agent import algorithm_solver

class AlgorithmIntelligenceSuite:
    def __init__(self):
        self.name = "Algorithm Intelligence Suite"
        self.version = "2.0"
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create SINGLE session directory for ALL outputs
        self.output_dir = f"algorithm_intelligence_results_{self.session_id}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create subdirectories within the main output directory
        self.visualizations_dir = os.path.join(self.output_dir, "visualizations")
        self.reports_dir = os.path.join(self.output_dir, "reports") 
        self.data_dir = os.path.join(self.output_dir, "data")
        
        os.makedirs(self.visualizations_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        print(f"🤖 {self.name} v{self.version}")
        print(f"📁 Main Directory: {os.path.abspath(self.output_dir)}")
        print(f"🎨 Visualizations: {os.path.abspath(self.visualizations_dir)}")
        print(f"📄 Reports: {os.path.abspath(self.reports_dir)}")
        print(f"🔬 Session ID: {self.session_id}")
        
    async def process_algorithm_request(self, user_input: str, input_type: str = "auto") -> Dict[str, Any]:
        """
        Complete pipeline processing user input through all objectives:
        1. Complexity Analysis with RAG
        2. Algorithm Problem Solving  
        3. Dynamic Visualizations
        4. Educational Enhancements
        """
        print(f"\n{'='*80}")
        print(f"🚀 PROCESSING ALGORITHM REQUEST")
        print(f"{'='*80}")
        print(f"📝 Input Type: {input_type}")
        print(f"📏 Input Length: {len(user_input)} characters")
        print(f"⏰ Start Time: {datetime.now().strftime('%H:%M:%S')}")
        
        results = {
            "session_id": self.session_id,
            "input": user_input,
            "input_type": input_type,
            "timestamp": datetime.now().isoformat(),
            "stages": {}
        }
        
        try:
            # STAGE 1: DETERMINE INPUT TYPE AND EXTRACT CODE
            print(f"\n🔍 STAGE 1: ANALYZING INPUT TYPE")
            code, problem_description = self._extract_code_and_problem(user_input, input_type)
            
            results["stages"]["input_analysis"] = {
                "detected_type": "code" if code else "problem",
                "code_extracted": code,
                "problem_description": problem_description
            }
            
            print(f"✅ Input analysis completed")
            print(f"   Code detected: {'Yes' if code else 'No'}")
            print(f"   Problem description: {'Yes' if problem_description else 'No'}")
            
            # STAGE 2: COMPLEXITY ANALYSIS (if code provided)
            if code:
                print(f"\n📊 STAGE 2: COMPLEXITY ANALYSIS WITH RAG")
                complexity_result = await orchestrator.process_request("complexity_analysis", {
                    "code": code,
                    "language": "python",
                    "platform": "general"
                })
                
                results["stages"]["complexity_analysis"] = complexity_result
                print(f"✅ Complexity analysis completed")
                
                # Extract key metrics
                if "agent_result" in complexity_result:
                    agent_result = complexity_result["agent_result"]
                    complexity_data = agent_result.get("complexity_analysis", {})
                    print(f"   Time Complexity: {complexity_data.get('time_complexity', 'Unknown')}")
                    print(f"   Space Complexity: {complexity_data.get('space_complexity', 'Unknown')}")
                    print(f"   RAG Enhanced: {len(agent_result.get('enhanced_rag_context', [])) > 0}")
            
            # STAGE 3: ALGORITHM PROBLEM SOLVING (if problem provided)
            if problem_description or not code:
                print(f"\n🧮 STAGE 3: ALGORITHM PROBLEM SOLVING")
                problem_input = problem_description or user_input
                
                solving_result = await algorithm_solver.solve_problem(problem_input, code)
                results["stages"]["algorithm_solving"] = solving_result
                print(f"✅ Algorithm solving completed")
                
                # Extract generated solution
                if "optimal_solution" in solving_result:
                    solution_code = solving_result["optimal_solution"].get("code", {})
                    if isinstance(solution_code, dict):
                        generated_code = solution_code.get("code", "")
                        print(f"   Generated solution: {len(generated_code)} characters")
                        
                        # Use generated code for further analysis if no original code
                        if not code and generated_code:
                            code = generated_code
                            print(f"   Using generated code for subsequent analysis")
            

# Add to imports
            from visualization_database.visualization_manager import visualization_manager

# Replace the visualization section in process_algorithm_request:
            # STAGE 4: DATABASE-POWERED VISUALIZATIONS
            print(f"\n🎨 STAGE 4: CREATING DATABASE-POWERED VISUALIZATIONS")
            
            # Extract user input and generated code
            user_problem = user_input
            generated_code = ""
            if "algorithm_solving" in results["stages"]:
                solution = results["stages"]["algorithm_solving"].get("optimal_solution", {})
                if "code" in solution:
                    code_data = solution["code"]
                    if isinstance(code_data, dict):
                        generated_code = code_data.get("code", "")
                    else:
                        generated_code = str(code_data)
            
            # Use visualization database
            visualization_success = visualization_manager.auto_visualize(user_problem, generated_code)
            
            # Get algorithm info for results
            best_match = visualization_manager.get_best_visualization(user_problem, generated_code)
            if best_match:
                category, algorithm, viz_type = best_match
                algo_info = visualization_manager.get_algorithm_info(category, algorithm)
                
                results["stages"]["visualizations"] = {
                    "database_powered": True,
                    "algorithm_detected": f"{category}/{algorithm}",
                    "algorithm_name": algo_info["name"],
                    "visualization_type": viz_type,
                    "success": visualization_success,
                    "files_generated": ["Generated by database system"],
                    "directory": "Current working directory"
                }
            else:
                results["stages"]["visualizations"] = {
                    "database_powered": True,
                    "algorithm_detected": "none",
                    "success": False,
                    "message": "No matching algorithm found in database"
                }
            
            print(f"   ✅ Database visualization: {'Success' if visualization_success else 'Failed'}")
            # STAGE 5: EDUCATIONAL REPORT GENERATION
            print(f"\n📚 STAGE 5: GENERATING EDUCATIONAL REPORT")
            
            educational_report = self._generate_educational_report(results)
            results["stages"]["educational_report"] = educational_report
            
            print(f"✅ Educational report generated")
            
            # STAGE 6: SAVE COMPREHENSIVE RESULTS
            print(f"\n💾 STAGE 6: SAVING COMPREHENSIVE RESULTS")
            
            # Save detailed JSON report in reports directory
            report_file = os.path.join(self.reports_dir, f"algorithm_intelligence_report_{self.session_id}.json")
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Generate HTML report in reports directory
            html_report = self._generate_html_report(results)
            html_file = os.path.join(self.reports_dir, f"algorithm_intelligence_report_{self.session_id}.html")
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_report)
            
            # Create index file pointing to all resources
            index_file = self._create_index_file(results)
            
            print(f"✅ Results saved:")
            print(f"   📄 JSON Report: {report_file}")
            print(f"   🌐 HTML Report: {html_file}")
            print(f"   📊 Index File: {index_file}")
            print(f"   🎨 Visualizations: {self.visualizations_dir}")
            
            # Open main results directory
            self._open_results_directory()
            
            return results
            
        except Exception as e:
            print(f"❌ Pipeline processing failed: {e}")
            import traceback
            traceback.print_exc()
            
            results["error"] = str(e)
            return results
    
    def _extract_code_and_problem(self, user_input: str, input_type: str) -> tuple:
        """Extract code and problem description from user input"""
        
        if input_type == "code":
            return user_input, None
        elif input_type == "problem":
            return None, user_input
        else:
            # Auto-detect
            if "def " in user_input or "class " in user_input or "import " in user_input:
                # Looks like code
                return user_input, None
            else:
                # Looks like a problem description
                return None, user_input
    
    def _generate_educational_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate educational analysis and recommendations"""
        
        report = {
            "learning_objectives": [],
            "key_concepts": [],
            "recommendations": [],
            "complexity_insights": [],
            "optimization_suggestions": []
        }
        
        # Extract insights from complexity analysis
        if "complexity_analysis" in results["stages"]:
            complexity = results["stages"]["complexity_analysis"]
            if "agent_result" in complexity:
                complexity_data = complexity["agent_result"].get("complexity_analysis", {})
                
                time_complexity = complexity_data.get("time_complexity", "Unknown")
                space_complexity = complexity_data.get("space_complexity", "Unknown")
                
                report["key_concepts"].extend([
                    f"Time Complexity: {time_complexity}",
                    f"Space Complexity: {space_complexity}"
                ])
                
                report["complexity_insights"].append(
                    f"This algorithm has {time_complexity} time complexity and {space_complexity} space complexity."
                )
                
                # Add specific recommendations based on complexity
                if "O(n²)" in time_complexity:
                    report["optimization_suggestions"].append(
                        "Consider optimizing this O(n²) algorithm using divide-and-conquer or dynamic programming approaches."
                    )
                elif "O(n log n)" in time_complexity:
                    report["recommendations"].append(
                        "This algorithm has good time complexity. Consider optimizing space usage if needed."
                    )
                elif "O(log n)" in time_complexity:
                    report["recommendations"].append(
                        "Excellent logarithmic time complexity! This algorithm scales very well with large inputs."
                    )
        
        # Extract insights from algorithm solving
        if "algorithm_solving" in results["stages"]:
            solving = results["stages"]["algorithm_solving"]
            
            if "problem_analysis" in solving:
                problem_analysis = solving["problem_analysis"]
                problem_type = problem_analysis.get("problem_type", "general")
                report["learning_objectives"].append(f"Understand {problem_type} algorithms")
                report["key_concepts"].append(f"Algorithm category: {problem_type}")
                
                # Add problem-specific insights
                if problem_type == "searching":
                    report["recommendations"].append(
                        "For searching problems, consider the trade-offs between time complexity and space requirements."
                    )
                elif problem_type == "sorting":
                    report["recommendations"].append(
                        "For sorting problems, consider whether stability and in-place sorting are required."
                    )
        
        # Add general learning objectives
        report["learning_objectives"].extend([
            "Analyze algorithm time and space complexity",
            "Understand Big O notation and growth rates", 
            "Compare different algorithmic approaches",
            "Visualize algorithm performance and behavior"
        ])
        
        return report
    
    def _generate_html_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive HTML report"""
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Algorithm Intelligence Report - {self.session_id}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #f5f7fa; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px 10px 0 0; }}
        .header h1 {{ margin: 0; font-size: 28px; }}
        .header p {{ margin: 10px 0 0 0; opacity: 0.9; }}
        .section {{ padding: 25px; border-bottom: 1px solid #eee; }}
        .section h2 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        .stage {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 15px 0; }}
        .stage h3 {{ color: #495057; margin-top: 0; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .metric {{ background: #e3f2fd; padding: 15px; border-radius: 6px; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #1976d2; }}
        .metric-label {{ color: #666; font-size: 14px; }}
        .code-block {{ background: #2d3748; color: #e2e8f0; padding: 20px; border-radius: 6px; overflow-x: auto; font-family: 'Courier New', monospace; }}
        .insights {{ background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 6px; padding: 15px; margin: 15px 0; }}
        .recommendations {{ background: #d1ecf1; border: 1px solid #bee5eb; border-radius: 6px; padding: 15px; margin: 15px 0; }}
        ul {{ padding-left: 20px; }}
        li {{ margin: 8px 0; }}
        .success {{ color: #28a745; font-weight: bold; }}
        .info {{ color: #17a2b8; font-weight: bold; }}
        .warning {{ color: #ffc107; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Algorithm Intelligence Analysis Report</h1>
            <p>Session ID: {self.session_id} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2>📊 Executive Summary</h2>
            <div class="metrics">
"""
        
        # Add metrics
        total_stages = len([k for k in results["stages"].keys() if not k.endswith("_error")])
        
        html += f"""
                <div class="metric">
                    <div class="metric-value">{total_stages}</div>
                    <div class="metric-label">Analysis Stages</div>
                </div>
"""
        
        if "complexity_analysis" in results["stages"]:
            complexity = results["stages"]["complexity_analysis"]
            if "agent_result" in complexity:
                complexity_data = complexity["agent_result"].get("complexity_analysis", {})
                html += f"""
                <div class="metric">
                    <div class="metric-value">{complexity_data.get('time_complexity', 'Unknown')}</div>
                    <div class="metric-label">Time Complexity</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{complexity_data.get('space_complexity', 'Unknown')}</div>
                    <div class="metric-label">Space Complexity</div>
                </div>
"""
        
        if "visualizations" in results["stages"]:
            viz_count = results["stages"]["visualizations"]["total_created"]
            html += f"""
                <div class="metric">
                    <div class="metric-value">{viz_count}</div>
                    <div class="metric-label">AI Visualizations</div>
                </div>
"""
        
        html += """
            </div>
        </div>
"""
        
        # Add each stage
        for stage_name, stage_data in results["stages"].items():
            if stage_name == "educational_report":
                continue
                
            html += f"""
        <div class="section">
            <h2>🔬 {stage_name.replace('_', ' ').title()}</h2>
            <div class="stage">
"""
            
            if stage_name == "complexity_analysis" and "agent_result" in stage_data:
                agent_result = stage_data["agent_result"]
                complexity_data = agent_result.get("complexity_analysis", {})
                
                html += f"""
                <h3>Complexity Analysis Results</h3>
                <p><strong>Time Complexity:</strong> <span class="success">{complexity_data.get('time_complexity', 'Unknown')}</span></p>
                <p><strong>Space Complexity:</strong> <span class="info">{complexity_data.get('space_complexity', 'Unknown')}</span></p>
                <p><strong>Analysis Confidence:</strong> {agent_result.get('confidence_score', 0.85):.1%}</p>
                
                <div class="insights">
                    <strong>💡 Complexity Insights:</strong>
                    <p>{complexity_data.get('reasoning', 'No specific reasoning available.')}</p>
                </div>
                
                <div class="recommendations">
                    <strong>🚀 Optimization Suggestions:</strong>
                    <ul>
"""
                
                suggestions = complexity_data.get('suggestions', ['Consider algorithm optimization opportunities', 'Analyze input data characteristics', 'Evaluate space-time trade-offs'])
                for suggestion in suggestions[:3]:
                    html += f"<li>{suggestion}</li>"
                
                html += "</ul></div>"
                
            elif stage_name == "algorithm_solving" and "optimal_solution" in stage_data:
                solution = stage_data["optimal_solution"]
                problem_analysis = stage_data.get("problem_analysis", {})
                
                html += f"""
                <h3>Algorithm Solution</h3>
                <p><strong>Problem Type:</strong> <span class="info">{problem_analysis.get('problem_type', 'Unknown')}</span></p>
                <p><strong>Difficulty:</strong> <span class="warning">{problem_analysis.get('difficulty', 'Medium')}</span></p>
                
                <div class="code-block">
{solution.get('code', {}).get('code', 'No code generated') if isinstance(solution.get('code'), dict) else str(solution.get('code', 'No code generated'))}
                </div>
"""
                
            elif stage_name == "visualizations":
                html += f"""
                <h3>AI-Generated Visualizations</h3>
                <p><strong>Total Created:</strong> <span class="success">{stage_data.get('total_created', 0)}</span></p>
                <p><strong>AI Generated:</strong> <span class="info">{stage_data.get('ai_generated', False)}</span></p>
                <p><strong>Groq Powered:</strong> <span class="info">{stage_data.get('groq_powered', False)}</span></p>
                <p><strong>Files:</strong> {', '.join(stage_data.get('files', []))}</p>
                <p><strong>Directory:</strong> <code>{stage_data.get('directory', 'visualizations/')}</code></p>
"""
            
            html += """
            </div>
        </div>
"""
        
        # Add educational report
        if "educational_report" in results["stages"]:
            edu_report = results["stages"]["educational_report"]
            
            html += f"""
        <div class="section">
            <h2>📚 Educational Analysis</h2>
            <div class="stage">
                <h3>Learning Objectives</h3>
                <ul>
"""
            
            for objective in edu_report.get("learning_objectives", []):
                html += f"<li>{objective}</li>"
            
            html += f"""
                </ul>
                
                <h3>Key Concepts</h3>
                <ul>
"""
            
            for concept in edu_report.get("key_concepts", []):
                html += f"<li>{concept}</li>"
            
            html += f"""
                </ul>
                
                <div class="recommendations">
                    <strong>📈 Recommendations for Improvement:</strong>
                    <ul>
"""
            
            for rec in edu_report.get("optimization_suggestions", []):
                html += f"<li>{rec}</li>"
            
            for rec in edu_report.get("recommendations", []):
                html += f"<li>{rec}</li>"
            
            html += """
                    </ul>
                </div>
            </div>
        </div>
"""
        
        html += f"""
        <div class="section">
            <h2>🎯 Summary</h2>
            <p>This comprehensive analysis processed your algorithm through multiple AI agents, providing insights into complexity, performance, and optimization opportunities. The generated visualizations help understand the algorithm's behavior and characteristics.</p>
            <p><strong class="success">Session completed successfully at {datetime.now().strftime('%H:%M:%S')}</strong></p>
            <p><strong>📁 All results saved to:</strong> <code>{self.output_dir}</code></p>
        </div>
    </div>
</body>
</html>
"""
        
        return html
    
    def _create_index_file(self, results: Dict[str, Any]) -> str:
        """Create an index file showing all generated content"""
        
        index_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Algorithm Intelligence Suite - Session {self.session_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f7fa; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        .section {{ margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }}
        .file-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
        .file-item {{ padding: 15px; border: 1px solid #eee; border-radius: 5px; text-align: center; }}
        .file-link {{ display: inline-block; margin: 10px; padding: 8px 16px; background: #007bff; color: white; text-decoration: none; border-radius: 4px; }}
        .file-link:hover {{ background: #0056b3; }}
        .viz-files {{ background: #e8f5e8; }}
        .report-files {{ background: #e8f0ff; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Algorithm Intelligence Suite Results</h1>
            <p><strong>Session ID:</strong> {self.session_id}</p>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section viz-files">
            <h2>🎨 AI-Generated Visualizations</h2>
            <p><strong>Directory:</strong> <code>visualizations/</code></p>
            <div class="file-grid">
                <div class="file-item">
                    <h3>Algorithm Execution</h3>
                    <p>Step-by-step algorithm visualization</p>
                    <a href="visualizations/" class="file-link">View Visualizations</a>
                </div>
                <div class="file-item">
                    <h3>Complexity Analysis</h3>
                    <p>Big O complexity charts and comparisons</p>
                    <a href="visualizations/" class="file-link">View Charts</a>
                </div>
                <div class="file-item">
                    <h3>Performance Comparison</h3>
                    <p>Algorithm performance benchmarks</p>
                    <a href="visualizations/" class="file-link">View Performance</a>
                </div>
            </div>
        </div>
        
        <div class="section report-files">
            <h2>📄 Analysis Reports</h2>
            <p><strong>Directory:</strong> <code>reports/</code></p>
            <div class="file-grid">
                <div class="file-item">
                    <h3>Comprehensive Report</h3>
                    <p>Complete analysis including complexity, algorithms, and insights</p>
                    <a href="reports/algorithm_intelligence_report_{self.session_id}.html" class="file-link">View HTML Report</a>
                    <a href="reports/algorithm_intelligence_report_{self.session_id}.json" class="file-link">View JSON Data</a>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>📁 Directory Structure</h2>
            <pre style="background: #f8f9fa; padding: 20px; border-radius: 5px; overflow-x: auto;">
algorithm_intelligence_results_{self.session_id}/
├── index.html (this file)
├── visualizations/
│   ├── algorithm_execution_ai_generated.png
│   ├── complexity_analysis_ai_generated.png
│   ├── performance_comparison_ai_generated.png
│   ├── *_ai_code.py (generated code files)
│   └── *_enhanced_fallback.png (fallback visualizations)
├── reports/
│   ├── algorithm_intelligence_report_{self.session_id}.html
│   └── algorithm_intelligence_report_{self.session_id}.json
└── data/
    └── (any additional data files)
            </pre>
        </div>
        
        <div class="section">
            <h2>🚀 Next Steps</h2>
            <ul>
                <li>📊 Review the generated visualizations to understand algorithm behavior</li>
                <li>📄 Read the comprehensive report for detailed insights</li>
                <li>🎨 Use the generated Python code for your own projects</li>
                <li>📈 Apply the optimization suggestions to improve your algorithms</li>
            </ul>
        </div>
    </div>
</body>
</html>
"""
        
        index_file = os.path.join(self.output_dir, "index.html")
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write(index_html)
        
        print(f"📊 Created index file: {index_file}")
        return index_file
    
    def _open_results_directory(self):
        """Open the main results directory"""
        try:
            import platform
            import subprocess
            
            abs_path = os.path.abspath(self.output_dir)
            system = platform.system()
            
            if system == "Windows":
                subprocess.run(f'explorer "{abs_path}"', shell=True)
            elif system == "Darwin":
                subprocess.run(["open", abs_path])
            elif system == "Linux":
                subprocess.run(["xdg-open", abs_path])
            
            print(f"📁 Opened main directory: {abs_path}")
            
        except Exception as e:
            print(f"⚠️ Could not open directory: {e}")
            print(f"📁 Manual path: {os.path.abspath(self.output_dir)}")

async def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="Algorithm Intelligence Suite - Comprehensive Analysis")
    parser.add_argument('--input', '-i', type=str, help='Algorithm code or problem description')
    parser.add_argument('--type', '-t', choices=['code', 'problem', 'auto'], default='auto', 
                       help='Input type (default: auto-detect)')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    # Initialize the suite
    suite = AlgorithmIntelligenceSuite()
    
    if args.interactive or not args.input:
        # Interactive mode
        print(f"\n🎯 Welcome to Algorithm Intelligence Suite!")
        print(f"Please provide your algorithm code or problem description:")
        print(f"(Press Ctrl+C to exit)")
        
        try:
            while True:
                print(f"\n" + "="*60)
                user_input = input("📝 Enter your algorithm/problem: ").strip()
                
                if not user_input:
                    print("❌ Please provide some input")
                    continue
                
                input_type = input("🔍 Input type (code/problem/auto) [auto]: ").strip() or "auto"
                
                print(f"\n🚀 Processing your request...")
                results = await suite.process_algorithm_request(user_input, input_type)
                
                if "error" not in results:
                    print(f"\n✅ Analysis completed successfully!")
                    print(f"📁 Results saved to: {suite.output_dir}")
                    print(f"🌐 Open index.html to view all results")
                else:
                    print(f"\n❌ Analysis failed: {results['error']}")
                
                continue_prompt = input("\n🔄 Process another algorithm? (y/n): ").strip().lower()
                if continue_prompt not in ['y', 'yes']:
                    break
                    
        except KeyboardInterrupt:
            print(f"\n👋 Thank you for using Algorithm Intelligence Suite!")
    
    else:
        # Command line mode
        print(f"\n🚀 Processing your request...")
        results = await suite.process_algorithm_request(args.input, args.type)
        
        if "error" not in results:
            print(f"\n✅ Analysis completed successfully!")
            print(f"📁 Results saved to: {suite.output_dir}")
        else:
            print(f"\n❌ Analysis failed: {results['error']}")

if __name__ == "__main__":
    asyncio.run(main())
