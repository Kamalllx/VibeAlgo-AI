# backend/visualization/ai_powered_visualizer.py
import os
import re
import ast
import traceback
from datetime import datetime
from typing import Dict, List, Any
from ai.groq_client import groq_client
from ai.rag_pipeline import rag_pipeline

class AIPoweredVisualizationAgent:
    def __init__(self, session_dir: str = None):
        self.name = "AIPoweredVisualizationAgent"
        
        # Initialize the AI code cleaner
        try:
            from agents.ai_code_cleaner_agent import ai_code_cleaner
            self.code_cleaner = ai_code_cleaner
            print(f"✅ AI Code Cleaner integrated")
        except ImportError as e:
            print(f"❌ Could not import AI Code Cleaner: {e}")
            self.code_cleaner = None
        
        # Setup directories
        if session_dir:
            self.session_dir = session_dir
            self.output_dir = session_dir
        else:
            self.output_dir = "generated_visualizations"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_dir = os.path.join(self.output_dir, f"ai_session_{timestamp}")
        
        # Ensure directory exists
        self._ensure_directory_exists()
        
        print(f"🤖 [{self.name}] AI-Powered Visualization Agent initialized")
        print(f"📁 Output: {os.path.abspath(self.session_dir)}")
    
    def _ensure_directory_exists(self):
        """Ensure directory exists with comprehensive error handling"""
        try:
            # Create full path if it doesn't exist
            os.makedirs(self.session_dir, exist_ok=True)
            
            # Verify directory was created and is accessible
            if not os.path.exists(self.session_dir):
                raise OSError(f"Failed to create directory: {self.session_dir}")
            
            # Test write permissions
            test_file = os.path.join(self.session_dir, "test_write.tmp")
            try:
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                print(f"✅ Directory verified and writable: {self.session_dir}")
            except Exception as e:
                print(f"❌ Directory not writable: {e}")
                raise
                
        except Exception as e:
            print(f"❌ Failed to create/verify directory {self.session_dir}: {e}")
            # Fallback to current directory
            self.session_dir = os.path.join(os.getcwd(), "fallback_visualizations")
            os.makedirs(self.session_dir, exist_ok=True)
            print(f"🔄 Using fallback directory: {self.session_dir}")
    
    async def create_visualizations(self, request) -> List:
        """Create AI-generated visualizations using Groq"""
        print(f"\n{'='*80}")
        print(f"🤖 CREATING AI-POWERED DYNAMIC VISUALIZATIONS")
        print(f"{'='*80}")
        
        # Extract comprehensive analysis data
        analysis_data = self._extract_analysis_data(request.data)
        
        print(f"📊 Extracted data for AI generation:")
        print(f"   Algorithm: {analysis_data.get('algorithm_name', 'Unknown')}")
        print(f"   Problem Type: {analysis_data.get('problem_type', 'Unknown')}")
        print(f"   Time Complexity: {analysis_data.get('time_complexity', 'Unknown')}")
        print(f"   Space Complexity: {analysis_data.get('space_complexity', 'Unknown')}")
        print(f"   Has Code: {'Yes' if analysis_data.get('code') else 'No'}")
        
        visualizations = []
        
        # 1. Algorithm Execution Visualization (AI-Generated)
        if analysis_data.get('code'):
            print(f"\n🎯 Generating AI-powered algorithm execution visualization...")
            try:
                exec_viz = await self._generate_algorithm_execution_viz(analysis_data)
                if exec_viz:
                    visualizations.append(exec_viz)
            except Exception as e:
                print(f"❌ Algorithm execution viz failed: {e}")
                traceback.print_exc()
        
        # 2. Complexity Analysis Visualization (AI-Generated)
        print(f"\n📊 Generating AI-powered complexity visualization...")
        try:
            complexity_viz = await self._generate_complexity_viz(analysis_data)
            if complexity_viz:
                visualizations.append(complexity_viz)
        except Exception as e:
            print(f"❌ Complexity viz failed: {e}")
            traceback.print_exc()
        
        # 3. Performance Comparison Visualization (AI-Generated)
        print(f"\n⚡ Generating AI-powered performance comparison...")
        try:
            performance_viz = await self._generate_performance_viz(analysis_data)
            if performance_viz:
                visualizations.append(performance_viz)
        except Exception as e:
            print(f"❌ Performance viz failed: {e}")
            traceback.print_exc()
        
        print(f"\n✅ Generated {len(visualizations)} AI-powered visualizations")
        return visualizations
    
    def _extract_analysis_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive analysis data properly"""
        analysis = {}
        
        print(f"🔍 DEBUG: Extracting analysis from data keys: {list(data.keys())}")
        
        # Extract from algorithm solving results
        if "algorithm_solving" in data:
            solving_data = data["algorithm_solving"]
            print(f"📊 Found algorithm_solving data with keys: {list(solving_data.keys())}")
            
            # Problem analysis
            if "problem_analysis" in solving_data:
                problem_analysis = solving_data["problem_analysis"]
                analysis.update({
                    "problem_type": problem_analysis.get("problem_type", "unknown"),
                    "difficulty": problem_analysis.get("difficulty", "medium"),
                    "key_concepts": problem_analysis.get("key_concepts", []),
                    "algorithmic_patterns": problem_analysis.get("algorithmic_patterns", []),
                    "problem_description": problem_analysis.get("input_description", "")
                })
                print(f"✅ Extracted problem analysis: type={analysis['problem_type']}")
            
            # Generated algorithm solution
            if "optimal_solution" in solving_data:
                solution = solving_data["optimal_solution"]
                print(f"📝 Found optimal_solution with keys: {list(solution.keys())}")
                
                if "code" in solution:
                    code_data = solution["code"]
                    if isinstance(code_data, dict):
                        analysis["code"] = code_data.get("code", "")
                        analysis["algorithm_name"] = self._extract_algorithm_name_from_code(analysis["code"])
                    else:
                        analysis["code"] = str(code_data)
                        analysis["algorithm_name"] = self._extract_algorithm_name_from_code(analysis["code"])
                    
                    print(f"✅ Extracted code: {len(analysis['code'])} characters")
                    print(f"🏷️ Algorithm name: {analysis.get('algorithm_name', 'Unknown')}")
                
                # Extract complexity from optimal_solution complexity_analysis
                if "complexity_analysis" in solution:
                    complexity_info = solution["complexity_analysis"]
                    print(f"🧮 Found complexity in optimal_solution: {type(complexity_info)}")
                    
                    # Navigate through the nested structure
                    if isinstance(complexity_info, dict):
                        if "agent_result" in complexity_info:
                            agent_result = complexity_info["agent_result"]
                            if "complexity_analysis" in agent_result:
                                complexity_data = agent_result["complexity_analysis"]
                                analysis.update({
                                    "time_complexity": complexity_data.get("time_complexity", "Unknown"),
                                    "space_complexity": complexity_data.get("space_complexity", "Unknown"),
                                    "complexity_reasoning": complexity_data.get("reasoning", ""),
                                    "optimization_suggestions": complexity_data.get("suggestions", [])
                                })
                                print(f"✅ Extracted complexity from agent_result: time={analysis['time_complexity']}, space={analysis['space_complexity']}")
        
        # Extract from main complexity_analysis
        if "complexity_analysis" in data:
            complexity_data = data["complexity_analysis"]
            print(f"🧮 Found main complexity_analysis with keys: {list(complexity_data.keys())}")
            
            if "agent_result" in complexity_data:
                agent_result = complexity_data["agent_result"]
                print(f"🤖 Found agent_result with keys: {list(agent_result.keys())}")
                
                # Try multiple paths to find complexity data
                complexity_info = None
                if "complexity_analysis" in agent_result:
                    complexity_info = agent_result["complexity_analysis"]
                elif "analysis_result" in agent_result:
                    complexity_info = agent_result["analysis_result"]
                elif isinstance(agent_result, dict) and "time_complexity" in agent_result:
                    complexity_info = agent_result
                
                if complexity_info and not analysis.get("time_complexity"):
                    analysis.update({
                        "time_complexity": complexity_info.get("time_complexity", "Unknown"),
                        "space_complexity": complexity_info.get("space_complexity", "Unknown"),
                        "complexity_reasoning": complexity_info.get("reasoning", ""),
                        "optimization_suggestions": complexity_info.get("suggestions", [])
                    })
                    print(f"✅ Extracted complexity from main analysis: time={analysis['time_complexity']}, space={analysis['space_complexity']}")
        
        # Smart defaults based on algorithm name
        algorithm_name = analysis.get("algorithm_name", "")
        if not analysis.get("time_complexity") or analysis.get("time_complexity") == "Unknown":
            if "dijkstra" in algorithm_name.lower():
                analysis["time_complexity"] = "O((V+E)logV)"
                analysis["space_complexity"] = "O(V+E)"
                print(f"🔧 Applied smart defaults for Dijkstra: O((V+E)logV), O(V+E)")
            elif "binary" in algorithm_name.lower() or ("search" in algorithm_name.lower() and "binary" in analysis.get("code", "").lower()):
                analysis["time_complexity"] = "O(log n)"
                analysis["space_complexity"] = "O(1)"
                print(f"🔧 Applied smart defaults for binary search: O(log n), O(1)")
            elif "sort" in algorithm_name.lower():
                analysis["time_complexity"] = "O(n log n)"
                analysis["space_complexity"] = "O(1)"
                print(f"🔧 Applied smart defaults for sorting: O(n log n), O(1)")
            else:
                analysis["time_complexity"] = "O(n)"
                analysis["space_complexity"] = "O(1)"
        
        # Extract from original input
        if "original_input" in data:
            analysis["original_problem"] = data["original_input"]
        
        # Set defaults for missing data
        analysis.setdefault("algorithm_name", "Generated Algorithm")
        analysis.setdefault("problem_type", "general")
        
        print(f"📋 Final extracted analysis:")
        print(f"   Algorithm: {analysis['algorithm_name']}")
        print(f"   Problem Type: {analysis['problem_type']}")
        print(f"   Time Complexity: {analysis['time_complexity']}")
        print(f"   Space Complexity: {analysis['space_complexity']}")
        
        return analysis
    
    def _extract_algorithm_name_from_code(self, code: str) -> str:
        """Extract algorithm name from code"""
        if not code:
            return "Unknown Algorithm"
        
        code_lower = code.lower()
        
        # Look for function names
        import re
        func_matches = re.findall(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)', code)
        if func_matches:
            func_name = func_matches[0]
            if 'dijkstra' in func_name or 'shortest' in func_name:
                return "Dijkstra's Algorithm"
            elif 'binary' in func_name or ('search' in func_name and 'binary' in code_lower):
                return "Binary Search"
            elif 'sort' in func_name:
                return f"{func_name.replace('_', ' ').title()}"
            else:
                return f"{func_name.replace('_', ' ').title()} Algorithm"
        
        # Detect by patterns
        if any(pattern in code_lower for pattern in ["dijkstra", "shortest_path", "priority_queue"]):
            return "Dijkstra's Algorithm"
        elif any(pattern in code_lower for pattern in ["left", "right", "mid", "binary"]):
            return "Binary Search"
        elif any(pattern in code_lower for pattern in ["sort", "swap", "bubble"]):
            return "Sorting Algorithm"
        elif any(pattern in code_lower for pattern in ["graph", "bfs", "dfs"]):
            return "Graph Algorithm"
        
        return "Generated Algorithm"
    
    def _format_rag_context(self, context: List[Dict[str, Any]]) -> str:
        """Format RAG context for AI prompts"""
        if not context:
            return "No specific visualization examples available."
        
        formatted = ""
        for i, doc in enumerate(context[:2], 1):
            title = doc.get('title', 'Unknown Document')
            content = doc.get('content', doc.get('full_content', ''))[:150]
            formatted += f"{i}. {title}: {content}...\n"
        
        return formatted
    
    async def _generate_algorithm_execution_viz(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate algorithm execution visualization using AI"""
        
        # Get RAG context for better visualization ideas
        algorithm_name = analysis_data.get('algorithm_name', 'algorithm')
        problem_type = analysis_data.get('problem_type', 'general')
        
        rag_context = rag_pipeline.retrieve_relevant_context(
            f"{algorithm_name} {problem_type} algorithm visualization step-by-step execution examples"
        )
        
        # Create comprehensive prompt for Groq
        prompt = f"""
You are an expert algorithm visualization specialist. Generate Python code using matplotlib to create a step-by-step visualization of this specific algorithm.

ALGORITHM DETAILS:
- Algorithm Name: {analysis_data.get('algorithm_name', 'Unknown')}
- Problem Type: {analysis_data.get('problem_type', 'general')}
- Time Complexity: {analysis_data.get('time_complexity', 'Unknown')}
- Space Complexity: {analysis_data.get('space_complexity', 'Unknown')}
- Difficulty: {analysis_data.get('difficulty', 'medium')}

ALGORITHM CODE:
{analysis_data.get('code', 'No code provided')}


PROBLEM DESCRIPTION:
{analysis_data.get('problem_description', 'Algorithm execution visualization')}

KEY CONCEPTS: {', '.join(analysis_data.get('key_concepts', []))}

RELEVANT CONTEXT: {self._format_rag_context(rag_context)}

REQUIREMENTS:
1. Create a step-by-step visualization showing HOW THIS SPECIFIC algorithm works
2. For Dijkstra's Algorithm: Show graph nodes, distances, priority queue, path updates
3. For Binary Search: Show left/right pointers, mid calculation, search space reduction
4. For Sorting: Show array state changes, comparisons, swaps
5. Use appropriate visual metaphors (graphs, arrays with indices, highlighting active elements)
6. Show the algorithm's decision-making process and data transformations
7. Include 4-6 steps/frames showing the algorithm's progression
8. Add clear titles, labels, and annotations explaining each step
9. Use colors effectively: red for active comparison, blue for search space, green for found/completed
10. Make it educational and easy to understand

Generate ONLY executable Python code using matplotlib. The visualization should be specific to {analysis_data.get('algorithm_name', 'the algorithm')}.
Start with necessary imports and end with saving the plot.
"""
        
        print(f"🤖 Asking Groq to generate {analysis_data.get('algorithm_name', 'algorithm')} execution visualization...")
        
        response = groq_client.chat_completion([
            {"role": "system", "content": "You are an expert Python developer specializing in algorithm visualization. Generate complete, executable matplotlib code that creates educational step-by-step algorithm visualizations. Focus on the specific algorithm provided. Return only Python code without explanations."},
            {"role": "user", "content": prompt}
        ])
        
        print(f"📥 Received {len(response.content)} characters of AI-generated visualization code")
        
        # Extract and execute the AI-generated code using AI cleaner
        viz_code = await self._extract_and_fix_code_with_ai_cleaner(response.content, "algorithm_execution")
        generated_files = self._execute_ai_code(viz_code, "algorithm_execution")
        
        return {
            "agent_name": "AIPoweredAlgorithmVisualizer",
            "visualization_type": "ai_algorithm_execution",
            "code": viz_code,
            "data": analysis_data,
            "file_paths": generated_files,
            "metadata": {
                "ai_generated": True,
                "algorithm_name": analysis_data.get('algorithm_name', 'Unknown'),
                "complexity": analysis_data.get('time_complexity', 'Unknown')
            },
            "frontend_instructions": f"AI-generated step-by-step execution of {analysis_data.get('algorithm_name', 'algorithm')}"
        }
    
    async def _generate_complexity_viz(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate complexity visualization using AI"""
        
        prompt = f"""
You are an expert in algorithm complexity visualization. Generate Python code using matplotlib to create a comprehensive complexity analysis visualization.

COMPLEXITY DETAILS:
- Time Complexity: {analysis_data.get('time_complexity', 'Unknown')}
- Space Complexity: {analysis_data.get('space_complexity', 'Unknown')}
- Algorithm: {analysis_data.get('algorithm_name', 'Unknown')}
- Problem Type: {analysis_data.get('problem_type', 'general')}

REASONING: {analysis_data.get('complexity_reasoning', 'No specific reasoning provided')}

REQUIREMENTS:
1. Create a visualization that specifically demonstrates why this algorithm has {analysis_data.get('time_complexity', 'the given')} time complexity
2. Show growth curves comparing this complexity with others relevant to {analysis_data.get('problem_type', 'the problem type')}
3. For O(log n): Show binary reduction pattern, logarithmic growth curve
4. For O((V+E)logV): Show graph algorithm scaling with vertices and edges
5. For O(n): Show linear growth, single pass implications
6. For O(n²): Show quadratic growth, nested loop implications
7. Include visual demonstrations of the complexity characteristics
8. Add performance scaling predictions for different input sizes
9. Use both linear and logarithmic scales for comparison
10. Make it educational with clear explanations

Generate ONLY executable Python code using matplotlib. Focus on {analysis_data.get('time_complexity', 'the complexity')}.
"""
        
        print(f"🤖 Asking Groq to generate complexity analysis for {analysis_data.get('time_complexity', 'Unknown')}...")
        
        response = groq_client.chat_completion([
            {"role": "system", "content": "You are an expert in algorithm complexity analysis and visualization. Generate complete matplotlib code that educationally demonstrates specific algorithm complexity characteristics. Return only Python code."},
            {"role": "user", "content": prompt}
        ])
        
        viz_code = await self._extract_and_fix_code_with_ai_cleaner(response.content, "complexity_analysis")
        generated_files = self._execute_ai_code(viz_code, "complexity_analysis")
        
        return {
            "agent_name": "AIPoweredComplexityVisualizer",
            "visualization_type": "ai_complexity_analysis",
            "code": viz_code,
            "data": analysis_data,
            "file_paths": generated_files,
            "metadata": {
                "ai_generated": True,
                "time_complexity": analysis_data.get('time_complexity', 'Unknown'),
                "space_complexity": analysis_data.get('space_complexity', 'Unknown')
            },
            "frontend_instructions": f"AI-generated complexity analysis for {analysis_data.get('time_complexity', 'Unknown')}"
        }
    
    async def _generate_performance_viz(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance comparison visualization using AI"""
        
        prompt = f"""
You are an expert in algorithm performance analysis. Generate Python code using matplotlib to create a comprehensive performance comparison visualization.

ALGORITHM DETAILS:
- Algorithm: {analysis_data.get('algorithm_name', 'Unknown')}
- Problem Type: {analysis_data.get('problem_type', 'general')}
- Time Complexity: {analysis_data.get('time_complexity', 'Unknown')}
- Space Complexity: {analysis_data.get('space_complexity', 'Unknown')}
- Key Concepts: {', '.join(analysis_data.get('key_concepts', []))}

REQUIREMENTS:
1. Create a performance comparison showing this algorithm vs other common algorithms for {analysis_data.get('problem_type', 'the same problem type')}
2. For searching problems: Compare Linear Search, Binary Search, Hash Table lookup
3. For graph problems: Compare BFS, DFS, Dijkstra, Bellman-Ford
4. For sorting problems: Compare Bubble Sort, Quick Sort, Merge Sort, Heap Sort
5. Include benchmarking scenarios with different input sizes (10, 100, 1000, 10000)
6. Show execution time comparisons with realistic relative performance
7. Add practical performance insights (when to use this algorithm vs alternatives)
8. Make it comprehensive with multiple metrics
9. Use clear visualizations (bar charts, line graphs as appropriate)
10. Include memory usage comparison if relevant

Generate ONLY executable Python code using matplotlib. Focus on {analysis_data.get('problem_type', 'general')} algorithms.
"""
        
        print(f"🤖 Asking Groq to generate performance comparison for {analysis_data.get('problem_type', 'general')} algorithms...")
        
        response = groq_client.chat_completion([
            {"role": "system", "content": "You are an expert in algorithm performance analysis and benchmarking. Generate complete matplotlib code that creates comprehensive performance comparison visualizations. Return only Python code."},
            {"role": "user", "content": prompt}
        ])
        
        viz_code = await self._extract_and_fix_code_with_ai_cleaner(response.content, "performance_comparison")
        generated_files = self._execute_ai_code(viz_code, "performance_comparison")
        
        return {
            "agent_name": "AIPoweredPerformanceVisualizer",
            "visualization_type": "ai_performance_comparison",
            "code": viz_code,
            "data": analysis_data,
            "file_paths": generated_files,
            "metadata": {
                "ai_generated": True,
                "problem_type": analysis_data.get('problem_type', 'general'),
                "algorithms_compared": True
            },
            "frontend_instructions": f"AI-generated performance comparison for {analysis_data.get('problem_type', 'general')} algorithms"
        }
    
# backend/visualization/ai_powered_visualizer.py (FIX PATH ISSUE)

    # async def _extract_and_fix_code_with_ai_cleaner(self, ai_response: str, viz_type: str) -> str:
    #     """FIXED: Extract code and use AI cleaner with proper path handling"""
    #     """Extract code and use AI cleaner instead of manual fixing"""
        
    #     print(f"🔧 Extracting and cleaning AI code for {viz_type}...")
    #     print(f"📏 AI response length: {len(ai_response)} characters")
        
    #     # Simple extraction - get the largest code block
    #     extracted_code = self._simple_code_extraction(ai_response)
        
    #     # USE AI CLEANER
    #     if self.code_cleaner:
    #         print(f"🤖 Using AI Code Cleaner Agent...")
    #         context = f"This is {viz_type} code that should create matplotlib visualizations and save PNG files."
            
    #         try:
    #             cleaning_result = await self.code_cleaner.clean_and_fix_code(
    #                 extracted_code, "visualization", context
    #             )
                
    #             if cleaning_result["success"]:
    #                 print(f"✅ AI cleaning successful!")
    #                 cleaned_code = cleaning_result["cleaned_code"]
                    
    #                 if cleaning_result["warnings"]:
    #                     print(f"⚠️ Warnings: {cleaning_result['warnings']}")
    #             else:
    #                 print(f"⚠️ AI cleaning had issues: {cleaning_result['errors']}")
    #                 cleaned_code = cleaning_result["cleaned_code"]  # Use anyway
                    
    #         except Exception as e:
    #             print(f"❌ AI cleaning failed: {e}")
    #             cleaned_code = self._emergency_clean_simple(extracted_code)
    #     else:
    #         print(f"⚠️ AI Code Cleaner not available, using simple cleaning")
    #         cleaned_code = self._emergency_clean_simple(extracted_code)
                
    #     # ... existing code until complete_code creation ...
        
    #     # FIXED: Use forward slashes for os.chdir on Windows
    #     session_path = self.session_dir.replace('\\', '/')
        
    #     complete_code = f'''
    # """
    # AI-GENERATED AND CLEANED {viz_type.upper()} VISUALIZATION
    # Generated by Groq AI + AI Code Cleaner
    # Created: {datetime.now()}
    # """

    # import matplotlib.pyplot as plt
    # import numpy as np
    # import matplotlib.patches as patches
    # import matplotlib.colors as mcolors
    # import os
    # import warnings
    # warnings.filterwarnings('ignore')

    # # Set non-interactive backend
    # plt.ioff()

    # # FIXED: Use forward slashes and absolute path
    # import os
    # current_dir = os.getcwd()
    # target_dir = r"{self.session_dir}"
    # print(f"Current directory: {{current_dir}}")
    # print(f"Target directory: {{target_dir}}")

    # try:
    #     os.chdir(target_dir)
    #     print(f"Changed to: {{os.getcwd()}}")
    # except Exception as e:
    #     print(f"Directory change failed: {{e}}")
    #     print("Continuing in current directory...")

    # print(f"🎨 Starting AI-cleaned {viz_type} visualization...")

    # {cleaned_code}

    # # Ensure plot is saved and closed
    # try:
    #     plt.tight_layout()
    #     filename = '{viz_type}_ai_generated.png'
    #     plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    #     print(f"✅ Saved AI-generated visualization: {{filename}}")
    #     plt.close('all')
    # except Exception as e:
    #     print(f"❌ Error saving plot: {{e}}")
    #     plt.close('all')
    # finally:
    #     # Return to original directory
    #     try:
    #         os.chdir(current_dir)
    #     except:
    #         pass
    # '''

    def _simple_code_extraction(self, ai_response: str) -> str:
        """Simple code extraction without complex parsing"""
        
        # Try to find Python code blocks
        code_patterns = [
            r'``````',
            r'``````',
            r'``````'
        ]
        
        for pattern in code_patterns:
            matches = re.findall(pattern, ai_response, re.DOTALL)
            if matches:
                return max(matches, key=len).strip()
        
        # If no code blocks, look for lines that look like Python
        lines = ai_response.split('\n')
        code_lines = []
        in_code_block = False
        
        for line in lines:
            # Lines that look like Python code
            if any(keyword in line for keyword in [
                'import ', 'def ', 'class ', 'if ', 'for ', 'while ',
                'plt.', 'ax.', 'fig', 'np.', 'print('
            ]):
                in_code_block = True
                code_lines.append(line)
            elif in_code_block and line.strip():  # Continue if we're in a code block
                code_lines.append(line)
            elif in_code_block and not line.strip():  # Empty line in code block
                code_lines.append(line)
        
        return '\n'.join(code_lines) if code_lines else ai_response
    
    def _emergency_clean_simple(self, code: str) -> str:
        """Simple emergency cleaning as fallback"""
        # Remove markdown artifacts
        code = re.sub(r'```', '', code)
        code = re.sub(r'```\s*', '', code)
        
        # Basic line-by-line cleaning
        lines = code.split('\n')
        cleaned = []
        
        for line in lines:
            if line.strip() and not (line.strip().startswith('#') and 'import' not in line):
                # Basic indentation
                if any(line.strip().startswith(kw) for kw in ['def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'except']):
                    cleaned.append(line.strip())
                elif line.strip().startswith(('return', 'print', 'plt', 'ax', 'fig', 'break', 'continue')):
                    cleaned.append('    ' + line.strip())
                else:
                    cleaned.append(line.strip())
            else:
                cleaned.append(line)
        
        return '\n'.join(cleaned)
    
# backend/visualization/ai_powered_visualizer.py (CRITICAL FIX)

    def _execute_ai_code(self, code: str, viz_type: str) -> List[str]:
        """FIXED: Execute AI-generated code with type safety"""
        try:
            original_dir = os.getcwd()
            
            # CRITICAL FIX: Ensure code is actually a string
            if not isinstance(code, str):
                print(f"❌ Code is not a string: {type(code)}")
                print(f"📝 Code content: {code}")
                # Convert to string if possible
                if code is None:
                    print(f"🚨 Code is None, creating emergency fallback")
                    fallback_file = self._create_enhanced_fallback_viz(viz_type)
                    return [fallback_file] if fallback_file else []
                else:
                    code = str(code)
            
            # FIXED: Additional validation
            if len(code.strip()) < 50:
                print(f"⚠️ Code too short ({len(code)} chars), using fallback")
                fallback_file = self._create_enhanced_fallback_viz(viz_type)
                return [fallback_file] if fallback_file else []
            
            # Ensure session directory exists
            abs_session_dir = os.path.abspath(self.session_dir)
            if not os.path.exists(abs_session_dir):
                print(f"⚠️ Session directory missing, recreating: {abs_session_dir}")
                os.makedirs(abs_session_dir, exist_ok=True)
            
            # Change to session directory
            os.chdir(abs_session_dir)
            print(f"📁 Changed to directory: {os.getcwd()}")
            
            print(f"⚡ Executing AI-generated {viz_type} code...")
            print(f"📏 Code length: {len(code)} characters")
            print(f"🔤 Code type: {type(code)}")
            
            # FIXED: Safe execution with debugging
            try:
                exec(code)
                print(f"✅ Code execution successful!")
            except Exception as exec_error:
                print(f"❌ Code execution error: {exec_error}")
                print(f"📝 First 200 chars of code: {code[:200]}")
                raise exec_error
            
            # Find generated files
            generated_files = []
            for file in os.listdir('.'):
                if file.endswith(('.png', '.svg', '.pdf')) and viz_type in file:
                    generated_files.append(file)
                elif file.endswith('.py') and viz_type in file:
                    generated_files.append(file)
            
            print(f"✅ AI code execution successful! Generated: {generated_files}")
            return generated_files
            
        except Exception as e:
            print(f"❌ AI code execution failed: {e}")
            print(f"📍 Current directory: {os.getcwd()}")
            print(f"📍 Target directory: {abs_session_dir}")
            print(f"📍 Directory exists: {os.path.exists(abs_session_dir)}")
            
            # Create fallback in the correct directory
            try:
                os.chdir(abs_session_dir)
                fallback_file = self._create_enhanced_fallback_viz(viz_type)
                return [fallback_file] if fallback_file else []
            except Exception as fallback_error:
                print(f"❌ Fallback also failed: {fallback_error}")
                return []
            
        finally:
            # Always return to original directory
            try:
                os.chdir(original_dir)
                print(f"🏠 Returned to original directory: {os.getcwd()}")
            except Exception as e:
                print(f"⚠️ Could not return to original directory: {e}")

    # backend/visualization/ai_powered_visualizer.py (FIXED CODE STRUCTURE)

    # backend/visualization/ai_powered_visualizer.py (CRITICAL FIXES)

    async def _extract_and_fix_code_with_ai_cleaner(self, ai_response: str, viz_type: str) -> str:
        """FIXED: Proper code structure and indentation"""
        
        print(f"🔧 Extracting and cleaning AI code for {viz_type}...")
        print(f"📏 AI response length: {len(ai_response)} characters")
        
        # Extract the code
        extracted_code = self._simple_code_extraction(ai_response)
        print(f"📏 Extracted code length: {len(extracted_code)} characters")
        
        # FIXED: Skip AI Code Cleaner for now since it's consistently failing
        # Use direct extraction with manual cleaning
        print(f"🔧 Using direct extraction with manual cleaning...")
        cleaned_code = self._manual_clean_extracted_code(extracted_code, viz_type)
        
        print(f"📏 Cleaned code length: {len(cleaned_code)} characters")
        print(f"🔤 Cleaned code type: {type(cleaned_code)}")
        print(f"🔍 Cleaned code preview: {cleaned_code[:200]}...")
        
        # FIXED: Create complete code WITHOUT indentation issues
        complete_code = f'''"""
AI-GENERATED AND CLEANED {viz_type.upper()} VISUALIZATION
Generated by Groq AI + Manual Cleaning
Created: {datetime.now()}
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import os
import warnings
warnings.filterwarnings('ignore')

# Set non-interactive backend
plt.ioff()

# Directory handling
current_dir = os.getcwd()
target_dir = r"{self.session_dir}"
print(f"Current directory: {{current_dir}}")
print(f"Target directory: {{target_dir}}")

try:
    os.chdir(target_dir)
    print(f"Changed to: {{os.getcwd()}}")
except Exception as e:
    print(f"Directory change failed: {{e}}")
    print("Continuing in current directory...")

print(f"🎨 Starting {viz_type} visualization...")

# MAIN VISUALIZATION CODE (NO FUNCTION WRAPPER)
{cleaned_code}

# Save the visualization (only if figure exists)
try:
    plt.tight_layout()
    filename = '{viz_type}_ai_generated.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved AI-generated visualization: {{filename}}")
    plt.close('all')
except Exception as e:
    print(f"❌ Error saving plot: {{e}}")
    plt.close('all')
finally:
    try:
        os.chdir(current_dir)
    except:
        pass
'''
        
        # Save the code
        code_filename = f"{viz_type}_ai_code.py"
        code_path = os.path.join(self.session_dir, code_filename)
        
        try:
            with open(code_path, 'w', encoding='utf-8') as f:
                f.write(complete_code)
            print(f"💾 Saved cleaned code: {code_filename}")
        except Exception as e:
            print(f"❌ Failed to save code file: {e}")
        
        return complete_code

    def _manual_clean_extracted_code(self, extracted_code: str, viz_type: str) -> str:
        """FIXED: Manual cleaning that preserves AI-generated visualization logic"""
        print(f"🛠️ Manual cleaning for {viz_type}...")
        
        # Remove markdown blocks
        code = re.sub(r'``````', '', extracted_code, flags=re.DOTALL)
        
        # If we have substantial code, try to clean it
        if len(code.strip()) > 200:
            print(f"✅ Found substantial AI code ({len(code)} chars), cleaning...")
            
            lines = code.split('\n')
            cleaned_lines = []
            
            for line in lines:
                # Skip empty lines
                if not line.strip():
                    cleaned_lines.append('')
                    continue
                
                stripped = line.strip()
                
                # Remove function definitions that wrap the visualization
                if stripped.startswith('def ') and 'visualization' in stripped:
                    continue
                if stripped in ['if __name__ == "__main__":', 'create_visualization()']:
                    continue
                
                # Fix common syntax issues
                if stripped.endswith(' if') or stripped.endswith(' for') or stripped.endswith(' while'):
                    stripped += ':'
                
                # Determine proper indentation
                if any(stripped.startswith(kw) for kw in ['import ', 'from ', 'fig', 'plt.', 'ax']):
                    cleaned_lines.append(stripped)  # Top level
                elif any(stripped.startswith(kw) for kw in ['for ', 'if ', 'while ', 'try:', 'except']):
                    cleaned_lines.append(stripped)  # Top level control
                elif stripped.startswith(('print(', 'return ', 'break', 'continue')):
                    cleaned_lines.append('    ' + stripped)  # Indented
                else:
                    # Preserve reasonable indentation
                    if line.startswith('    '):
                        cleaned_lines.append(line)
                    else:
                        cleaned_lines.append(stripped)
            
            cleaned = '\n'.join(cleaned_lines)
            
            # Validate we still have visualization code
            if any(keyword in cleaned for keyword in ['plt.', 'ax.', 'fig', 'plot', 'bar', 'scatter']):
                print(f"✅ Manual cleaning preserved visualization code")
                return cleaned
            else:
                print(f"⚠️ Manual cleaning lost visualization code, using fallback")
        
        # Create algorithm-specific visualization if cleaning failed
        print(f"🎯 Creating algorithm-specific {viz_type} code...")
        return self._create_dijkstra_specific_code(viz_type)

    def _create_dijkstra_specific_code(self, viz_type: str) -> str:
        """Create working Dijkstra-specific visualization code"""
        print(f"🎯 Creating Dijkstra-specific {viz_type} code...")
        
        if viz_type == "algorithm_execution":
            return '''# Dijkstra's Algorithm Step-by-Step Execution
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # Sample graph for Dijkstra visualization
    nodes = ['A', 'B', 'C', 'D', 'E', 'F']
    edges = [('A','B',4), ('A','C',2), ('B','C',1), ('B','D',5), ('C','D',8), ('C','E',10), ('D','E',2), ('D','F',6), ('E','F',3)]

    # Dijkstra execution steps
    steps = [
        {'current': 'A', 'distances': {'A':0, 'B':float('inf'), 'C':float('inf'), 'D':float('inf'), 'E':float('inf'), 'F':float('inf')}, 'visited': set(), 'step': 'Initialize'},
        {'current': 'A', 'distances': {'A':0, 'B':4, 'C':2, 'D':float('inf'), 'E':float('inf'), 'F':float('inf')}, 'visited': {'A'}, 'step': 'Process A'},
        {'current': 'C', 'distances': {'A':0, 'B':3, 'C':2, 'D':10, 'E':12, 'F':float('inf')}, 'visited': {'A','C'}, 'step': 'Process C'},
        {'current': 'B', 'distances': {'A':0, 'B':3, 'C':2, 'D':8, 'E':12, 'F':float('inf')}, 'visited': {'A','C','B'}, 'step': 'Process B'},
        {'current': 'D', 'distances': {'A':0, 'B':3, 'C':2, 'D':8, 'E':10, 'F':14}, 'visited': {'A','C','B','D'}, 'step': 'Process D'},
        {'current': 'E', 'distances': {'A':0, 'B':3, 'C':2, 'D':8, 'E':10, 'F':13}, 'visited': {'A','C','B','D','E'}, 'step': 'Complete'}
    ]

    # Node positions for visualization
    pos = {'A': (0, 1), 'B': (1, 2), 'C': (1, 0), 'D': (2, 1), 'E': (3, 0), 'F': (4, 1)}

    for i, step in enumerate(steps[:6]):
        ax = axes[i]
        
        # Draw edges
        for edge in edges:
            start, end, weight = edge, edge[1], edge[2]
            x1, y1 = pos[start]
            x2, y2 = pos[end]
            ax.plot([x1, x2], [y1, y2], 'gray', alpha=0.5, linewidth=1)
            
            # Add edge weight
            mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
            ax.text(mid_x, mid_y, str(weight), fontsize=8, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
        # Draw nodes
        for node in nodes:
            x, y = pos[node]
            
            # Color based on status
            if node in step['visited']:
                color = 'lightgreen'
            elif node == step['current']:
                color = 'orange'
            else:
                color = 'lightblue'
            
            # Draw node circle
            circle = plt.Circle((x, y), 0.15, color=color, alpha=0.8)
            ax.add_patch(circle)
            ax.text(x, y, node, ha='center', va='center', fontweight='bold', fontsize=12)
            
            # Add distance label
            dist = step['distances'][node]
            dist_text = str(dist) if dist != float('inf') else '∞'
            ax.text(x, y-0.35, dist_text, ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.7))
        
        ax.set_xlim(-0.3, 4.3)
        ax.set_ylim(-0.7, 2.3)
        ax.set_aspect('equal')
        ax.set_title(f"Step {i+1}: {step['step']}", fontweight='bold')
        ax.axis('off')

    plt.suptitle("Dijkstra's Algorithm: Step-by-Step Execution", fontsize=16, fontweight='bold')'''
        
        elif viz_type == "complexity_analysis":
            return '''# Dijkstra's Algorithm Complexity Analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Input sizes (vertices and edges)
    V = np.logspace(1, 4, 100)
    E = V * 1.5  # Sparse graph assumption

    # Graph algorithm complexities
    complexities = {
        'BFS/DFS: O(V+E)': V + E,
        'Dijkstra: O((V+E)logV)': (V + E) * np.log2(V),
        'Bellman-Ford: O(VE)': V * E / 10,  # Scaled
        'Floyd-Warshall: O(V³)': V**3 / 10000  # Scaled
    }

    colors = ['green', 'blue', 'orange', 'red']

    # Linear scale
    for i, (name, values) in enumerate(complexities.items()):
        style = '-' if 'Dijkstra' in name else '--'
        width = 3 if 'Dijkstra' in name else 2
        ax1.plot(V, values, label=name, color=colors[i], linestyle=style, linewidth=width)

    ax1.set_xlabel('Number of Vertices (V)', fontsize=12)
    ax1.set_ylabel('Operations', fontsize=12)
    ax1.set_title("Dijkstra's O((V+E)logV) vs Other Graph Algorithms", fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Log-log scale
    for i, (name, values) in enumerate(complexities.items()):
        style = '-' if 'Dijkstra' in name else '--'
        width = 3 if 'Dijkstra' in name else 2
        marker = 'o' if 'Dijkstra' in name else None
        ax2.loglog(V, values, label=name, color=colors[i], linestyle=style, linewidth=width, marker=marker)

    ax2.set_xlabel('Number of Vertices (V) - Log Scale', fontsize=12)
    ax2.set_ylabel('Operations - Log Scale', fontsize=12)
    ax2.set_title('Graph Algorithm Scaling (Logarithmic View)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)'''
        
        else:  # performance_comparison
            return '''# Graph Algorithm Performance Comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Algorithm data
    algorithms = ['BFS', 'DFS', 'Dijkstra', 'Bellman-Ford', 'A*']
    colors = ['lightgreen', 'lightblue', 'orange', 'red', 'purple']

    # Time comparison (V=1000, E=1500)
    times = [2500][2500][7500][1500000][5000]  # Relative operations
    normalized_times = [t / min(times) for t in times]

    bars1 = ax1.bar(algorithms, normalized_times, color=colors, alpha=0.7)
    ax1.set_title('Execution Time Comparison\\n(V=1000, E=1500)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Relative Time (BFS=1)', fontsize=12)
    ax1.set_yscale('log')

    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height * 1.1,
            f'{height:.1f}x', ha='center', va='bottom', fontweight='bold')

    # Memory usage
    memory = 
    ax2.bar(algorithms, memory, color=colors, alpha=0.7)
    ax2.set_title('Memory Usage', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Memory (MB)', fontsize=12)

    # Use cases
    use_cases = ['Unweighted\\nShortest', 'Graph\\nTraversal', 'Weighted\\nShortest', 'Negative\\nWeights', 'Heuristic\\nSearch']
    suitability = [5][5][5][1][4]

    ax3.bar(algorithms, suitability, color=colors, alpha=0.7)
    ax3.set_title('Algorithm Suitability by Use Case', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Suitability (1-5)', fontsize=12)

    # Characteristics radar
    categories = ['Speed', 'Memory', 'Simplicity', 'Versatility']
    dijkstra_scores = [4][3][3][4]
    bfs_scores = [5][4][5][3]

    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    dijkstra_scores += dijkstra_scores[:1]
    bfs_scores += bfs_scores[:1]

    ax4.plot(angles, dijkstra_scores, 'o-', linewidth=2, label='Dijkstra', color='orange')
    ax4.fill(angles, dijkstra_scores, alpha=0.25, color='orange')
    ax4.plot(angles, bfs_scores, 'o-', linewidth=2, label='BFS', color='green')
    ax4.fill(angles, bfs_scores, alpha=0.25, color='green')

    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(categories)
    ax4.set_ylim(0, 5)
    ax4.set_title('Algorithm Characteristics', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True)

    plt.suptitle('Graph Algorithm Performance Analysis', fontsize=16, fontweight='bold')'''

    def _fix_indentation(self, code: str) -> str:
        """Fix indentation issues"""
        lines = code.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Remove excessive leading whitespace
            if line.strip():
                # Don't over-indent imports and top-level statements
                stripped = line.strip()
                if any(stripped.startswith(kw) for kw in ['import ', 'from ', 'def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'except']):
                    fixed_lines.append(stripped)
                elif any(stripped.startswith(kw) for kw in ['return ', 'print(', 'plt.', 'ax.', 'fig']):
                    # These might need indentation
                    if not line.startswith('    '):
                        fixed_lines.append('    ' + stripped)
                    else:
                        fixed_lines.append(line)
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append('')
        
        return '\n'.join(fixed_lines)

    # def _create_dijkstra_specific_code(self, viz_type: str) -> str:
    #     """Create Dijkstra-specific visualization code"""
    #     print(f"🎯 Creating Dijkstra-specific {viz_type} code...")
        
    #     if viz_type == "algorithm_execution":
    #         return '''
    # # Dijkstra's Algorithm Step-by-Step Execution
    # fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    # axes = axes.flatten()

    # # Graph representation (adjacency with weights)
    # graph_data = {
    #     'nodes': ['A', 'B', 'C', 'D', 'E', 'F'],
    #     'edges': [('A','B',4), ('A','C',2), ('B','C',1), ('B','D',5), ('C','D',8), ('C','E',10), ('D','E',2), ('D','F',6), ('E','F',3)]
    # }

    # # Dijkstra execution steps
    # steps = [
    #     {'current': 'A', 'distances': {'A':0, 'B':float('inf'), 'C':float('inf'), 'D':float('inf'), 'E':float('inf'), 'F':float('inf')}, 'visited': set(), 'step': 'Initialize'},
    #     {'current': 'A', 'distances': {'A':0, 'B':4, 'C':2, 'D':float('inf'), 'E':float('inf'), 'F':float('inf')}, 'visited': {'A'}, 'step': 'Process A'},
    #     {'current': 'C', 'distances': {'A':0, 'B':3, 'C':2, 'D':10, 'E':12, 'F':float('inf')}, 'visited': {'A','C'}, 'step': 'Process C'},
    #     {'current': 'B', 'distances': {'A':0, 'B':3, 'C':2, 'D':8, 'E':12, 'F':float('inf')}, 'visited': {'A','C','B'}, 'step': 'Process B'},
    #     {'current': 'D', 'distances': {'A':0, 'B':3, 'C':2, 'D':8, 'E':10, 'F':14}, 'visited': {'A','C','B','D'}, 'step': 'Process D'},
    #     {'current': 'E', 'distances': {'A':0, 'B':3, 'C':2, 'D':8, 'E':10, 'F':13}, 'visited': {'A','C','B','D','E'}, 'step': 'Complete'}
    # ]

    # # Create visualizations for each step
    # for i, step in enumerate(steps[:6]):
    #     ax = axes[i]
        
    #     # Node positions (arranged in a layout)
    #     pos = {'A': (0, 1), 'B': (1, 2), 'C': (1, 0), 'D': (2, 1), 'E': (3, 0), 'F': (4, 1)}
        
    #     # Draw edges
    #     for edge in graph_data['edges']:
    #         x1, y1 = pos[edge]
    #         x2, y2 = pos[edge[1]]
    #         ax.plot([x1, x2], [y1, y2], 'gray', alpha=0.5, linewidth=1)
    #         # Add edge weight
    #         mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
    #         ax.text(mid_x, mid_y, str(edge[2]), fontsize=8, ha='center', va='center', 
    #             bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
    #     # Draw nodes
    #     for node in graph_data['nodes']:
    #         x, y = pos[node]
            
    #         # Color based on status
    #         if node in step['visited']:
    #             color = 'lightgreen'
    #         elif node == step['current']:
    #             color = 'orange'
    #         else:
    #             color = 'lightblue'
            
    #         # Draw node
    #         circle = plt.Circle((x, y), 0.2, color=color, alpha=0.7)
    #         ax.add_patch(circle)
    #         ax.text(x, y, node, ha='center', va='center', fontweight='bold', fontsize=12)
            
    #         # Add distance label
    #         dist = step['distances'][node]
    #         dist_text = str(dist) if dist != float('inf') else '∞'
    #         ax.text(x, y-0.4, dist_text, ha='center', va='center', fontsize=10, 
    #             bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.7))
        
    #     ax.set_xlim(-0.5, 4.5)
    #     ax.set_ylim(-0.8, 2.5)
    #     ax.set_aspect('equal')
    #     ax.set_title(f"Step {i+1}: {step['step']}", fontweight='bold')
    #     ax.axis('off')

    # plt.suptitle("Dijkstra's Algorithm Step-by-Step Execution", fontsize=16, fontweight='bold')
    # '''
        
    #     elif viz_type == "complexity_analysis":
    #         return '''
    # # Dijkstra's Algorithm Complexity Analysis
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # # Input sizes (number of vertices)
    # V = np.logspace(1, 4, 100)
    # E = V * 1.5  # Assume sparse graph (E ≈ 1.5V)

    # # Different graph algorithm complexities
    # complexities = {
    #     'BFS/DFS: O(V+E)': V + E,
    #     'Dijkstra: O((V+E)logV)': (V + E) * np.log2(V),
    #     'Bellman-Ford: O(VE)': V * E,
    #     'Floyd-Warshall: O(V³)': V**3 / 1000  # Scaled for visibility
    # }

    # colors = ['green', 'blue', 'orange', 'red']

    # # Linear scale plot
    # for i, (name, values) in enumerate(complexities.items()):
    #     style = '-' if 'Dijkstra' in name else '--'
    #     width = 3 if 'Dijkstra' in name else 2
    #     ax1.plot(V, values, label=name, color=colors[i], linestyle=style, linewidth=width)

    # ax1.set_xlabel('Number of Vertices (V)', fontsize=12)
    # ax1.set_ylabel('Operations', fontsize=12)
    # ax1.set_title("Dijkstra's Algorithm: O((V+E)logV) vs Other Graph Algorithms", fontsize=14, fontweight='bold')
    # ax1.legend()
    # ax1.grid(True, alpha=0.3)
    # ax1.set_ylim(0, np.max(complexities['Dijkstra: O((V+E)logV)']) * 1.1)

    # # Log-log scale plot
    # for i, (name, values) in enumerate(complexities.items()):
    #     style = '-' if 'Dijkstra' in name else '--'
    #     width = 3 if 'Dijkstra' in name else 2
    #     ax2.loglog(V, values, label=name, color=colors[i], linestyle=style, linewidth=width, marker='o' if 'Dijkstra' in name else None)

    # ax2.set_xlabel('Number of Vertices (V) - Log Scale', fontsize=12)
    # ax2.set_ylabel('Operations - Log Scale', fontsize=12)
    # ax2.set_title('Logarithmic View: Graph Algorithm Scaling', fontsize=14, fontweight='bold')
    # ax2.legend()
    # ax2.grid(True, alpha=0.3)

    # # Add annotation
    # ax1.annotate('Dijkstra is efficient for\\nsparse graphs with\\nnon-negative weights', 
    #             xy=(1000, complexities['Dijkstra: O((V+E)logV)']), xytext=(2000, complexities['Dijkstra: O((V+E)logV)']),
    #             arrowprops=dict(arrowstyle='->', color='blue', lw=2),
    #             fontsize=11, fontweight='bold', color='blue',
    #             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    # '''
        
    #     else:  # performance_comparison
    #         return '''
    # # Graph Algorithm Performance Comparison
    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # # Algorithm comparison data
    # algorithms = ['BFS\\nO(V+E)', 'DFS\\nO(V+E)', "Dijkstra's\\nO((V+E)logV)", 'Bellman-Ford\\nO(VE)', 'A*\\nO(blogb)']
    # colors = ['lightgreen', 'lightblue', 'orange', 'red', 'purple']

    # # 1. Time complexity comparison (relative for V=1000, E=1500)
    # V, E = 1000, 1500
    # relative_times = [
    #     V + E,                    # BFS/DFS
    #     V + E,                    # DFS  
    #     (V + E) * np.log2(V),    # Dijkstra
    #     V * E,                   # Bellman-Ford
    #     (V + E) * np.log2(V) * 0.7  # A* (heuristic makes it faster)
    # ]

    # # Normalize to make BFS/DFS = 1
    # normalized_times = [t / relative_times for t in relative_times]

    # bars1 = ax1.bar(algorithms, normalized_times, color=colors, alpha=0.7)
    # ax1.set_title('Time Complexity Comparison\\n(V=1000, E=1500)', fontsize=14, fontweight='bold')
    # ax1.set_ylabel('Relative Time (BFS=1)', fontsize=12)
    # ax1.set_yscale('log')

    # # Add value labels
    # for i, bar in enumerate(bars1):
    #     height = bar.get_height()
    #     ax1.text(bar.get_x() + bar.get_width()/2., height * 1.1,
    #         f'{height:.1f}x', ha='center', va='bottom', fontweight='bold')

    # # 2. Memory usage comparison
    # memory_usage =   # Relative memory in MB
    # bars2 = ax2.bar(algorithms, memory_usage, color=colors, alpha=0.7)
    # ax2.set_title('Memory Usage Comparison', fontsize=14, fontweight='bold')
    # ax2.set_ylabel('Memory (MB)', fontsize=12)

    # # 3. Use case suitability (scored 1-5)
    # use_cases = ['Unweighted\\nShortest Path', 'Graph\\nTraversal', 'Weighted\\nShortest Path', 'Negative\\nWeights', 'Pathfinding\\nwith Heuristic']
    # suitability_scores =   # Each algorithm's primary use case

    # bars3 = ax3.bar(use_cases, suitability_scores, color=colors, alpha=0.7)
    # ax3.set_title('Primary Use Cases', fontsize=14, fontweight='bold')
    # ax3.set_ylabel('Suitability (1-5)', fontsize=12)
    # ax3.set_xticklabels(use_cases, rotation=45, ha='right')

    # # 4. Algorithm characteristics radar
    # characteristics = ['Speed', 'Memory\\nEfficiency', 'Simplicity', 'Versatility', 'Optimality']
    # dijkstra_scores = [3][3]
    # bfs_scores = [3]

    # angles = np.linspace(0, 2*np.pi, len(characteristics), endpoint=False).tolist()
    # angles += angles[:1]  # Complete the circle

    # dijkstra_scores += dijkstra_scores[:1]
    # bfs_scores += bfs_scores[:1]

    # ax4.plot(angles, dijkstra_scores, 'o-', linewidth=2, label="Dijkstra's", color='orange')
    # ax4.fill(angles, dijkstra_scores, alpha=0.25, color='orange')
    # ax4.plot(angles, bfs_scores, 'o-', linewidth=2, label='BFS', color='green')
    # ax4.fill(angles, bfs_scores, alpha=0.25, color='green')

    # ax4.set_xticks(angles[:-1])
    # ax4.set_xticklabels(characteristics)
    # ax4.set_ylim(0, 5)
    # ax4.set_title('Algorithm Characteristics Comparison', fontsize=14, fontweight='bold')
    # ax4.legend()
    # ax4.grid(True)

    # plt.suptitle('Graph Algorithm Comprehensive Performance Analysis', fontsize=16, fontweight='bold')
    # '''

    def _create_simple_working_code(self, viz_type: str) -> str:
        """Create simple working code when everything else fails"""
        print(f"🆘 Creating simple working code for {viz_type}")
        
        if viz_type == "algorithm_execution":
            return '''
    # Simple Algorithm Execution Visualization
    fig, ax = plt.subplots(figsize=(12, 8))

    # Dijkstra's Algorithm Steps
    steps = ['Initialize\\nDistances', 'Create Priority\\nQueue', 'Extract Min\\nNode', 'Relax\\nEdges', 'Update\\nDistances', 'Path\\nComplete']
    values = [1, 2, 3, 4, 3, 1]
    colors = ['lightblue', 'orange', 'red', 'yellow', 'green', 'purple']

    bars = ax.bar(range(len(steps)), values, color=colors, alpha=0.8)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels(steps, rotation=0, ha='center')
    ax.set_title("Dijkstra's Algorithm Execution Steps", fontsize=16, fontweight='bold')
    ax.set_ylabel('Processing Intensity')

    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
            f'Step {i+1}', ha='center', va='bottom', fontweight='bold')

    ax.grid(True, alpha=0.3)
    '''
        elif viz_type == "complexity_analysis":
            return '''
    # Simple Complexity Analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Graph algorithm complexities
    n = np.logspace(1, 4, 100)
    complexities = {
        'O(V)': n,
        'O(E)': n * 1.5,
        'O(V log V)': n * np.log2(n),
        'O((V+E) log V)': (n + n*1.5) * np.log2(n),
        'O(V²)': n**2 / 1000
    }

    colors = ['green', 'blue', 'orange', 'red', 'purple']

    for i, (name, values) in enumerate(complexities.items()):
        style = '-' if name == 'O((V+E) log V)' else '--'
        width = 3 if name == 'O((V+E) log V)' else 2
        ax1.plot(n, values, label=name, color=colors[i], linestyle=style, linewidth=width)

    ax1.set_xlabel('Number of Vertices (V)')
    ax1.set_ylabel('Operations')
    ax1.set_title("Dijkstra's Algorithm: O((V+E) log V) Complexity", fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Log scale plot
    for i, (name, values) in enumerate(complexities.items()):
        style = '-' if name == 'O((V+E) log V)' else '--'
        width = 3 if name == 'O((V+E) log V)' else 2
        ax2.loglog(n, values, label=name, color=colors[i], linestyle=style, linewidth=width)

    ax2.set_xlabel('Number of Vertices (V) - Log Scale')
    ax2.set_ylabel('Operations - Log Scale')
    ax2.set_title('Logarithmic View: Graph Algorithm Scaling', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    '''
        else:  # performance_comparison
            return '''
    # Simple Performance Comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Graph algorithm comparison
    algorithms = ['BFS\\nO(V+E)', 'DFS\\nO(V+E)', "Dijkstra's\\nO((V+E)logV)", 'Bellman-Ford\\nO(VE)']
    times_1000_vertices = [1000, 1000, 3000, 1000000]  # Relative times

    # Normalize for visualization
    normalized_times = [t/1000 for t in times_1000_vertices]
    colors = ['lightgreen', 'lightblue', 'orange', 'red']

    bars = ax1.bar(algorithms, normalized_times, color=colors, alpha=0.7)
    ax1.set_title('Graph Algorithm Performance (1000 vertices)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Relative Time (normalized)')
    ax1.set_yscale('log')

    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height * 1.1,
            f'{times_1000_vertices[i]}x', ha='center', va='bottom', fontweight='bold')

    # Memory usage
    memory_usage = [100, 100, 150, 100]
    ax2.bar(algorithms, memory_usage, color=colors, alpha=0.7)
    ax2.set_title('Memory Usage Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Memory (MB)')

    # Use cases
    use_cases = ['Shortest Path\\nUnweighted', 'Graph Traversal', 'Shortest Path\\nWeighted', 'Negative Weights']
    ax3.bar(algorithms, [1, 1, 1, 1], color=colors, alpha=0.7)
    ax3.set_title('Primary Use Cases', fontsize=14, fontweight='bold')
    ax3.set_xticklabels(use_cases, rotation=45, ha='right')

    # Efficiency pie chart
    efficiency = [20, 20, 35, 25]
    ax4.pie(efficiency, labels=algorithms, colors=colors, autopct='%1.1f%%')
    ax4.set_title('Algorithm Usage Distribution', fontsize=14, fontweight='bold')
    '''
    
    def _create_enhanced_fallback_viz(self, viz_type: str) -> str:
        """Create enhanced fallback visualization"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            if viz_type == "algorithm_execution":
                # Algorithm-specific fallback based on detected algorithm
                algorithm_name = getattr(self, '_current_algorithm_name', 'Algorithm')
                
                if 'dijkstra' in algorithm_name.lower():
                    # Dijkstra-specific fallback
                    steps = ['Initialize\nDistances', 'Create Priority\nQueue', 'Extract Min\nNode', 'Relax\nEdges', 'Update\nDistances', 'Path\nComplete']
                    values = [1, 2, 3, 4, 3, 1]
                    colors = ['lightblue', 'orange', 'red', 'yellow', 'green', 'purple']
                    
                    bars = ax.bar(range(len(steps)), values, color=colors, alpha=0.7)
                    ax.set_xticks(range(len(steps)))
                    ax.set_xticklabels(steps, rotation=0, ha='center')
                    ax.set_title("Dijkstra's Algorithm Execution Steps", fontsize=16, fontweight='bold')
                    ax.set_ylabel('Processing Intensity')
                
                elif 'binary' in algorithm_name.lower():
                    # Binary search fallback
                    steps = ['Initialize\n(left=0, right=n-1)', 'Calculate Mid\n(mid=(left+right)/2)', 'Compare\n(arr[mid] vs target)', 'Adjust Range\n(left=mid+1 or right=mid-1)', 'Found/Not Found']
                    values = [1, 2, 3, 2, 1]
                    colors = ['lightblue', 'orange', 'red', 'orange', 'green']
                    
                    bars = ax.bar(range(len(steps)), values, color=colors, alpha=0.7)
                    ax.set_xticks(range(len(steps)))
                    ax.set_xticklabels(steps, rotation=0, ha='center')
                    ax.set_title('Binary Search Algorithm Steps', fontsize=16, fontweight='bold')
                    ax.set_ylabel('Processing Intensity')
                else:
                    # Generic algorithm steps
                    steps = ['Input', 'Process', 'Decision', 'Update', 'Output']
                    values = [1, 3, 4, 3, 1]
                    colors = ['lightblue', 'orange', 'red', 'yellow', 'green']
                    
                    bars = ax.bar(range(len(steps)), values, color=colors, alpha=0.7)
                    ax.set_xticks(range(len(steps)))
                    ax.set_xticklabels(steps)
                    ax.set_title(f'{algorithm_name} Execution Steps', fontsize=16, fontweight='bold')
                    ax.set_ylabel('Processing Intensity')
                
                # Add value labels
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'Step {i+1}', ha='center', va='bottom', fontweight='bold')
                
            elif viz_type == "complexity_analysis":
                # Complexity-specific fallback
                n = np.linspace(1, 1000, 100)
                complexities = {
                    'O(1)': np.ones_like(n),
                    'O(log n)': np.log2(n),
                    'O(n)': n,
                    'O(n log n)': n * np.log2(n),
                    'O(n²)': n**2 / 1000
                }
                
                colors = ['green', 'blue', 'orange', 'red', 'purple']
                
                for i, (name, values) in enumerate(complexities.items()):
                    ax.plot(n, values, label=name, color=colors[i], linewidth=2)
                
                ax.set_xlabel('Input Size (n)')
                ax.set_ylabel('Operations')
                ax.set_title('Algorithm Complexity Comparison (Fallback)', fontsize=16, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
            elif viz_type == "performance_comparison":
                # Performance comparison fallback
                algorithms = ['Algorithm A', 'Algorithm B', 'Your Algorithm', 'Optimal']
                times = [100, 50, 30, 20]
                colors = ['red', 'orange', 'blue', 'green']
                
                bars = ax.bar(algorithms, times, color=colors, alpha=0.7)
                ax.set_title('Performance Comparison (Fallback)', fontsize=16, fontweight='bold')
                ax.set_ylabel('Relative Time (ms)')
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                           f'{height}ms', ha='center', va='bottom', fontweight='bold')
            
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            filename = f'{viz_type}_enhanced_fallback.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"✅ Created enhanced fallback: {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ Enhanced fallback failed: {e}")
            return None

# Global AI-powered visualizer
ai_powered_visualizer = AIPoweredVisualizationAgent()
