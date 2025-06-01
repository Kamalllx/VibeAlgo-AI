# backend/visualization/dynamic_visualization_orchestrator.py
import os
import ast
import re
from datetime import datetime
from typing import Dict, List, Any, Tuple
from ai.groq_client import groq_client
from ai.rag_pipeline import rag_pipeline

class DynamicVisualizationOrchestrator:
    def __init__(self):
        self.name = "DynamicVisualizationOrchestrator"
        self.output_dir = "generated_visualizations"
        
        # Create session directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(self.output_dir, f"dynamic_session_{timestamp}")
        os.makedirs(self.session_dir, exist_ok=True)
        
        print(f"🎨 [{self.name}] Dynamic Visualization Orchestrator initialized")
        print(f"📁 Output: {os.path.abspath(self.session_dir)}")
    
    async def create_visualizations(self, request) -> List:
        """Create TRULY DYNAMIC visualizations based on actual analysis"""
        print(f"\n{'='*80}")
        print(f"🎨 CREATING DYNAMIC AI-GENERATED VISUALIZATIONS")
        print(f"{'='*80}")
        
        # Extract actual data from the analysis
        analysis_data = self._extract_comprehensive_analysis(request.data)
        
        print(f"📊 Analysis extracted:")
        print(f"   Algorithm Type: {analysis_data.get('algorithm_type', 'Unknown')}")
        print(f"   Time Complexity: {analysis_data.get('time_complexity', 'Unknown')}")
        print(f"   Problem Type: {analysis_data.get('problem_type', 'Unknown')}")
        print(f"   Has Code: {'Yes' if analysis_data.get('code') else 'No'}")
        
        # Generate scenario-specific visualizations
        visualizations = []
        
        # 1. Algorithm-Specific Execution Visualization
        if analysis_data.get('code'):
            print(f"\n🔍 Generating algorithm-specific execution visualization...")
            exec_viz = await self._generate_algorithm_execution_viz(analysis_data)
            if exec_viz:
                visualizations.append(exec_viz)
        
        # 2. Complexity-Specific Analysis Visualization  
        print(f"\n📊 Generating complexity-specific visualization...")
        complexity_viz = await self._generate_complexity_specific_viz(analysis_data)
        if complexity_viz:
            visualizations.append(complexity_viz)
        
        # 3. Problem-Type Specific Visualization
        print(f"\n🎯 Generating problem-type specific visualization...")
        problem_viz = await self._generate_problem_type_viz(analysis_data)
        if problem_viz:
            visualizations.append(problem_viz)
        
        print(f"\n✅ Generated {len(visualizations)} dynamic visualizations")
        return visualizations
    
    def _extract_comprehensive_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract all analysis data comprehensively"""
        analysis = {}
        
        print(f"🔍 Extracting analysis from request data...")
        
        # Extract from algorithm solving results
        if "algorithm_solving" in data:
            solving_data = data["algorithm_solving"]
            
            # Problem analysis
            if "problem_analysis" in solving_data:
                problem_analysis = solving_data["problem_analysis"]
                analysis.update({
                    "problem_type": problem_analysis.get("problem_type", "unknown"),
                    "difficulty": problem_analysis.get("difficulty", "medium"),
                    "key_concepts": problem_analysis.get("key_concepts", []),
                    "algorithmic_patterns": problem_analysis.get("algorithmic_patterns", [])
                })
            
            # Generated code
            if "optimal_solution" in solving_data and "code" in solving_data["optimal_solution"]:
                code_data = solving_data["optimal_solution"]["code"]
                if isinstance(code_data, dict):
                    analysis["code"] = code_data.get("code", "")
                else:
                    analysis["code"] = str(code_data)
        
        # Extract from complexity analysis
        if "complexity_analysis" in data:
            complexity_data = data["complexity_analysis"]
            if "agent_result" in complexity_data:
                agent_result = complexity_data["agent_result"]
                if "complexity_analysis" in agent_result:
                    complexity_info = agent_result["complexity_analysis"]
                    analysis.update({
                        "time_complexity": complexity_info.get("time_complexity", "Unknown"),
                        "space_complexity": complexity_info.get("space_complexity", "Unknown"),
                        "complexity_reasoning": complexity_info.get("reasoning", "")
                    })
        
        # Detect algorithm type from code
        if analysis.get("code"):
            analysis["algorithm_type"] = self._detect_algorithm_type(analysis["code"])
        
        return analysis
    
    def _detect_algorithm_type(self, code: str) -> str:
        """Detect specific algorithm type from code"""
        code_lower = code.lower()
        
        # Binary search patterns
        if any(pattern in code_lower for pattern in ["left", "right", "mid", "binary", "//", "left <= right"]):
            return "binary_search"
        
        # Sorting patterns
        elif any(pattern in code_lower for pattern in ["bubble", "sort", "swap", "range(len"]):
            return "sorting"
        
        # Graph patterns  
        elif any(pattern in code_lower for pattern in ["graph", "bfs", "dfs", "queue", "stack", "visited"]):
            return "graph_algorithm"
        
        # Dynamic programming
        elif any(pattern in code_lower for pattern in ["dp", "memo", "cache", "dp[", "dynamic"]):
            return "dynamic_programming"
        
        # Tree patterns
        elif any(pattern in code_lower for pattern in ["tree", "node", "left", "right", "root"]):
            return "tree_algorithm"
        
        return "general_algorithm"
    
    async def _generate_algorithm_execution_viz(self, analysis_data: Dict[str, Any]) -> str:
        """Generate visualization showing actual algorithm execution"""
        
        algorithm_type = analysis_data.get("algorithm_type", "general")
        code = analysis_data.get("code", "")
        
        # Get RAG context for this specific algorithm type
        rag_context = rag_pipeline.retrieve_relevant_context(
            f"{algorithm_type} algorithm visualization step by step execution"
        )
        
        viz_prompt = f"""
        Generate Python visualization code that shows the SPECIFIC execution of this algorithm:
        
        Algorithm Type: {algorithm_type}
        Code:
        ```
        {code}
        ```
        
        Analysis: {analysis_data}
        
        RAG Context: {self._format_rag_context(rag_context)}
        
        Create visualization code that:
        1. Shows step-by-step execution of THIS SPECIFIC algorithm
        2. Uses the ACTUAL algorithm logic, not generic examples
        3. Creates meaningful visual representation of the algorithm's behavior
        4. Shows key decision points and data transformations
        
        For {algorithm_type}, focus on:
        - Binary search: Show left/right pointer movement, mid calculation, search space reduction
        - Sorting: Show actual array state changes, comparisons, swaps
        - Graph: Show node exploration, path finding, traversal order
        - Tree: Show tree traversal, node visits, path decisions
        
        Return ONLY executable Python code using matplotlib.
        """
        
        print(f"🤖 Generating {algorithm_type} execution visualization...")
        response = groq_client.chat_completion([
            {"role": "system", "content": "You are an expert at creating algorithm-specific visualizations. Generate precise, relevant visualization code."},
            {"role": "user", "content": viz_prompt}
        ])
        
        # Extract and save the generated code
        viz_code = self._extract_and_enhance_code(response.content, f"{algorithm_type}_execution")
        
        # Execute the code to generate visualization
        generated_files = self._execute_visualization_code(viz_code, f"{algorithm_type}_execution")
        
        return {
            "agent_name": "DynamicAlgorithmVisualizer",
            "visualization_type": f"{algorithm_type}_execution",
            "code": viz_code,
            "data": analysis_data,
            "file_paths": generated_files,
            "metadata": {"algorithm_specific": True, "dynamic_generated": True},
            "frontend_instructions": f"Interactive {algorithm_type} execution visualization"
        }
    
    async def _generate_complexity_specific_viz(self, analysis_data: Dict[str, Any]) -> str:
        """Generate visualization specific to the algorithm's complexity"""
        
        time_complexity = analysis_data.get("time_complexity", "O(n)")
        space_complexity = analysis_data.get("space_complexity", "O(1)")
        algorithm_type = analysis_data.get("algorithm_type", "general")
        
        viz_prompt = f"""
        Generate Python visualization code that shows the SPECIFIC complexity characteristics of this algorithm:
        
        Time Complexity: {time_complexity}
        Space Complexity: {space_complexity}
        Algorithm Type: {algorithm_type}
        
        Create visualization that:
        1. Shows growth curve for {time_complexity} specifically
        2. Compares with other complexities relevant to {algorithm_type}
        3. Demonstrates why this algorithm has {time_complexity} complexity
        4. Shows space usage pattern for {space_complexity}
        
        For {time_complexity}:
        - O(log n): Show binary reduction, logarithmic growth
        - O(n): Show linear growth, single pass implications  
        - O(n²): Show quadratic growth, nested loop implications
        - O(n log n): Show divide-conquer efficiency
        
        Return ONLY executable Python code using matplotlib.
        """
        
        print(f"📊 Generating {time_complexity} complexity visualization...")
        response = groq_client.chat_completion([
            {"role": "system", "content": "You are an expert at complexity analysis visualization. Create precise complexity-specific charts."},
            {"role": "user", "content": viz_prompt}
        ])
        
        viz_code = self._extract_and_enhance_code(response.content, f"complexity_{time_complexity.replace('(', '').replace(')', '').replace(' ', '_')}")
        generated_files = self._execute_visualization_code(viz_code, f"complexity_analysis")
        
        return {
            "agent_name": "DynamicComplexityVisualizer", 
            "visualization_type": "complexity_specific",
            "code": viz_code,
            "data": analysis_data,
            "file_paths": generated_files,
            "metadata": {"complexity_focused": True, "time_complexity": time_complexity},
            "frontend_instructions": f"Complexity analysis for {time_complexity}"
        }
    
    async def _generate_problem_type_viz(self, analysis_data: Dict[str, Any]) -> str:
        """Generate visualization specific to the problem type"""
        
        problem_type = analysis_data.get("problem_type", "general")
        key_concepts = analysis_data.get("key_concepts", [])
        
        viz_prompt = f"""
        Generate Python visualization code specific to this problem type:
        
        Problem Type: {problem_type}
        Key Concepts: {key_concepts}
        Analysis: {analysis_data}
        
        Create visualization that demonstrates:
        1. Problem-specific patterns and characteristics
        2. Visual representation of the problem domain
        3. Solution approach visualization
        4. Key insights specific to {problem_type} problems
        
        For {problem_type}:
        - Searching: Show search space, target finding, elimination
        - Sorting: Show ordering process, comparison strategies
        - Graph: Show connectivity, paths, traversal patterns
        - Dynamic Programming: Show subproblem relationships, memoization
        
        Return ONLY executable Python code using matplotlib.
        """
        
        print(f"🎯 Generating {problem_type} problem-type visualization...")
        response = groq_client.chat_completion([
            {"role": "system", "content": "You are an expert at problem-domain visualization. Create problem-specific visual insights."},
            {"role": "user", "content": viz_prompt}
        ])
        
        viz_code = self._extract_and_enhance_code(response.content, f"problem_type_{problem_type}")
        generated_files = self._execute_visualization_code(viz_code, f"problem_type")
        
        return {
            "agent_name": "DynamicProblemVisualizer",
            "visualization_type": "problem_type_specific", 
            "code": viz_code,
            "data": analysis_data,
            "file_paths": generated_files,
            "metadata": {"problem_focused": True, "problem_type": problem_type},
            "frontend_instructions": f"Problem-type visualization for {problem_type}"
        }
    
    def _extract_and_enhance_code(self, response: str, viz_type: str) -> str:
        """Extract code and enhance with proper structure"""
        
        # Extract code from response
        code_match = re.search(r'``````', response, re.DOTALL)
        if code_match:
            code = code_match.group(1)
        else:
            code_match = re.search(r'``````', response, re.DOTALL)
            if code_match:
                code = code_match.group(1)
            else:
                code = response
        
        # Enhance code with proper structure
        enhanced_code = f'''
"""
DYNAMIC {viz_type.upper()} VISUALIZATION
Generated by AI based on actual analysis
Created: {datetime.now()}
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import os

# Change to output directory
os.chdir(r"{self.session_dir}")

{code}

# Save the visualization
plt.tight_layout()
filename = '{viz_type}_dynamic.png'
plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Saved dynamic visualization: {{filename}}")
plt.close()
'''
        
        # Save the code
        code_filename = f"{viz_type}_code.py"
        code_path = os.path.join(self.session_dir, code_filename)
        with open(code_path, 'w', encoding='utf-8') as f:
            f.write(enhanced_code)
        
        print(f"💾 Saved visualization code: {code_filename}")
        return enhanced_code
    
    def _execute_visualization_code(self, code: str, viz_type: str) -> List[str]:
        """Execute the generated visualization code"""
        try:
            original_dir = os.getcwd()
            os.chdir(self.session_dir)
            
            # Execute the code
            exec(code)
            
            # Find generated files
            generated_files = []
            for file in os.listdir('.'):
                if file.endswith('.png') and viz_type in file:
                    generated_files.append(file)
            
            print(f"✅ Generated files: {generated_files}")
            return generated_files
            
        except Exception as e:
            print(f"❌ Visualization execution failed: {e}")
            return []
        finally:
            os.chdir(original_dir)
    
    def _format_rag_context(self, context: List[Dict[str, Any]]) -> str:
        """Format RAG context for prompts"""
        if not context:
            return "No specific context available."
        
        formatted = ""
        for i, doc in enumerate(context[:2], 1):
            formatted += f"{i}. {doc['title']}: {doc.get('content', '')[:200]}...\n"
        
        return formatted

# Replace the original orchestrator
dynamic_orchestrator = DynamicVisualizationOrchestrator()
