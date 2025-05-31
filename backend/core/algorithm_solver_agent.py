
import re
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from ai.groq_client import groq_client
from ai.rag_pipeline import rag_pipeline

@dataclass
class ProblemStatement:
    title: str
    description: str
    input_format: str
    output_format: str
    constraints: List[str]
    examples: List[Dict[str, Any]]
    difficulty: str
    categories: List[str]

@dataclass
class AlgorithmicApproach:
    approach_name: str
    time_complexity: str
    space_complexity: str
    description: str
    steps: List[str]
    pros: List[str]
    cons: List[str]
    use_cases: List[str]

@dataclass
class CodeSolution:
    language: str
    code: str
    explanation: str
    test_cases: List[Dict[str, Any]]
    optimization_notes: List[str]

class AlgorithmSolverAgent:
    def __init__(self):
        self.name = "AlgorithmSolver"
        self.role = "Algorithm Problem Solving Specialist"
        
        print(f"[{self.name}] Algorithm Solver Agent initialized")
    
    async def solve_problem(self, problem_input: str, user_solution: str = None) -> Dict[str, Any]:
        """Complete algorithm problem solving pipeline"""
        print(f"\n{'='*90}")
        print(f"[{self.name}] ALGORITHM PROBLEM SOLVING PIPELINE")
        print(f"{'='*90}")
        
        # Step 1: Parse and understand the problem
        problem_analysis = await self._analyze_problem(problem_input)
        
        # Step 2: Generate optimal algorithmic approaches  
        approaches = await self._generate_approaches(problem_analysis)
        
        # Step 3: Generate code solution
        best_approach = approaches[0] if approaches and isinstance(approaches, list) and len(approaches) > 0 else None
        code_solution = await self._generate_code_solution(problem_analysis, best_approach)
        
        # Step 4: Analyze generated solution if code was extracted successfully
        complexity_analysis = None
        if code_solution.get('code') and len(code_solution['code']) > 50:
            try:
                from core.agent_orchestrator import orchestrator
                complexity_analysis = await orchestrator.process_request("complexity_analysis", {
                    "code": code_solution.get('code', ''),
                    "language": "python"
                })
                print(f"Complexity analysis completed for generated code")
            except Exception as e:
                print(f"Complexity analysis failed: {e}")
                complexity_analysis = {"error": str(e)}
        else:
            print(f"Skipping complexity analysis - no valid code generated")
        
        # Step 5: Compare with user solution if provided
        user_comparison = None
        if user_solution:
            user_comparison = await self._compare_solutions(code_solution, user_solution, problem_analysis)
        
        return {
            "agent_name": self.name,
            "timestamp": datetime.now().isoformat(),
            "problem_analysis": problem_analysis,
            "algorithmic_approaches": approaches,
            "optimal_solution": {
                "code": code_solution,
                "complexity_analysis": complexity_analysis
            },
            "user_solution_comparison": user_comparison,
            "learning_resources": await self._suggest_learning_resources(problem_analysis)
        }
    
    async def _analyze_problem(self, problem_input: str) -> Dict[str, Any]:
        """Analyze and understand the algorithm problem"""
        print(f"\nSTEP 1: PROBLEM ANALYSIS")
        print(f"{'─'*50}")
        print(f"Problem Input:")
        print(f"{'─'*30}")
        print(problem_input[:500] + "..." if len(problem_input) > 500 else problem_input)
        print(f"{'─'*30}")
        
        # RAG retrieval for similar problems
        rag_context = rag_pipeline.retrieve_relevant_context(
            f"algorithm problem solving {problem_input[:200]}"
        )
        
        analysis_prompt = f"""You are an expert algorithm problem analyst. Analyze this problem and return ONLY valid JSON:

Problem: {problem_input}

Return exactly this JSON structure (no other text):
{{
    "problem_type": "sorting|searching|dynamic_programming|graph|array|string|etc",
    "difficulty": "easy|medium|hard",
    "key_concepts": ["concept1", "concept2"],
    "input_description": "description of input",
    "output_description": "description of expected output", 
    "constraints": ["constraint1", "constraint2"],
    "examples": [
        {{"input": "example", "output": "result", "explanation": "why"}}
    ],
    "similar_problems": ["problem1", "problem2"],
    "algorithmic_patterns": ["pattern1", "pattern2"]
}}"""
        
        print(f"Analyzing problem with AI + RAG context...")
        response = groq_client.chat_completion([
            {"role": "system", "content": "You are an algorithm analyst. Return ONLY valid JSON, no other text."},
            {"role": "user", "content": analysis_prompt}
        ])
        
        print(f"\nRAW PROBLEM ANALYSIS:")
        print(f"{'─'*40}")
        print(response.content)
        print(f"{'─'*40}")
        
        try:
            analysis = self._extract_json_from_response(response.content)
            if analysis:
                print(f"Problem analysis successful")
                print(f"Problem Type: {analysis.get('problem_type', 'Unknown')}")
                print(f"Difficulty: {analysis.get('difficulty', 'Unknown')}")
                print(f"Key Concepts: {', '.join(analysis.get('key_concepts', []))}")
                return analysis
            else:
                return self._fallback_problem_analysis(problem_input)
        except Exception as e:
            print(f"JSON parsing failed: {e}")
            return self._fallback_problem_analysis(problem_input)
    
    async def _generate_approaches(self, problem_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate multiple algorithmic approaches"""
        print(f"\nSTEP 2: ALGORITHMIC APPROACH GENERATION")
        print(f"{'─'*50}")
        
        # RAG retrieval for relevant algorithms
        problem_type = problem_analysis.get('problem_type', 'general')
        rag_context = rag_pipeline.retrieve_relevant_context(
            f"{problem_type} algorithms approaches solutions"
        )
        
        approaches_prompt = f"""Based on this problem analysis, generate 2-3 different algorithmic approaches.

Problem Analysis: {json.dumps(problem_analysis, indent=2)}

For each approach, provide a JSON object with:
- name: approach name
- time_complexity: Big O notation
- space_complexity: Big O notation  
- description: brief description
- steps: list of algorithm steps
- pros: list of advantages
- cons: list of disadvantages

Return ONLY a JSON array of approaches, no other text:
[
  {{
    "name": "Approach Name",
    "time_complexity": "O(...)",
    "space_complexity": "O(...)",
    "description": "Brief description",
    "steps": ["step1", "step2"],
    "pros": ["pro1", "pro2"],
    "cons": ["con1", "con2"]
  }}
]"""
        
        print(f"Generating algorithmic approaches...")
        response = groq_client.chat_completion([
            {"role": "system", "content": "You are an algorithm design expert. Return ONLY valid JSON array, no other text."},
            {"role": "user", "content": approaches_prompt}
        ])
        
        print(f"\nRAW APPROACHES GENERATION:")
        print(f"{'─'*40}")
        print(response.content)
        print(f"{'─'*40}")
        
        try:
            approaches = self._extract_json_array_from_response(response.content)
            if approaches:
                print(f"Generated {len(approaches)} algorithmic approaches")
                for i, approach in enumerate(approaches, 1):
                    print(f"   {i}. {approach.get('name', 'Unknown')} - {approach.get('time_complexity', 'Unknown')}")
                return approaches
            else:
                return self._fallback_approaches(problem_analysis)
        except Exception as e:
            print(f"Approaches parsing failed: {e}")
            return self._fallback_approaches(problem_analysis)
    
    async def _generate_code_solution(self, problem_analysis: Dict[str, Any], best_approach: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Python code solution - ENHANCED WITH DEBUG"""
        print(f"\nSTEP 3: CODE SOLUTION GENERATION")
        print(f"{'─'*50}")
        
        code_prompt = f"""Generate a complete Python solution for this problem:

Problem: {problem_analysis.get('input_description', 'Find solution')}
Approach: {best_approach.get('name', 'Optimal approach') if best_approach else 'Direct solution'}

Generate ONLY Python code in this exact format:
```
def solution_function(input_param):
    \"\"\"
    Description: {problem_analysis.get('output_description', 'Solution function')}
    Time Complexity: O(n)
    Space Complexity: O(1)
    \"\"\"
    # Implementation here
    return result

# Test cases
test_cases = [
    {{"input": "test1", "expected": "result1"}},
    {{"input": "test2", "expected": "result2"}}
]
```

Return ONLY the code block above, no other text or explanations."""
        
        print(f"Generating Python code solution...")
        print(f"Code generation prompt:")
        print(f"{'─'*30}")
        print(code_prompt[:300] + "..." if len(code_prompt) > 300 else code_prompt)
        print(f"{'─'*30}")
        
        response = groq_client.chat_completion([
            {"role": "system", "content": "You are a Python code generator. Return ONLY the requested Python code in markdown format, no explanations."},
            {"role": "user", "content": code_prompt}
        ])
        
        print(f"\nRAW CODE GENERATION RESPONSE:")
        print(f"{'─'*40}")
        print(response.content)
        print(f"{'─'*40}")
        
        # ENHANCED: Extract code with better debugging
        generated_code = self._extract_code_from_response(response.content)
        
        print(f"Code extraction result:")
        print(f"   Length: {len(generated_code)} characters")
        print(f"   Lines: {len(generated_code.splitlines())} lines")
        print(f"   Contains 'def': {'def ' in generated_code}")
        
        return {
            "language": "python",
            "code": generated_code,
            "full_response": response.content,
            "explanation": self._extract_explanation(response.content),
            "test_cases": self._extract_test_cases(response.content)
        }
    
    def _extract_code_from_response(self, response: str) -> str:
        """Extract code from LLM response - COMPLETELY FIXED WITH DEBUG"""
        print(f"\nCODE EXTRACTION DEBUG:")
        print(f"{'─'*30}")
        print(f"Input response length: {len(response)} characters")
        
        # Debug: Print first 200 chars to see structure
        print(f"Response preview:")
        print(f"'{response[:200]}...'")
        
        # Pattern 1: Python code blocks with ```
        pattern1 = r'```python\n(.*?)```'
        matches1 = re.findall(pattern1, response, re.DOTALL)
        print(f"Pattern 1 (```python): Found {len(matches1)} matches")
        if matches1:
            code = matches1[0].strip()
            print(f"Extracted code using Pattern 1 (length: {len(code)})")
            print(f"Code preview: {code[:100]}...")
            return code
        
        # Pattern 2: General code blocks with ```
        pattern2 = r'```\n?(.*?)```'
        matches2 = re.findall(pattern2, response, re.DOTALL)
        print(f"Pattern 2 (```): Found {len(matches2)} matches")
        if matches2:
            for i, match in enumerate(matches2):
                if 'def ' in match or 'import ' in match or len(match.strip()) > 50:
                    code = match.strip()
                    print(f"Extracted code using Pattern 2, match {i} (length: {len(code)})")
                    print(f"Code preview: {code[:100]}...")
                    return code
        
        # Pattern 3: Single backticks
        pattern3 = r'`([^`]*def[^`]*)`'
        matches3 = re.findall(pattern3, response, re.DOTALL)
        print(f"Pattern 3 (single `): Found {len(matches3)} matches")
        if matches3:
            code = matches3[0].strip()
            print(f"Extracted code using Pattern 3 (length: {len(code)})")
            return code
        
        # Pattern 4: Look for def functions directly
        def_pattern = r'(def\s+\w+.*?(?=\n\n|\n#|\ndef\s|\nclass\s|\Z))'
        def_matches = re.findall(def_pattern, response, re.DOTALL)
        print(f"Pattern 4 (def function): Found {len(def_matches)} matches")
        if def_matches:
            code = def_matches[0].strip()
            print(f"Extracted code using Pattern 4 (length: {len(code)})")
            return code
        
        # Pattern 5: Extract everything between first 'def' and last return/pass
        lines = response.split('\n')
        code_lines = []
        in_function = False
        
        for line in lines:
            if 'def ' in line and not in_function:
                in_function = True
                code_lines.append(line)
                print(f"Found function start: {line.strip()}")
            elif in_function:
                code_lines.append(line)
                if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                    if not line.strip().startswith('#') and not line.strip().startswith('```'):
                        if 'test_cases' in line or 'Test cases' in line:
                            continue
                        if line.strip() and not any(word in line.lower() for word in ['def ', 'import ', 'test', '#']):
                            break
        
        if code_lines:
            extracted_code = '\n'.join(code_lines).strip()
            print(f"Extracted code using Pattern 5 (length: {len(extracted_code)})")
            print(f"Code preview: {extracted_code[:100]}...")
            return extracted_code
        
        print(f"All extraction patterns failed")
        print(f"Generating fallback code template")
        
        default_code = '''def find_maximum(arr):
    """
    Find the maximum element in an array
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if not arr:
        raise ValueError("Array cannot be empty")
    
    max_element = arr
    for element in arr[1:]:
        if element > max_element:
            max_element = element
    
    return max_element

# Test cases
test_cases = [
    {"input": , "expected": 5},
    {"input": [-1, -2, -3], "expected": -1}
]'''
        
        print(f"Using fallback template (length: {len(default_code)})")
        return default_code
    def _extract_json_from_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from response - FINAL FIX"""
        print(f"JSON EXTRACTION: Input length {len(response)}")
        
        # Clean the response
        cleaned = response.strip()
        
        # Remove markdown code blocks - IMPROVED LOGIC
        if '```json' in cleaned:
            # Extract content between ```json and ```
            start = cleaned.find('```json') + 7
            end = cleaned.find('```', start)
            if end != -1:
                cleaned = cleaned[start:end].strip()
        elif '```' in cleaned:
            # Extract content between ```
            parts = cleaned.split('```')
            if len(parts) >= 3:
                cleaned = parts[1].strip()
            elif len(parts) == 2:
                cleaned = parts[1].strip()
        
        # Try direct parsing
        try:
            result = json.loads(cleaned)
            print(f"JSON extraction successful")
            return result
        except Exception as e:
            print(f"Direct JSON parse failed: {e}")
        
        # Try finding JSON block with regex
        try:
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            json_match = re.search(json_pattern, response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
                print(f"Regex JSON extraction successful")
                return result
        except Exception as e:
            print(f"Regex JSON extraction failed: {e}")
        
        return None

    def _extract_json_array_from_response(self, response: str) -> Optional[List[Dict[str, Any]]]:
        """Extract JSON array from response - FINAL FIX"""
        print(f"ARRAY EXTRACTION: Input length {len(response)}")
        
        # Clean the response
        cleaned = response.strip()
        
        # Remove markdown code blocks
        if '```json' in cleaned:
            start = cleaned.find('```json') + 7
            end = cleaned.find('```', start)
            if end != -1:
                cleaned = cleaned[start:end].strip()
        elif '```' in cleaned:
            parts = cleaned.split('```')
            if len(parts) >= 3:
                cleaned = parts[1].strip()
        
        # Try direct parsing
        try:
            result = json.loads(cleaned)
            if isinstance(result, list):
                print(f"Array extraction successful")
                return result
        except Exception as e:
            print(f"Direct array parse failed: {e}")
        
        # Try finding array with regex
        try:
            array_pattern = r'\[[^\[\]]*(?:\{[^{}]*\}[^\[\]]*)*\]'
            array_match = re.search(array_pattern, response, re.DOTALL)
            if array_match:
                array_str = array_match.group()
                result = json.loads(array_str)
                if isinstance(result, list):
                    print(f"Regex array extraction successful")
                    return result
        except Exception as e:
            print(f"Regex array extraction failed: {e}")
        
        return None
    async def _compare_solutions(self, optimal_solution: Dict[str, Any], user_solution: str, problem_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Compare user solution with optimal solution"""
        print(f"\nSTEP 4: SOLUTION COMPARISON")
        print(f"{'─'*50}")
        
        try:
            from core.agent_orchestrator import orchestrator
            user_complexity = await orchestrator.process_request("complexity_analysis", {
                "code": user_solution,
                "language": "python"
            })
        except Exception as e:
            print(f"User solution analysis failed: {e}")
            user_complexity = {"error": str(e)}
        
        comparison_prompt = f"""Compare these two solutions for the same problem:

Problem: {problem_analysis.get('problem_type', 'Algorithm Problem')}

Optimal Solution:
```
{optimal_solution['code']}
```

User Solution:
```
{user_solution}
```

Provide detailed comparison:
1. Correctness analysis
2. Time/space complexity comparison  
3. Code quality assessment
4. Specific improvements for user solution
5. Learning recommendations

Be constructive and educational."""
        
        print(f"Comparing solutions...")
        response = groq_client.chat_completion([
            {"role": "system", "content": "You are a coding mentor providing constructive feedback on algorithm solutions."},
            {"role": "user", "content": comparison_prompt}
        ])
        
        print(f"\nRAW SOLUTION COMPARISON:")
        print(f"{'─'*40}")
        print(response.content)
        print(f"{'─'*40}")
        
        return {
            "user_complexity_analysis": user_complexity,
            "comparison_report": response.content,
            "improvements_suggested": self._extract_improvements(response.content),
            "learning_areas": self._extract_learning_areas(response.content)
        }
    
    async def _suggest_learning_resources(self, problem_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Suggest learning resources based on problem type"""
        problem_type = problem_analysis.get('problem_type', 'general')
        key_concepts = problem_analysis.get('key_concepts', [])
        
        learning_context = rag_pipeline.retrieve_relevant_context(
            f"learning resources {problem_type} {' '.join(key_concepts)}"
        )
        
        return [
            {
                "type": "documentation",
                "title": f"{problem_type.title()} Algorithms Guide",
                "description": f"Comprehensive guide to {problem_type} algorithms",
                "relevance": "high"
            },
            {
                "type": "practice",
                "title": "Similar Problems",
                "description": f"Practice more {problem_type} problems",
                "relevance": "medium"
            }
        ]
    
    def _format_rag_context(self, context: List[Dict[str, Any]]) -> str:
        """Format RAG context for prompts"""
        if not context:
            return "No specific algorithmic knowledge retrieved."
        
        formatted = ""
        for i, doc in enumerate(context[:3], 1):
            formatted += f"{i}. {doc['title']}\n"
            formatted += f"   {doc.get('content', doc.get('full_content', ''))[:300]}...\n\n"
        
        return formatted
    
    def _fallback_problem_analysis(self, problem_input: str) -> Dict[str, Any]:
        """Fallback problem analysis"""
        problem_lower = problem_input.lower()
        if 'sort' in problem_lower:
            problem_type = 'sorting'
        elif 'search' in problem_lower or 'find' in problem_lower:
            problem_type = 'searching'
        elif 'graph' in problem_lower:
            problem_type = 'graph'
        elif 'array' in problem_lower or 'list' in problem_lower:
            problem_type = 'array'
        else:
            problem_type = 'general'
            
        return {
            "problem_type": problem_type,
            "difficulty": "medium",
            "key_concepts": [problem_type, "algorithm"],
            "input_description": "Problem input data",
            "output_description": "Expected solution output",
            "constraints": ["Standard constraints"],
            "examples": [{"input": "sample", "output": "expected", "explanation": "example"}],
            "similar_problems": [],
            "algorithmic_patterns": [problem_type]
        }
    
    def _fallback_approaches(self, problem_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fallback approaches"""
        problem_type = problem_analysis.get('problem_type', 'general')
        
        return [
            {
                "name": f"Linear {problem_type.title()} Approach",
                "time_complexity": "O(n)",
                "space_complexity": "O(1)",
                "description": f"Simple iterative {problem_type} solution",
                "steps": ["Iterate through data", "Apply logic", "Return result"],
                "pros": ["Simple to implement", "Easy to understand"],
                "cons": ["May not be optimal for large inputs"],
                "use_cases": ["Small to medium datasets"]
            },
            {
                "name": f"Optimized {problem_type.title()} Approach", 
                "time_complexity": "O(n log n)",
                "space_complexity": "O(log n)",
                "description": f"More efficient {problem_type} solution",
                "steps": ["Divide problem", "Solve subproblems", "Combine results"],
                "pros": ["Better performance", "Scalable"],
                "cons": ["More complex implementation"],
                "use_cases": ["Large datasets", "Performance critical applications"]
            }
        ]
    
    def _extract_explanation(self, response: str) -> str:
        """Extract explanation from response"""
        lines = response.split('\n')
        explanation_lines = []
        
        for line in lines:
            if any(word in line.lower() for word in ['explanation', 'description', 'algorithm']):
                explanation_lines.append(line.strip())
        
        return '\n'.join(explanation_lines[:3]) if explanation_lines else "Algorithm solution generated"
    
    def _extract_test_cases(self, response: str) -> List[Dict[str, Any]]:
        """Extract test cases from response"""
        test_pattern = r'test_cases.*?$$(.*?)$$'
        matches = re.findall(test_pattern, response, re.DOTALL | re.IGNORECASE)
        
        if matches:
            try:
                test_str = '[' + matches + ']'
                return json.loads(test_str)
            except:
                pass
        
        return [
            {"input": "test_input", "expected": "expected_output", "description": "Basic test case"}
        ]
    
    def _extract_improvements(self, response: str) -> List[str]:
        """Extract improvement suggestions"""
        lines = response.split('\n')
        improvements = []
        
        for line in lines:
            if any(word in line.lower() for word in ['improve', 'optimize', 'better', 'consider']):
                improvements.append(line.strip())
        
        return improvements[:5]
    
    def _extract_learning_areas(self, response: str) -> List[str]:
        """Extract learning areas"""
        lines = response.split('\n')
        learning_areas = []
        
        for line in lines:
            if any(word in line.lower() for word in ['learn', 'study', 'practice', 'understand']):
                learning_areas.append(line.strip())
        
        return learning_areas[:3]

algorithm_solver = AlgorithmSolverAgent()
