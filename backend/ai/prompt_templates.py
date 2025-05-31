
# backend/ai/prompt_templates.py
from typing import Dict, Any, List

class PromptTemplates:
    """Collection of prompt templates for different AI tasks"""
    
    @staticmethod
    def complexity_analysis_prompt(code: str, language: str = "python") -> str:
        """Generate prompt for complexity analysis"""
        return f"""
You are an expert algorithm analyst. Analyze the following {language} code and provide:

1. Time complexity (Big O notation)
2. Space complexity (Big O notation)  
3. Brief explanation of the analysis
4. Optimization suggestions if applicable

Code to analyze:
{code}
Provide your analysis in a structured format focusing on accuracy and practical insights.
        """.strip()
    
    @staticmethod
    def optimization_prompt(code: str, current_complexity: str, target_complexity: str = None) -> str:
        """Generate prompt for code optimization"""
        target_text = f" to achieve {target_complexity}" if target_complexity else ""
        
        return f"""
You are a coding optimization expert. The following code currently has {current_complexity} complexity.
Suggest specific optimizations{target_text}:

{code}
Provide:
1. Specific optimization techniques
2. Alternative algorithms if applicable
3. Trade-offs to consider
4. Code examples if helpful

Focus on practical, implementable suggestions.
        """.strip()
    
    @staticmethod
    def contest_strategy_prompt(submissions: List[Dict[str, Any]], contest_duration: int = 180) -> str:
        """Generate prompt for contest strategy optimization"""
        submission_summary = "\n".join([
            f"Problem {sub.get('problem_id')}: {sub.get('status')} in {sub.get('time_taken')}min (Score: {sub.get('score')})"
            for sub in submissions[:10]  # Limit to first 10 for brevity
        ])
        
        return f"""
You are a competitive programming strategist. Analyze this contest performance and provide optimization strategies:

Contest Duration: {contest_duration} minutes
Submissions:
{submission_summary}

Provide strategic advice on:
1. Time allocation per problem
2. Problem selection strategy  
3. Areas for improvement
4. Contest-specific tactics

Focus on actionable insights for future contests.
        """.strip()
    
    @staticmethod
    def learning_path_prompt(user_progress: Dict[str, Any], target_skill: str) -> str:
        """Generate prompt for personalized learning path"""
        stats = user_progress.get('statistics', {})
        weak_areas = user_progress.get('skill_gaps', [])
        
        return f"""
You are a DSA learning mentor. Based on this student's progress, create a personalized learning path for {target_skill}:

Current Stats:
- Total Problems: {stats.get('total_problems', 0)}
- Solve Rate: {stats.get('solve_rate', 0):.1f}%
- Weak Areas: {', '.join(weak_areas[:5])}

Target Skill: {target_skill}

Provide:
1. Step-by-step learning plan
2. Recommended practice problems
3. Key concepts to master
4. Timeline estimate
5. Progress milestones

Make it practical and achievable.
        """.strip()
    
    @staticmethod
    def code_review_prompt(code: str, focus_areas: List[str] = None) -> str:
        """Generate prompt for code review"""
        focus_text = f"Pay special attention to: {', '.join(focus_areas)}" if focus_areas else ""
        
        return f"""
You are a senior software engineer conducting a code review. Analyze this code for:

1. Code quality and readability
2. Performance and efficiency
3. Best practices adherence
4. Potential bugs or issues
5. Suggestions for improvement

{focus_text}

Code to review:
{code}
Provide constructive feedback with specific examples and actionable suggestions.
        """.strip()
    
    @staticmethod
    def algorithm_explanation_prompt(algorithm_name: str, user_level: str = "intermediate") -> str:
        """Generate prompt for algorithm explanation"""
        return f"""
You are a computer science teacher explaining algorithms to {user_level} level students.

Explain the {algorithm_name} algorithm with:

1. High-level concept and intuition
2. Step-by-step process
3. Time and space complexity analysis
4. When to use this algorithm
5. Simple example walkthrough
6. Common variations or optimizations

Adjust the explanation depth for {user_level} level understanding.
        """.strip()

# Global templates instance
templates = PromptTemplates()
