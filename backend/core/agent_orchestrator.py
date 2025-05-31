# backend/core/agent_orchestrator.py (COMPLETE VERSION)
import asyncio
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from ai.groq_client import groq_client
from ai.rag_pipeline import rag_pipeline
from ai.model_context import mcp

@dataclass
class AgentThought:
    agent_name: str
    thought_type: str
    content: str
    timestamp: datetime
    confidence: float

class ComplexityAnalysisAgent:
    def __init__(self):
        self.name = "ComplexityAnalyzer"
        self.role = "Algorithm Complexity Specialist"
        self.thought_chain = []
    
    async def analyze_complexity(self, code: str, language: str = "python") -> Dict[str, Any]:
        print(f"\n{'='*80}")
        print(f"🔍 [{self.name}] STARTING DETAILED COMPLEXITY ANALYSIS")
        print(f"{'='*80}")
        print(f"📝 Input Code ({language}):")
        print(f"{'─'*40}")
        print(code)
        print(f"{'─'*40}")
        print(f"📏 Code Length: {len(code)} characters")
        print(f"📄 Lines of Code: {len(code.splitlines())}")
        
        # Step 1: AI Reasoning Process
        print(f"\n🧠 STEP 1: AI REASONING PROCESS")
        print(f"{'─'*50}")
        
        reasoning_prompt = f"""
You are an expert algorithm analyst. Analyze this {language} code step by step:
Code:
{code}

Think through this systematically:
1. Identify all loops, recursive calls, and operations
2. Determine the input size variable (usually n)
3. Count how operations scale with input size
4. Calculate time complexity
5. Analyze space usage and calculate space complexity
6. Provide clear reasoning for your conclusions

Be specific and show your work.
"""
        
        print(f"🤖 [{self.name}] Sending reasoning prompt to AI...")
        print(f"📤 PROMPT SENT TO LLM:")
        print(f"{reasoning_prompt[:200]}... [truncated for display]")
        
        reasoning_response = groq_client.chat_completion([
            {"role": "system", "content": "You are a world-class algorithm complexity analyst. Be thorough and precise."},
            {"role": "user", "content": reasoning_prompt}
        ])
        
        print(f"\n🎯 RAW LLM REASONING RESPONSE:")
        print(f"{'─'*60}")
        print(reasoning_response.content)
        print(f"{'─'*60}")
        print(f"✅ Tokens Used: {reasoning_response.tokens_used}")
        print(f"🎯 Model: {reasoning_response.model}")
        print(f"✅ Success: {reasoning_response.success}")
        
        # Step 2: Extract Structured Complexity Data
        print(f"\n🔧 STEP 2: EXTRACTING STRUCTURED COMPLEXITY DATA")
        print(f"{'─'*50}")
        
        extraction_prompt = f"""
Based on your previous analysis, extract the exact complexity values:

Previous Analysis:
{reasoning_response.content}

Return ONLY a JSON object with this exact format:
{{
    "time_complexity": "O(...)",
    "space_complexity": "O(...)", 
    "reasoning": "Brief explanation of why",
    "loop_count": number,
    "nested_depth": number,
    "suggestions": ["suggestion1", "suggestion2"]
}}

Be precise with Big O notation.
"""
        
        print(f"🤖 [{self.name}] Extracting structured data...")
        extraction_response = groq_client.chat_completion([
            {"role": "system", "content": "You are a data extraction specialist. Return only valid JSON."},
            {"role": "user", "content": extraction_prompt}
        ])
        
        print(f"\n📊 RAW EXTRACTION RESPONSE:")
        print(f"{'─'*60}")
        print(extraction_response.content)
        print(f"{'─'*60}")
        
        # Parse structured data
        try:
            # Clean the response to extract JSON
            json_match = re.search(r'\{.*\}', extraction_response.content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                structured_data = json.loads(json_str)
                print(f"✅ Successfully parsed structured data")
                print(f"📈 Time Complexity: {structured_data.get('time_complexity', 'Unknown')}")
                print(f"💾 Space Complexity: {structured_data.get('space_complexity', 'Unknown')}")
            else:
                # Fallback parsing
                structured_data = self._fallback_complexity_extraction(reasoning_response.content)
                print(f"⚠️ Used fallback parsing")
        except Exception as e:
            print(f"❌ JSON parsing failed: {str(e)}")
            structured_data = self._fallback_complexity_extraction(reasoning_response.content)
        
        # Step 3: RAG Knowledge Retrieval
        print(f"\n📚 STEP 3: RAG KNOWLEDGE RETRIEVAL")
        print(f"{'─'*50}")
        
        rag_query = f"algorithm complexity analysis {language} {structured_data.get('time_complexity', '')}"
        rag_context = rag_pipeline.retrieve_relevant_context(rag_query, code)
        
        print(f"🔍 RAG Query: '{rag_query}'")
        print(f"📖 Retrieved {len(rag_context)} knowledge pieces:")
        for i, doc in enumerate(rag_context):
            print(f"   {i+1}. {doc['name']} (relevance: {doc['relevance_score']:.2f})")
            print(f"      {doc['data'].get('description', 'No description')[:100]}...")
        
        # Step 4: Final Analysis Compilation
        print(f"\n🎯 STEP 4: COMPILING FINAL ANALYSIS")
        print(f"{'─'*50}")
        
        final_result = {
            "agent_name": self.name,
            "analysis_timestamp": datetime.now().isoformat(),
            "input_metadata": {
                "code_length": len(code),
                "language": language,
                "lines_of_code": len(code.splitlines())
            },
            "ai_processing": {
                "reasoning_response": reasoning_response.content,
                "extraction_response": extraction_response.content,
                "reasoning_tokens": reasoning_response.tokens_used,
                "extraction_tokens": extraction_response.tokens_used,
                "total_tokens": reasoning_response.tokens_used + extraction_response.tokens_used
            },
            "complexity_analysis": structured_data,
            "rag_knowledge": rag_context,
            "confidence_score": 0.9 if reasoning_response.success else 0.3,
            "processing_steps": [
                "AI reasoning analysis",
                "Structured data extraction", 
                "RAG knowledge retrieval",
                "Final compilation"
            ]
        }
        
        print(f"✅ Analysis Complete!")
        print(f"📊 Final Time Complexity: {structured_data.get('time_complexity', 'Unknown')}")
        print(f"📊 Final Space Complexity: {structured_data.get('space_complexity', 'Unknown')}")
        print(f"🎯 Confidence: {final_result['confidence_score']:.1%}")
        print(f"🔢 Total Tokens Used: {final_result['ai_processing']['total_tokens']}")
        
        return final_result
    
    def _fallback_complexity_extraction(self, text: str) -> Dict[str, Any]:
        """Fallback method to extract complexity from text"""
        print(f"🔧 Using fallback complexity extraction...")
        
        # Look for O() patterns
        time_patterns = re.findall(r'O\([^)]+\)', text)
        time_complexity = time_patterns[0] if time_patterns else "O(1)"
        
        # Simple space complexity estimation
        if "recursive" in text.lower() or "recursion" in text.lower():
            space_complexity = "O(n)"
        elif "array" in text.lower() or "list" in text.lower():
            space_complexity = "O(n)"
        else:
            space_complexity = "O(1)"
        
        return {
            "time_complexity": time_complexity,
            "space_complexity": space_complexity,
            "reasoning": "Extracted using pattern matching",
            "loop_count": text.lower().count("loop"),
            "nested_depth": text.lower().count("nested"),
            "suggestions": ["Consider algorithm optimization", "Review data structure choices"]
        }

class DSAProgressAgent:
    def __init__(self):
        self.name = "DSATracker"
        self.role = "Learning Progress Specialist"
    
    async def analyze_progress(self, user_id: str, submission_data: Dict[str, Any]) -> Dict[str, Any]:
        print(f"\n{'='*80}")
        print(f"📊 [{self.name}] ANALYZING LEARNING PROGRESS")
        print(f"{'='*80}")
        print(f"👤 User ID: {user_id}")
        print(f"📝 Submission Data:")
        print(json.dumps(submission_data, indent=2))
        
        # Get historical context
        print(f"\n🔍 RETRIEVING HISTORICAL CONTEXT")
        context = mcp.get_context(user_id, "learning_session", "dsa_progress")
        historical_data = context.data if context else {}
        print(f"📚 Historical entries: {len(historical_data.get('submissions', []))}")
        
        # AI Analysis
        progress_prompt = f"""
Analyze this student's DSA learning progress:

New Submission: {json.dumps(submission_data)}
Historical Data: {json.dumps(historical_data, indent=2)}

Provide detailed analysis:
1. Learning velocity and trends
2. Strength areas and weaknesses  
3. Skill gap identification
4. Personalized recommendations
5. Next study topics
6. Estimated improvement timeline

Be specific and actionable.
"""
        
        print(f"\n🧠 SENDING PROGRESS ANALYSIS TO AI...")
        response = groq_client.chat_completion([
            {"role": "system", "content": "You are an expert learning analytics specialist."},
            {"role": "user", "content": progress_prompt}
        ])
        
        print(f"\n🎯 RAW AI PROGRESS ANALYSIS:")
        print(f"{'─'*60}")
        print(response.content)
        print(f"{'─'*60}")
        
        # Update context
        updated_data = historical_data.copy()
        updated_data['last_submission'] = submission_data
        updated_data['last_analysis'] = response.content
        updated_data['analysis_timestamp'] = datetime.now().isoformat()
        
        mcp.update_context(user_id, "learning_session", "dsa_progress", updated_data)
        
        result = {
            "agent_name": self.name,
            "user_id": user_id,
            "submission_data": submission_data,
            "historical_context": historical_data,
            "ai_analysis": response.content,
            "recommendations": self._extract_recommendations(response.content),
            "next_topics": self._extract_next_topics(response.content),
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        print(f"✅ Progress analysis complete!")
        return result
    
    def _extract_recommendations(self, text: str) -> List[str]:
        # Extract recommendations from AI response
        lines = text.split('\n')
        recommendations = []
        in_rec_section = False
        
        for line in lines:
            if 'recommendation' in line.lower():
                in_rec_section = True
            elif in_rec_section and line.strip():
                if line.strip().startswith(('-', '•', '*', '1.', '2.')):
                    recommendations.append(line.strip())
        
        return recommendations[:5]  # Top 5
    
    def _extract_next_topics(self, text: str) -> List[str]:
        # Extract suggested topics
        topics = []
        if 'dynamic programming' in text.lower():
            topics.append('dynamic_programming')
        if 'graph' in text.lower():
            topics.append('graphs')
        if 'tree' in text.lower():
            topics.append('trees')
        return topics

class AgentOrchestrator:
    def __init__(self):
        self.agents = {
            "complexity": ComplexityAnalysisAgent(),
            "dsa_progress": DSAProgressAgent()
        }
    
    async def process_request(self, request_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        print(f"\n🎭 AGENT ORCHESTRATOR: Processing {request_type}")
        print(f"⏰ Start Time: {datetime.now().strftime('%H:%M:%S')}")
        
        if request_type == "complexity_analysis":
            agent = self.agents["complexity"]
            result = await agent.analyze_complexity(data["code"], data.get("language", "python"))
            
        elif request_type == "dsa_progress":
            agent = self.agents["dsa_progress"] 
            result = await agent.analyze_progress(data["user_id"], data["submission"])
            
        else:
            return {"error": f"Unknown request type: {request_type}"}
        
        orchestrator_result = {
            "request_type": request_type,
            "processing_metadata": {
                "start_time": datetime.now().isoformat(),
                "agent_used": result["agent_name"],
                "success": True,
                "orchestrator_version": "1.0"
            },
            "agent_result": result
        }
        
        print(f"🎯 ORCHESTRATOR: {request_type} completed successfully")
        print(f"⏰ End Time: {datetime.now().strftime('%H:%M:%S')}")
        
        return orchestrator_result

# Global orchestrator
orchestrator = AgentOrchestrator()
