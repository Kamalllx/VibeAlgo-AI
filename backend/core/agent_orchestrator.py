# backend/core/agent_orchestrator.py (COMPLETE CORRECTED VERSION)
import asyncio
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Import after defining classes to avoid circular imports
from ai.groq_client import groq_client

@dataclass
class AgentThought:
    agent_name: str
    thought_type: str
    content: str
    timestamp: datetime
    confidence: float

@dataclass
class AgentAction:
    agent_name: str
    action_type: str
    parameters: Dict[str, Any]
    expected_outcome: str
    timestamp: datetime

class BaseAgent:
    def __init__(self, name: str, role: str, capabilities: List[str]):
        self.name = name
        self.role = role
        self.capabilities = capabilities
        self.memory = []
        self.thought_chain = []
        self.active = True
    
    async def think(self, problem: str, context: Dict[str, Any]) -> AgentThought:
        """Agent reasoning process"""
        print(f"🤖 [{self.name}] THINKING: {problem[:100]}...")
        
        # Build contextual prompt
        prompt = f"""
        You are {self.name}, a {self.role} agent. 
        
        Your capabilities: {', '.join(self.capabilities)}
        
        Problem to solve: {problem}
        
        Context: {json.dumps(context, indent=2)}
        
        Think step by step about how to approach this problem.
        What is your analysis and recommended approach?
        
        Respond in this format:
        ANALYSIS: [Your analysis]
        APPROACH: [Your recommended approach]
        CONFIDENCE: [0.0-1.0]
        NEXT_STEPS: [What should happen next]
        """
        
        response = groq_client.chat_completion([
            {"role": "system", "content": "You are a specialized AI agent that thinks analytically."},
            {"role": "user", "content": prompt}
        ])
        
        thought = AgentThought(
            agent_name=self.name,
            thought_type="reasoning",
            content=response.content,
            timestamp=datetime.now(),
            confidence=0.8
        )
        
        self.thought_chain.append(thought)
        print(f"💭 [{self.name}] THOUGHT: {response.content[:200]}...")
        return thought

class ComplexityAnalysisAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="ComplexityAnalyzer", 
            role="Algorithm Complexity Specialist",
            capabilities=["ast_parsing", "complexity_calculation", "optimization_suggestions"]
        )
        print(f"🔍 [{self.name}] Complexity Analysis Agent initialized")
    
    async def analyze_complexity(self, code: str, language: str = "python") -> Dict[str, Any]:
        """MAIN COMPLEXITY ANALYSIS METHOD - COMPLETE IMPLEMENTATION"""
        print(f"\n{'='*80}")
        print(f"🔍 [{self.name}] STARTING DETAILED COMPLEXITY ANALYSIS")
        print(f"{'='*80}")
        print(f"📝 Input Code ({language}):")
        print(f"{'─'*40}")
        print(code)
        print(f"{'─'*40}")
        print(f"📏 Code Length: {len(code)} characters")
        print(f"📄 Lines of Code: {len(code.splitlines())}")
        
        # Import RAG pipeline here to avoid circular imports
        try:
            from ai.rag_pipeline import rag_pipeline
            rag_available = True
        except ImportError as e:
            print(f"⚠️ RAG pipeline not available: {e}")
            rag_available = False
            rag_context = []
        
        # Step 1: Enhanced RAG Knowledge Retrieval
        if rag_available:
            print(f"\n📚 STEP 1: ENHANCED RAG KNOWLEDGE RETRIEVAL")
            print(f"{'─'*50}")
            
            rag_query = f"algorithm complexity analysis {language} time space complexity"
            rag_context = rag_pipeline.retrieve_relevant_context(rag_query, code)
        else:
            rag_context = []
        
        # Step 2: AI Reasoning with or without RAG Context
        print(f"\n🧠 STEP 2: AI REASONING PROCESS")
        print(f"{'─'*50}")
        
        if rag_available and rag_context:
            enhanced_prompt = rag_pipeline.generate_enhanced_prompt(
                f"Analyze the time and space complexity of this {language} code step by step:",
                rag_context
            )
            full_prompt = f"{enhanced_prompt}\n\nCode to analyze:\n{code}"
        else:
            full_prompt = f"""
Analyze the time and space complexity of this {language} code step by step:

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
        
        print(f"🤖 [{self.name}] Sending prompt to AI...")
        reasoning_response = groq_client.chat_completion([
            {"role": "system", "content": "You are a world-class algorithm complexity analyst. Analyze the provided code thoroughly."},
            {"role": "user", "content": full_prompt}
        ])
        
        print(f"\n🎯 RAW LLM REASONING RESPONSE:")
        print(f"{'─'*60}")
        print(reasoning_response.content)
        print(f"{'─'*60}")
        print(f"✅ Tokens Used: {reasoning_response.tokens_used}")
        print(f"🎯 Model: {reasoning_response.model}")
        print(f"✅ Success: {reasoning_response.success}")
        
        # Step 3: Extract Structured Complexity Data
        print(f"\n🔧 STEP 3: EXTRACTING STRUCTURED COMPLEXITY DATA")
        print(f"{'─'*50}")
        
        extraction_prompt = f"""
Based on your analysis of this code:

```
{code}
```

Your analysis:
{reasoning_response.content}

Extract the complexity information in EXACTLY this JSON format (no other text):
{{
    "time_complexity": "O(...)",
    "space_complexity": "O(...)", 
    "reasoning": "Brief explanation",
    "loop_count": number,
    "nested_depth": number,
    "suggestions": ["suggestion1", "suggestion2"]
}}
"""
        
        print(f"🤖 [{self.name}] Extracting structured data...")
        extraction_response = groq_client.chat_completion([
            {"role": "system", "content": "You are a data extraction specialist. Return ONLY valid JSON, no other text."},
            {"role": "user", "content": extraction_prompt}
        ])
        
        print(f"\n📊 RAW EXTRACTION RESPONSE:")
        print(f"{'─'*60}")
        print(extraction_response.content)
        print(f"{'─'*60}")
        
        # Parse structured data
        complexity_data = self._extract_complexity_from_response(extraction_response.content)
        
        # Step 4: Learn from this interaction (if RAG available)
        if rag_available and rag_context and reasoning_response.success:
            try:
                retrieved_doc_ids = [doc['id'] for doc in rag_context]
                quality_score = 4.5 if reasoning_response.tokens_used > 200 else 3.5
                
                rag_pipeline.learn_from_feedback(
                    rag_query, 
                    retrieved_doc_ids, 
                    reasoning_response.content,
                    quality_score
                )
                print(f"🧠 RAG learning update completed")
            except Exception as e:
                print(f"⚠️ RAG learning failed: {e}")
        
        # Step 5: Final Analysis Compilation
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
            "enhanced_rag_context": rag_context,
            "ai_processing": {
                "reasoning_response": reasoning_response.content,
                "extraction_response": extraction_response.content,
                "reasoning_tokens": reasoning_response.tokens_used,
                "extraction_tokens": extraction_response.tokens_used,
                "total_tokens": reasoning_response.tokens_used + extraction_response.tokens_used,
                "rag_enhanced": len(rag_context) > 0
            },
            "complexity_analysis": complexity_data,
            "confidence_score": 0.9 if reasoning_response.success else 0.3,
            "processing_steps": [
                "Enhanced RAG knowledge retrieval" if rag_available else "Direct analysis",
                "AI reasoning with context",
                "Structured data extraction", 
                "Learning feedback" if rag_available else "Static analysis",
                "Final compilation"
            ]
        }
        
        if rag_available:
            try:
                final_result["rag_stats"] = rag_pipeline.get_stats()
            except:
                pass
        
        print(f"✅ Analysis complete!")
        print(f"📊 Final Time Complexity: {complexity_data.get('time_complexity', 'Unknown')}")
        print(f"📊 Final Space Complexity: {complexity_data.get('space_complexity', 'Unknown')}")
        print(f"🎯 Confidence: {final_result['confidence_score']:.1%}")
        print(f"🔢 Total Tokens Used: {final_result['ai_processing']['total_tokens']}")
        
        return final_result
    
    def _extract_complexity_from_response(self, text: str) -> Dict[str, Any]:
        """Extract complexity from LLM response - COMPLETE METHOD"""
        print(f"🔧 Extracting complexity from response...")
        
        try:
            # Try to extract JSON first
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                
                # Fix common JSON issues
                json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)  # Remove trailing commas
                json_str = re.sub(r'(\w+):', r'"\1":', json_str)    # Quote unquoted keys
                
                structured_data = json.loads(json_str)
                print(f"✅ Successfully parsed JSON complexity data")
                return structured_data
        except Exception as e:
            print(f"❌ JSON parsing failed: {e}")
        
        # Fallback: regex pattern matching
        print(f"🔄 Using fallback pattern matching...")
        
        # Look for O() patterns
        time_patterns = re.findall(r'O\([^)]+\)', text)
        time_complexity = time_patterns[0] if time_patterns else "O(1)"
        
        # Look for space complexity mentions
        space_complexity = "O(1)"
        if "space complexity" in text.lower():
            space_matches = re.findall(r'space.*?O\([^)]+\)', text, re.IGNORECASE)
            if space_matches:
                space_pattern = re.search(r'O\([^)]+\)', space_matches[0])
                if space_pattern:
                    space_complexity = space_pattern.group()
        
        # Extract suggestions
        suggestions = []
        if "optimize" in text.lower():
            suggestions.append("Consider algorithm optimization")
        if "efficient" in text.lower():
            suggestions.append("Look for more efficient approaches")
        if "sort" in text.lower():
            suggestions.append("Consider using efficient sorting algorithms")
        
        fallback_data = {
            "time_complexity": time_complexity,
            "space_complexity": space_complexity,
            "reasoning": "Extracted using pattern matching from AI response",
            "loop_count": text.lower().count("loop"),
            "nested_depth": text.lower().count("nested"),
            "suggestions": suggestions or ["Review algorithm efficiency", "Consider data structure optimizations"]
        }
        
        print(f"✅ Fallback extraction completed")
        print(f"📈 Time Complexity: {fallback_data['time_complexity']}")
        print(f"💾 Space Complexity: {fallback_data['space_complexity']}")
        
        return fallback_data

class DSAProgressAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="DSATracker",
            role="Learning Progress Specialist", 
            capabilities=["progress_analysis", "skill_gap_identification", "learning_path_generation"]
        )
        print(f"📊 [{self.name}] DSA Progress Agent initialized")
    
    async def analyze_progress(self, user_id: str, submission_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze DSA learning progress"""
        print(f"\n📊 [{self.name}] ANALYZING LEARNING PROGRESS")
        print(f"👤 User ID: {user_id}")
        
        # Simple progress analysis for now
        return {
            "agent_name": self.name,
            "user_id": user_id,
            "analysis": "Progress analysis completed",
            "timestamp": datetime.now().isoformat()
        }

class ContestStrategyAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="ContestOptimizer",
            role="Competitive Programming Strategist",
            capabilities=["performance_analysis", "strategy_optimization", "time_management"]
        )
        print(f"🏆 [{self.name}] Contest Strategy Agent initialized")
    
    async def optimize_strategy(self, contest_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize contest strategy"""
        print(f"\n🏆 [{self.name}] Optimizing contest strategy...")
        
        return {
            "agent_name": self.name,
            "contest_optimization": "Strategy optimization completed",
            "timestamp": datetime.now().isoformat()
        }

class AgentOrchestrator:
    def __init__(self):
        print(f"🎭 Initializing Agent Orchestrator...")
        
        # Initialize agents
        self.agents = {
            "complexity": ComplexityAnalysisAgent(),
            "dsa_progress": DSAProgressAgent(), 
            "contest_strategy": ContestStrategyAgent()
        }
        self.active_sessions = {}
        
        print(f"✅ Agent Orchestrator initialized with {len(self.agents)} agents")
    
    async def process_request(self, request_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process requests to appropriate agents"""
        print(f"\n🎭 AGENT ORCHESTRATOR: Processing {request_type} request")
        print(f"📥 Input data: {json.dumps(data, indent=2)[:200]}...")
        print(f"⏰ Start Time: {datetime.now().strftime('%H:%M:%S')}")
        
        try:
            if request_type == "complexity_analysis":
                agent = self.agents["complexity"]
                print(f"🔍 Using agent: {agent.name}")
                result = await agent.analyze_complexity(data["code"], data.get("language", "python"))
                
            elif request_type == "dsa_progress":
                agent = self.agents["dsa_progress"]
                result = await agent.analyze_progress(data["user_id"], data["submission"])
                
            elif request_type == "contest_optimization":
                agent = self.agents["contest_strategy"]
                result = await agent.optimize_strategy(data)
                
            else:
                return {"error": f"Unknown request type: {request_type}"}
            
            orchestrator_result = {
                "request_type": request_type,
                "processing_metadata": {
                    "start_time": datetime.now().isoformat(),
                    "agent_used": result["agent_name"],
                    "success": True,
                    "orchestrator_version": "2.0"
                },
                "agent_result": result
            }
            
            print(f"🎯 ORCHESTRATOR: {request_type} completed successfully")
            print(f"⏰ End Time: {datetime.now().strftime('%H:%M:%S')}")
            
            return orchestrator_result
            
        except Exception as e:
            print(f"❌ ORCHESTRATOR ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return {
                "request_type": request_type,
                "processing_metadata": {
                    "start_time": datetime.now().isoformat(),
                    "success": False,
                    "error": str(e)
                },
                "agent_result": {"error": str(e)}
            }

# Global orchestrator instance
print(f"🚀 Creating global agent orchestrator...")
orchestrator = AgentOrchestrator()
print(f"✅ Global agent orchestrator ready!")