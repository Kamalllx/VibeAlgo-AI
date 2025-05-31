# backend/api/complexity.py (FIXED)
from flask import Blueprint, request, jsonify
import asyncio
from core.agent_orchestrator import orchestrator

bp = Blueprint('complexity', __name__)

@bp.route('/analyze', methods=['POST'])
def analyze_code_complexity():
    """Enhanced agentic complexity analysis with full visibility - FIXED"""
    
    def run_async_analysis():
        return asyncio.run(orchestrator.process_request("complexity_analysis", {
            "code": request.json["code"],
            "language": request.json.get("language", "python"),
            "platform": request.json.get("platform", "general")
        }))
    
    try:
        print(f"\n🚀 API ENDPOINT: /complexity/analyze")
        print(f"📥 Request received at {request.headers.get('X-Forwarded-For', request.remote_addr)}")
        
        data = request.get_json()
        print(f"📝 Request payload: {len(str(data))} characters")
        
        # Run agentic analysis
        result = run_async_analysis()
        
        # Extract key data for response - FIXED KEY NAMES
        agent_result = result["agent_result"]
        complexity_data = agent_result["complexity_analysis"]
        
        response_data = {
            "success": True,
            "agentic_processing": True,
            "complexity_analysis": {
                "time_complexity": complexity_data.get("time_complexity", "Unknown"),
                "space_complexity": complexity_data.get("space_complexity", "Unknown"),
                "reasoning": complexity_data.get("reasoning", "No reasoning available"),
                "suggestions": complexity_data.get("suggestions", [])
            },
            "ai_processing_details": agent_result["ai_processing"],
            "rag_knowledge": agent_result["enhanced_rag_context"],  # FIXED: Use correct key
            "agent_metadata": {
                "agent_name": agent_result["agent_name"],
                "confidence": agent_result["confidence_score"],
                "processing_steps": agent_result["processing_steps"]
            },
            "full_agent_response": result  # Complete agent response for debugging
        }
        
        print(f"✅ API RESPONSE: Complexity analysis completed")
        print(f"📊 Time Complexity: {complexity_data.get('time_complexity', 'Unknown')}")
        print(f"📊 Space Complexity: {complexity_data.get('space_complexity', 'Unknown')}")
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ API ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False, 
            "error": str(e),
            "agentic_processing": False
        }), 500
