# backend/api/visualization.py
from flask import Blueprint, request, jsonify
import asyncio
import json
from visualization.visualization_orchestrator import visualization_orchestrator, VisualizationRequest

bp = Blueprint('visualization', __name__)

@bp.route('/generate', methods=['POST'])
def generate_visualizations():
    """Generate comprehensive visualizations using AI agents"""
    
    def run_visualization():
        return asyncio.run(create_visualizations())
    
    async def create_visualizations():
        data = request.get_json()
        
        # Create visualization request
        viz_request = VisualizationRequest(
            scenario_type=data.get("scenario_type", "complexity_analysis"),
            data=data.get("data", {}),
            requirements=data.get("requirements", ["educational", "interactive"]),
            output_format=data.get("output_format", "interactive"),
            target_platform=data.get("target_platform", "web")
        )
        
        # Generate visualizations using orchestrator
        results = await visualization_orchestrator.create_visualizations(viz_request)
        
        return results
    
    try:
        print(f"\n🎨 API ENDPOINT: /visualization/generate")
        print(f"📊 Request received for visualization generation")
        
        results = run_visualization()
        
        # Format response
        response_data = {
            "success": True,
            "visualizations_generated": len(results),
            "results": [
                {
                    "agent": result.agent_name,
                    "type": result.visualization_type,
                    "files": result.file_paths,
                    "metadata": result.metadata,
                    "frontend_instructions": result.frontend_instructions
                } for result in results
            ],
            "integration_guide": _get_integration_guide()
        }
        
        print(f"✅ Generated {len(results)} visualizations successfully")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ Visualization generation error: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@bp.route('/complexity-viz', methods=['POST'])
def generate_complexity_visualization():
    """Generate complexity-specific visualizations"""
    
    def run_complexity_viz():
        return asyncio.run(create_complexity_viz())
    
    async def create_complexity_viz():
        data = request.get_json()
        
        viz_request = VisualizationRequest(
            scenario_type="complexity_analysis",
            data=data,
            requirements=["big_o_comparison", "growth_curves", "interactive"],
            output_format="interactive",
            target_platform="web"
        )
        
        results = await visualization_orchestrator.create_visualizations(viz_request)
        return results
    
    try:
        results = run_complexity_viz()
        
        return jsonify({
            "success": True,
            "complexity_visualizations": [r.metadata for r in results],
            "files_generated": [f for result in results for f in result.file_paths]
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@bp.route('/algorithm-animation', methods=['POST'])
def generate_algorithm_animation():
    """Generate algorithm animation visualizations"""
    
    def run_algorithm_animation():
        return asyncio.run(create_algorithm_animation())
    
    async def create_algorithm_animation():
        data = request.get_json()
        
        viz_request = VisualizationRequest(
            scenario_type="algorithm_execution",
            data=data,
            requirements=["step_by_step", "animation", "educational"],
            output_format="animated",
            target_platform="web"
        )
        
        results = await visualization_orchestrator.create_visualizations(viz_request)
        return results
    
    try:
        results = run_algorithm_animation()
        
        return jsonify({
            "success": True,
            "animations_created": len(results),
            "animation_files": [f for result in results for f in result.file_paths]
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

def _get_integration_guide():
    """Get comprehensive integration guide"""
    return {
        "web_integration": {
            "static_images": "Use <img> tags with responsive CSS",
            "interactive_charts": "Embed HTML or use Plotly.js/D3.js",
            "animations": "Use HTML5 video or CSS animations"
        },
        "mobile_optimization": {
            "responsive_design": "CSS media queries and flexible layouts",
            "touch_interactions": "Touch-friendly controls and gestures",
            "performance": "Lazy loading and progressive enhancement"
        },
        "accessibility": {
            "screen_readers": "Alt text and ARIA labels",
            "keyboard_navigation": "Tab navigation and shortcuts",
            "color_blind_support": "Color-blind friendly palettes"
        }
    }
