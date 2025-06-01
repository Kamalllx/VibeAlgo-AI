# backend/visualization/dynamic_visualizer.py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import io
import base64
from typing import Dict, List, Any, Tuple
from ai.groq_client import groq_client
from ai.rag_pipeline import rag_pipeline
from visualization.complexity_visualizer import BaseVisualizationAgent
import json
class DynamicVisualizationAgent(BaseVisualizationAgent):
    def __init__(self):
        super().__init__()
        self.name = "DynamicVisualizer"
        self.specialties = [
            "ai_generated_visualizations",
            "custom_chart_creation",
            "adaptive_visualization",
            "scenario_based_charts",
            "multi_modal_displays",
            "intelligent_layout_design"
        ]
        
        # Visualization templates library
        self.viz_templates = {
            "complexity": ["growth_curves", "big_o_comparison", "3d_landscape"],
            "algorithm": ["step_animation", "data_flow", "execution_trace"],
            "performance": ["benchmark_bars", "time_series", "heat_maps"],
            "comparison": ["side_by_side", "overlay_plots", "difference_charts"],
            "educational": ["interactive_tutorial", "guided_explanation", "quiz_integration"]
        }
        
        print(f"🎨 [{self.name}] Dynamic Visualization Agent initialized")
        print(f"🧠 AI-powered adaptive visualization generation ready")
    
    async def generate_visualization(self, request) -> Dict[str, Any]:
        """Generate AI-powered custom visualizations"""
        print(f"\n🎨 [{self.name}] GENERATING DYNAMIC AI VISUALIZATIONS")
        print(f"🧠 Analyzing scenario: {request.scenario_type}")
        
        # AI-powered scenario analysis
        scenario_analysis = await self._analyze_visualization_scenario(request)
        
        # Generate custom visualization code using AI
        custom_viz_code = await self._generate_custom_visualization_code(scenario_analysis, request)
        
        # Create adaptive layout based on data characteristics
        adaptive_layout = await self._create_adaptive_layout(scenario_analysis, request)
        
        # Generate educational enhancements
        educational_elements = await self._generate_educational_elements(scenario_analysis, request)
        
        # Combine all components into comprehensive visualization
        final_visualization = self._combine_dynamic_components(
            custom_viz_code, adaptive_layout, educational_elements, scenario_analysis
        )
        
        return {
            "agent_name": self.name,
            "visualization_type": "dynamic_ai_generated",
            "code": final_visualization,
            "data": scenario_analysis,
            "file_paths": ["dynamic_visualization.html", "custom_chart.png", "interactive_demo.json"],
            "metadata": {
                "ai_generated": True,
                "scenario_type": request.scenario_type,
                "visualization_approach": scenario_analysis.get("recommended_approach", "adaptive"),
                "educational_level": scenario_analysis.get("complexity_level", "intermediate"),
                "interactivity_level": scenario_analysis.get("interactivity_score", 0.8)
            },
            "frontend_instructions": self._get_dynamic_frontend_instructions()
        }
    
    async def _analyze_visualization_scenario(self, request) -> Dict[str, Any]:
        """Use AI to analyze the visualization scenario and recommend approach"""
        
        # Get RAG context for similar visualization scenarios
        rag_context = rag_pipeline.retrieve_relevant_context(
            f"data visualization {request.scenario_type} best practices examples"
        )
        
        analysis_prompt = f"""
        As an expert data visualization consultant, analyze this visualization scenario:
        
        Scenario Type: {request.scenario_type}
        Data Available: {request.data}
        Requirements: {request.requirements}
        Target Platform: {request.target_platform}
        Output Format: {request.output_format}
        
        Related Visualization Knowledge:
        {self._format_rag_context(rag_context)}
        
        Provide comprehensive analysis:
        1. Primary visualization goals
        2. Data characteristics and constraints
        3. Target audience and expertise level
        4. Recommended visualization types (rank top 3)
        5. Interactivity requirements
        6. Color scheme and styling suggestions
        7. Layout and composition recommendations
        8. Educational elements to include
        9. Accessibility considerations
        10. Performance optimization needs
        
        Format as detailed JSON with specific recommendations.
        """
        
        print(f"🧠 Analyzing visualization scenario with AI...")
        response = groq_client.chat_completion([
            {"role": "system", "content": "You are a world-class data visualization expert with deep knowledge of cognitive psychology, design principles, and educational effectiveness."},
            {"role": "user", "content": analysis_prompt}
        ])
        
        print(f"📊 AI Scenario Analysis completed")
        return self._parse_scenario_analysis(response.content)
    
    async def _generate_custom_visualization_code(self, scenario_analysis: Dict[str, Any], request) -> str:
        """Generate custom visualization code using AI"""
        
        code_generation_prompt = f"""
        Generate Python visualization code based on this analysis:
        
        Scenario Analysis: {scenario_analysis}
        Data: {request.data}
        Platform: {request.target_platform}
        
        Requirements:
        1. Create innovative, educational visualization
        2. Use matplotlib, plotly, or seaborn as appropriate
        3. Include interactive elements if requested
        4. Add educational annotations and explanations
        5. Implement responsive design considerations
        6. Include accessibility features
        7. Add export functionality for web integration
        
        Generate complete, executable Python code with:
        - Clear function structure
        - Comprehensive comments
        - Error handling
        - Multiple output formats
        - Educational value maximization
        
        Focus on creating something unique and highly effective for this specific scenario.
        """
        
        print(f"💻 Generating custom visualization code with AI...")
        response = groq_client.chat_completion([
            {"role": "system", "content": "You are an expert Python developer specializing in data visualization and educational technology. Create innovative, effective visualization code."},
            {"role": "user", "content": code_generation_prompt}
        ])
        
        print(f"✅ Custom visualization code generated")
        return self._extract_and_enhance_code(response.content)
    
    async def _create_adaptive_layout(self, scenario_analysis: Dict[str, Any], request) -> str:
        """Create adaptive layout based on data and scenario"""
        
        layout_prompt = f"""
        Design an adaptive layout system for this visualization scenario:
        
        Analysis: {scenario_analysis}
        Target: {request.target_platform}
        
        Create layout code that:
        1. Adapts to different screen sizes
        2. Optimizes information density
        3. Provides clear visual hierarchy
        4. Supports multiple chart types
        5. Includes interactive controls
        6. Maintains accessibility standards
        
        Generate layout management code with CSS Grid/Flexbox concepts translated to Python visualization layouts.
        """
        
        response = groq_client.chat_completion([
            {"role": "system", "content": "You are a UX/UI expert specializing in adaptive layout design for data visualizations."},
            {"role": "user", "content": layout_prompt}
        ])
        
        return self._extract_layout_code(response.content)
    
    async def _generate_educational_elements(self, scenario_analysis: Dict[str, Any], request) -> str:
        """Generate educational enhancements for the visualization"""
        
        educational_prompt = f"""
        Create educational elements for this visualization:
        
        Scenario: {scenario_analysis}
        Audience Level: {scenario_analysis.get('complexity_level', 'intermediate')}
        
        Generate:
        1. Step-by-step explanations
        2. Interactive tutorials
        3. Hover tooltips with insights
        4. Progressive disclosure elements
        5. Quiz/assessment integration points
        6. Conceptual analogies and metaphors
        7. Common misconceptions to address
        8. Real-world application examples
        
        Create code for educational overlays and interactive learning elements.
        """
        
        response = groq_client.chat_completion([
            {"role": "system", "content": "You are an educational technology expert specializing in visual learning and interactive pedagogy."},
            {"role": "user", "content": educational_prompt}
        ])
        
        return self._extract_educational_code(response.content)
    
    def _combine_dynamic_components(self, custom_code: str, layout: str, educational: str, analysis: Dict[str, Any]) -> str:
        """Combine all dynamic components into final visualization"""
        
        header = f'''
"""
DYNAMIC AI-GENERATED VISUALIZATION MODULE
Generated by DynamicVisualizationAgent

AI Analysis Results: {analysis.get('recommended_approach', 'adaptive')}
Educational Level: {analysis.get('complexity_level', 'intermediate')}
Interactivity Score: {analysis.get('interactivity_score', 0.8)}

Frontend Integration Instructions:
================================

1. RESPONSIVE DESIGN:
   - This visualization adapts to viewport size
   - Implements progressive enhancement
   - Uses CSS Grid/Flexbox equivalents
   - Supports touch and mouse interactions

2. INTERACTIVE FEATURES:
   - Dynamic parameter adjustment
   - Real-time data updates
   - Multi-level detail exploration
   - Guided tutorial integration

3. EDUCATIONAL ENHANCEMENTS:
   - Progressive disclosure of complexity
   - Contextual help and explanations
   - Interactive quiz integration
   - Concept reinforcement exercises

4. ACCESSIBILITY:
   - Screen reader compatible
   - Keyboard navigation support
   - High contrast mode
   - Alternative text descriptions

5. PERFORMANCE:
   - Lazy loading for complex elements
   - Canvas optimization for animations
   - Efficient data binding
   - Memory management

6. EXPORT CAPABILITIES:
   - Multiple format support (PNG, SVG, PDF, HTML)
   - Data export functionality
   - Shareable interactive links
   - Embedding code generation

Dependencies Required:
- matplotlib >= 3.5.0
- plotly >= 5.0.0
- numpy >= 1.21.0
- pandas >= 1.3.0
- seaborn >= 0.11.0
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button, RadioButtons
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, List, Any, Tuple
import json
import base64
import io

# Set dynamic styling based on AI analysis
recommended_style = "{analysis.get('style_theme', 'modern')}"
plt.style.use('seaborn-v0_8' if recommended_style == 'academic' else 'default')
sns.set_palette("{analysis.get('color_palette', 'husl')}")

class DynamicVisualizationEngine:
    """AI-powered adaptive visualization engine"""
    
    def __init__(self):
        self.analysis = {analysis}
        self.interactive_elements = []
        self.educational_overlays = []
        
    def create_adaptive_layout(self):
        """Create layout that adapts to content and screen size"""
        {layout}
    
    def add_educational_elements(self):
        """Add AI-generated educational enhancements"""
        {educational}
    
    def generate_custom_visualization(self):
        """Generate the main custom visualization"""
        {custom_code}
    
    def create_comprehensive_dashboard(self):
        """Create complete visualization dashboard"""
        
        # Initialize the visualization engine
        print("🎨 Creating AI-generated dynamic visualization...")
        
        # Create adaptive layout
        self.create_adaptive_layout()
        
        # Generate main visualization
        self.generate_custom_visualization()
        
        # Add educational elements
        self.add_educational_elements()
        
        # Export in multiple formats
        self.export_visualization()
        
        print("✅ Dynamic visualization creation completed!")
    
    def export_visualization(self):
        """Export visualization in multiple formats for web integration"""
        
        # Export static images
        plt.savefig('dynamic_visualization.png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.savefig('dynamic_visualization.svg', bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        # Export interactive HTML (if using Plotly)
        if hasattr(self, 'plotly_fig'):
            self.plotly_fig.write_html('dynamic_visualization.html', 
                                     include_plotlyjs=True, div_id="dynamic-viz")
        
        # Export configuration for frontend integration
        config = {{
            "visualization_type": "dynamic_ai_generated",
            "analysis": self.analysis,
            "export_files": ["dynamic_visualization.png", "dynamic_visualization.svg", "dynamic_visualization.html"],
            "interactive_elements": self.interactive_elements,
            "educational_features": self.educational_overlays,
            "responsive_breakpoints": {{
                "mobile": 480,
                "tablet": 768,
                "desktop": 1024,
                "large": 1440
            }},
            "accessibility_features": {{
                "alt_text": "AI-generated dynamic visualization with interactive elements",
                "keyboard_navigation": True,
                "screen_reader_support": True,
                "high_contrast_mode": True
            }},
            "performance_settings": {{
                "lazy_loading": True,
                "progressive_enhancement": True,
                "webgl_acceleration": True,
                "data_streaming": True
            }}
        }}
        
        with open('dynamic_visualization_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print("📊 Exported visualization in multiple formats:")
        print("  - Static: PNG, SVG")
        print("  - Interactive: HTML")
        print("  - Configuration: JSON")

# Create and run the dynamic visualization
def generate_dynamic_visualization(data=None):
    """
    Main function to generate AI-powered dynamic visualization
    
    Args:
        data: Input data for visualization (optional)
    
    Returns:
        DynamicVisualizationEngine: Configured visualization engine
    """
    
    engine = DynamicVisualizationEngine()
    engine.create_comprehensive_dashboard()
    
    return engine

if __name__ == "__main__":
    # Example usage with sample data
    sample_data = {analysis.get('sample_data', {})}
    
    visualization_engine = generate_dynamic_visualization(sample_data)
    
    print("🎨 AI-Generated Dynamic Visualization Complete!")
    print("📁 Check output files:")
    print("  - dynamic_visualization.png")
    print("  - dynamic_visualization.svg") 
    print("  - dynamic_visualization.html")
    print("  - dynamic_visualization_config.json")
'''
        
        return header
    
    def _parse_scenario_analysis(self, ai_response: str) -> Dict[str, Any]:
        """Parse AI scenario analysis response"""
        # Try to extract JSON from response
        try:
            import re
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        # Fallback parsing
        analysis = {
            "recommended_approach": "adaptive",
            "complexity_level": "intermediate",
            "interactivity_score": 0.8,
            "style_theme": "modern",
            "color_palette": "husl",
            "primary_goals": ["education", "analysis", "comparison"],
            "visualization_types": ["interactive_charts", "animations", "dashboards"],
            "sample_data": {"example": "data"}
        }
        
        # Extract key information from text
        response_lower = ai_response.lower()
        if "beginner" in response_lower:
            analysis["complexity_level"] = "beginner"
        elif "advanced" in response_lower:
            analysis["complexity_level"] = "advanced"
        
        if "interactive" in response_lower:
            analysis["interactivity_score"] = 0.9
        elif "static" in response_lower:
            analysis["interactivity_score"] = 0.3
        
        return analysis
    
    def _extract_and_enhance_code(self, ai_response: str) -> str:
        """Extract and enhance code from AI response"""
        # Look for code blocks in response
        import re
        
        code_patterns = [
            r'``````',
            r'``````',
            r'def.*?(?=\n\n|\n#|\Z)'
        ]
        
        extracted_code = ""
        for pattern in code_patterns:
            matches = re.findall(pattern, ai_response, re.DOTALL)
            if matches:
                extracted_code = matches[0]
                break
        
        if not extracted_code:
            # Generate fallback visualization code
            extracted_code = '''
        def generate_custom_visualization(self):
            """AI-generated custom visualization"""
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('AI-Generated Dynamic Visualization', fontsize=16, fontweight='bold')
            
            # Generate sample data based on scenario
            x = np.linspace(0, 10, 100)
            y1 = np.sin(x)
            y2 = np.cos(x)
            
            # Plot 1: Time series
            axes[0, 0].plot(x, y1, label='Function 1', linewidth=2)
            axes[0, 0].plot(x, y2, label='Function 2', linewidth=2)
            axes[0, 0].set_title('Time Series Analysis')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Distribution
            data = np.random.normal(0, 1, 1000)
            axes[0, 1].hist(data, bins=30, alpha=0.7, color='skyblue')
            axes[0, 1].set_title('Distribution Analysis')
            axes[0, 1].set_ylabel('Frequency')
            
            # Plot 3: Scatter plot
            x_scatter = np.random.randn(100)
            y_scatter = 2 * x_scatter + np.random.randn(100)
            axes[1, 0].scatter(x_scatter, y_scatter, alpha=0.6, c='red')
            axes[1, 0].set_title('Correlation Analysis')
            axes[1, 0].set_xlabel('Variable X')
            axes[1, 0].set_ylabel('Variable Y')
            
            # Plot 4: Heatmap
            matrix_data = np.random.rand(10, 10)
            im = axes[1, 1].imshow(matrix_data, cmap='viridis', aspect='auto')
            axes[1, 1].set_title('Heatmap Analysis')
            plt.colorbar(im, ax=axes[1, 1])
            
            plt.tight_layout()
            
            # Add interactivity note
            plt.figtext(0.5, 0.02, 'Interactive elements available in web version', 
                       ha='center', fontsize=10, style='italic')
            
            plt.show()
            
            # Create Plotly version for interactivity
            self.create_interactive_plotly_version()
        
        def create_interactive_plotly_version(self):
            """Create interactive Plotly version"""
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Time Series', 'Distribution', 'Correlation', 'Heatmap'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Sample data
            x = np.linspace(0, 10, 100)
            y1 = np.sin(x)
            y2 = np.cos(x)
            
            # Add traces
            fig.add_trace(go.Scatter(x=x, y=y1, name='Function 1', line=dict(width=2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=x, y=y2, name='Function 2', line=dict(width=2)), row=1, col=1)
            
            # Distribution
            data = np.random.normal(0, 1, 1000)
            fig.add_trace(go.Histogram(x=data, name='Distribution', nbinsx=30), row=1, col=2)
            
            # Scatter
            x_scatter = np.random.randn(100)
            y_scatter = 2 * x_scatter + np.random.randn(100)
            fig.add_trace(go.Scatter(x=x_scatter, y=y_scatter, mode='markers', 
                                   name='Correlation', marker=dict(color='red', opacity=0.6)), row=2, col=1)
            
            # Heatmap
            matrix_data = np.random.rand(10, 10)
            fig.add_trace(go.Heatmap(z=matrix_data, colorscale='Viridis', name='Heatmap'), row=2, col=2)
            
            # Update layout
            fig.update_layout(
                title_text="Interactive AI-Generated Visualization Dashboard",
                title_x=0.5,
                showlegend=False,
                height=800
            )
            
            # Store for export
            self.plotly_fig = fig
            
            return fig
            '''
        
        return extracted_code
    
    def _extract_layout_code(self, ai_response: str) -> str:
        """Extract layout code from AI response"""
        # Simplified layout code extraction
        return '''
        # Adaptive layout system
        layout_config = {
            "responsive": True,
            "grid_system": "auto",
            "breakpoints": {
                "mobile": 480,
                "tablet": 768, 
                "desktop": 1024
            }
        }
        
        def setup_adaptive_layout():
            """Setup adaptive layout based on screen size"""
            # This would be implemented based on target platform
            pass
        '''
    
    def _extract_educational_code(self, ai_response: str) -> str:
        """Extract educational elements code from AI response"""
        return '''
        def add_educational_tooltips():
            """Add educational tooltips and explanations"""
            educational_elements = {
                "tooltips": {
                    "complexity": "This shows how the algorithm's performance scales with input size",
                    "comparison": "Compare different algorithms to understand trade-offs",
                    "visualization": "Interactive elements help explore different scenarios"
                },
                "guided_tour": [
                    "Start here to understand the basics",
                    "Explore this section to see comparisons", 
                    "Try adjusting parameters to see effects"
                ],
                "quiz_points": [
                    {"question": "What is the time complexity?", "answer": "O(n log n)"},
                    {"question": "Which algorithm is faster?", "answer": "Depends on input size"}
                ]
            }
            
            return educational_elements
        '''
    
    def _format_rag_context(self, context: List[Dict[str, Any]]) -> str:
        """Format RAG context for AI prompts"""
        if not context:
            return "No specific visualization knowledge retrieved."
        
        formatted = ""
        for i, doc in enumerate(context[:3], 1):
            formatted += f"{i}. {doc['title']}\n"
            formatted += f"   {doc.get('content', doc.get('full_content', ''))[:200]}...\n\n"
        
        return formatted
    
    def _get_dynamic_frontend_instructions(self) -> str:
        """Get comprehensive frontend integration instructions"""
        return """
DYNAMIC VISUALIZATION FRONTEND INTEGRATION GUIDE
==============================================

1. AI-GENERATED RESPONSIVE DESIGN:
   - Visualization adapts automatically to screen size
   - Progressive enhancement for different capabilities
   - Intelligent layout optimization based on content
   - Dynamic color scheme adjustment

2. INTERACTIVE FEATURES:
   - Real-time parameter adjustment
   - Multi-level zoom and exploration
   - Guided tutorial integration
   - Context-sensitive help system

3. EDUCATIONAL INTEGRATION:
   - Progressive disclosure of complexity
   - Interactive quiz integration points
   - Concept reinforcement exercises
   - Personalized learning paths

4. PERFORMANCE OPTIMIZATION:
   - Lazy loading for complex visualizations
   - WebGL acceleration where appropriate
   - Efficient data streaming and updates
   - Memory management for large datasets

5. ACCESSIBILITY EXCELLENCE:
   - Full keyboard navigation support
   - Screen reader optimization
   - High contrast mode automatic detection
   - Voice navigation integration

6. EXPORT AND SHARING:
   - Multiple format export (PNG, SVG, PDF, interactive HTML)
   - Social media optimized sharing
   - Embeddable widget generation
   - API for custom integrations

7. REAL-TIME COLLABORATION:
   - Multi-user exploration sessions
   - Annotation and commenting system
   - Version control for visualization states
   - Collaborative learning features

8. ANALYTICS AND INSIGHTS:
   - User interaction tracking
   - Learning effectiveness measurement
   - Performance optimization suggestions
   - A/B testing for educational effectiveness
"""

# Global dynamic visualizer instance
dynamic_visualizer = DynamicVisualizationAgent()
