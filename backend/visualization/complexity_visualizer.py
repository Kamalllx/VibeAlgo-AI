# backend/visualization/complexity_visualizer.py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Any
from ai.groq_client import groq_client
from visualization.base_visualizer import BaseVisualizationAgent

class ComplexityVisualizationAgent(BaseVisualizationAgent):
    def __init__(self):
        super().__init__()
        self.name = "ComplexityVisualizer"
        self.specialties = [
            "time_complexity_graphs",
            "space_complexity_charts", 
            "big_o_comparisons",
            "asymptotic_analysis",
            "complexity_trends"
        ]
        
        print(f"📊 [{self.name}] Complexity Visualization Agent initialized")
    
    async def generate_visualization(self, request) -> Dict[str, Any]:
        """Generate complexity-focused visualizations"""
        print(f"\n📊 [{self.name}] GENERATING COMPLEXITY VISUALIZATIONS")
        
        # Extract complexity data
        complexity_data = self._extract_complexity_data(request.data)
        
        # Generate AI-powered visualization recommendations
        viz_recommendations = await self._get_ai_recommendations(complexity_data, request)
        
        # Generate multiple visualization types
        visualizations = []
        
        # 1. Big O Comparison Chart
        if "comparison" in request.requirements:
            big_o_viz = self._create_big_o_comparison(complexity_data)
            visualizations.append(big_o_viz)
        
        # 2. Complexity Growth Curves
        growth_viz = self._create_complexity_curves(complexity_data)
        visualizations.append(growth_viz)
        
        # 3. Interactive Complexity Explorer
        if request.output_format == "interactive":
            interactive_viz = self._create_interactive_complexity_explorer(complexity_data)
            visualizations.append(interactive_viz)
        
        # 4. 3D Complexity Landscape
        if "advanced" in request.requirements:
            landscape_viz = self._create_3d_complexity_landscape(complexity_data)
            visualizations.append(landscape_viz)
        
        # Combine all visualizations
        combined_code = self._combine_complexity_visualizations(visualizations)
        
        return {
            "agent_name": self.name,
            "visualization_type": "complexity_analysis",
            "code": combined_code,
            "data": complexity_data,
            "file_paths": ["complexity_chart.png", "complexity_interactive.html"],
            "metadata": {
                "chart_types": ["big_o_comparison", "growth_curves", "interactive_explorer"],
                "complexity_analyzed": complexity_data.get("time_complexity", "Unknown")
            },
            "frontend_instructions": self._get_frontend_instructions()
        }
    
    async def _get_ai_recommendations(self, complexity_data: Dict[str, Any], request) -> List[str]:
        """Get AI recommendations for best visualization approaches"""
        recommendation_prompt = f"""
        As a data visualization expert, recommend the best visualization approaches for this complexity analysis:
        
        Complexity Data: {complexity_data}
        Scenario: {request.scenario_type}
        Requirements: {request.requirements}
        
        Suggest:
        1. Most effective chart types for this complexity
        2. Visual encodings (color, size, animation)
        3. Interactive elements to include
        4. Comparative elements if applicable
        
        Focus on clarity and educational value.
        """
        
        response = groq_client.chat_completion([
            {"role": "system", "content": "You are a data visualization expert specializing in algorithm complexity visualization."},
            {"role": "user", "content": recommendation_prompt}
        ])
        
        return self._parse_recommendations(response.content)
    def _parse_recommendations(self, ai_response: str) -> List[str]:
        """Parse AI recommendations into actionable items - MISSING METHOD"""
        lines = ai_response.split('\n')
        recommendations = []
        for line in lines:
            if line.strip() and (line.startswith('-') or line.startswith('•') or line.startswith('*')):
                recommendations.append(line.strip())
        return recommendations[:5]  # Top 5 recommendations
    def _create_big_o_comparison(self, complexity_data: Dict[str, Any]) -> str:
        """Create Big O notation comparison visualization"""
        return '''
def create_big_o_comparison():
    """
    Big O Complexity Comparison Chart
    
    Frontend Integration:
    - Export as SVG for scalability
    - Use consistent color scheme
    - Add hover tooltips for details
    - Responsive design for mobile
    """
    
    # Big O complexities and their growth rates
    n_values = np.logspace(1, 3, 50)  # 10 to 1000
    
    complexities = {
        'O(1)': np.ones_like(n_values),
        'O(log n)': np.log2(n_values),
        'O(n)': n_values,
        'O(n log n)': n_values * np.log2(n_values),
        'O(n²)': n_values ** 2,
        'O(2^n)': 2 ** (n_values / 100)  # Scaled down for visualization
    }
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Linear scale plot
    colors = plt.cm.viridis(np.linspace(0, 1, len(complexities)))
    for i, (name, values) in enumerate(complexities.items()):
        if name != 'O(2^n)':  # Exclude exponential for linear plot
            ax1.plot(n_values, values, label=name, color=colors[i], linewidth=3)
    
    ax1.set_xlabel('Input Size (n)', fontsize=12)
    ax1.set_ylabel('Operations', fontsize=12)
    ax1.set_title('Time Complexity Comparison (Linear Scale)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Log scale plot
    for i, (name, values) in enumerate(complexities.items()):
        ax2.loglog(n_values, values, label=name, color=colors[i], linewidth=3, marker='o', markersize=4)
    
    ax2.set_xlabel('Input Size (n)', fontsize=12)
    ax2.set_ylabel('Operations (log scale)', fontsize=12)
    ax2.set_title('Time Complexity Comparison (Log Scale)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Highlight current algorithm's complexity
    current_complexity = complexity_data.get('time_complexity', 'O(n)')
    if current_complexity in complexities:
        for ax in [ax1, ax2]:
            # Add annotation for current algorithm
            ax.annotate(f'Your Algorithm: {current_complexity}', 
                       xy=(500, complexities[current_complexity][25]), 
                       xytext=(600, complexities[current_complexity][25] * 2),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2),
                       fontsize=12, fontweight='bold', color='red')
    
    plt.savefig('big_o_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('big_o_comparison.svg', bbox_inches='tight')  # For web
    plt.show()
'''
    
    def _create_complexity_curves(self, complexity_data: Dict[str, Any]) -> str:
        """Create detailed complexity growth curves"""
        return '''
def create_complexity_curves():
    """
    Detailed Complexity Growth Analysis
    
    Frontend Integration:
    - Interactive zoom capabilities
    - Tooltip showing exact values
    - Export as both PNG and interactive HTML
    - Responsive breakpoints for different screen sizes
    """
    
    # Create detailed analysis for specific complexity
    current_complexity = complexity_data.get('time_complexity', 'O(n)')
    space_complexity = complexity_data.get('space_complexity', 'O(1)')
    
    # Generate data points
    n_values = np.arange(1, 101)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Time Complexity Growth
    if 'O(n²)' in current_complexity:
        time_values = n_values ** 2
        ax1.fill_between(n_values, time_values, alpha=0.3, color='red', label='Worst Case')
        ax1.plot(n_values, time_values * 0.5, '--', color='orange', label='Average Case')
        ax1.plot(n_values, n_values, ':', color='green', label='Best Case (if optimized)')
    elif 'O(n log n)' in current_complexity:
        time_values = n_values * np.log2(n_values)
        ax1.fill_between(n_values, time_values, alpha=0.3, color='blue')
        ax1.plot(n_values, time_values, linewidth=3, color='blue')
    elif 'O(n)' in current_complexity:
        time_values = n_values
        ax1.fill_between(n_values, time_values, alpha=0.3, color='green')
        ax1.plot(n_values, time_values, linewidth=3, color='green')
    
    ax1.set_title(f'Time Complexity: {current_complexity}', fontweight='bold')
    ax1.set_xlabel('Input Size (n)')
    ax1.set_ylabel('Time Units')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Space Complexity Analysis
    if 'O(n)' in space_complexity:
        space_values = n_values
    elif 'O(log n)' in space_complexity:
        space_values = np.log2(n_values)
    else:  # O(1)
        space_values = np.ones_like(n_values)
    
    ax2.bar(n_values[::10], space_values[::10], alpha=0.7, color='purple')
    ax2.set_title(f'Space Complexity: {space_complexity}', fontweight='bold')
    ax2.set_xlabel('Input Size (n)')
    ax2.set_ylabel('Memory Units')
    ax2.grid(True, alpha=0.3)
    
    # 3. Performance Prediction
    actual_input_sizes = [100, 1000, 10000, 100000]
    if 'O(n²)' in current_complexity:
        predicted_times = [x**2 / 10000 for x in actual_input_sizes]  # Normalized
    elif 'O(n log n)' in current_complexity:
        predicted_times = [x * np.log2(x) / 10000 for x in actual_input_sizes]
    else:
        predicted_times = [x / 1000 for x in actual_input_sizes]
    
    bars = ax3.bar(range(len(actual_input_sizes)), predicted_times, 
                   color=['green', 'yellow', 'orange', 'red'])
    ax3.set_title('Performance Prediction', fontweight='bold')
    ax3.set_xlabel('Input Size')
    ax3.set_ylabel('Predicted Time (seconds)')
    ax3.set_xticks(range(len(actual_input_sizes)))
    ax3.set_xticklabels([f'{x:,}' for x in actual_input_sizes])
    
    # Add value labels on bars
    for bar, value in zip(bars, predicted_times):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}s', ha='center', va='bottom')
    
    # 4. Complexity Comparison with Common Algorithms
    algorithms = ['Linear Search', 'Binary Search', 'Bubble Sort', 'Merge Sort', 'Quick Sort']
    complexities_values = [1000, 13, 1000000, 13000, 10000]  # For n=1000
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'gold', 'lightpink']
    
    bars = ax4.barh(algorithms, complexities_values, color=colors)
    ax4.set_title('Algorithm Comparison (n=1000)', fontweight='bold')
    ax4.set_xlabel('Operations')
    ax4.set_xscale('log')
    
    # Highlight current algorithm if in list
    current_algorithm = complexity_data.get('algorithm_name', '')
    if current_algorithm in algorithms:
        idx = algorithms.index(current_algorithm)
        bars[idx].set_color('red')
        bars[idx].set_alpha(0.8)
    
    plt.tight_layout()
    plt.savefig('complexity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
'''
    
    def _create_interactive_complexity_explorer(self, complexity_data: Dict[str, Any]) -> str:
        """Create interactive Plotly visualization"""
        return '''
def create_interactive_complexity_explorer():
    """
    Interactive Complexity Explorer using Plotly
    
    Frontend Integration:
    - Embed as HTML iframe or use Plotly.js directly
    - Maintain interactivity in web version
    - Add custom controls for parameter adjustment
    - Export options for sharing
    """
    
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    
    # Create subplot with secondary y-axis
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Time vs Space Complexity', 'Growth Rate Analysis', 
                       'Performance Scaling', 'Algorithm Comparison'),
        specs=[[{"secondary_y": True}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "radar"}]]
    )
    
    # Input size range
    n_values = list(range(1, 101))
    
    # Time complexity data
    current_complexity = complexity_data.get('time_complexity', 'O(n)')
    if 'O(n²)' in current_complexity:
        time_data = [x**2 for x in n_values]
    elif 'O(n log n)' in current_complexity:
        time_data = [x * np.log2(x) if x > 1 else x for x in n_values]
    else:
        time_data = n_values
    
    # Space complexity data
    space_complexity = complexity_data.get('space_complexity', 'O(1)')
    if 'O(n)' in space_complexity:
        space_data = n_values
    else:
        space_data = [1] * len(n_values)
    
    # Add time complexity trace
    fig.add_trace(
        go.Scatter(x=n_values, y=time_data, name='Time Complexity',
                  line=dict(color='red', width=3),
                  hovertemplate='Input Size: %{x}<br>Time Units: %{y}<extra></extra>'),
        row=1, col=1
    )
    
    # Add space complexity trace
    fig.add_trace(
        go.Scatter(x=n_values, y=space_data, name='Space Complexity',
                  line=dict(color='blue', width=3),
                  yaxis='y2',
                  hovertemplate='Input Size: %{x}<br>Space Units: %{y}<extra></extra>'),
        row=1, col=1
    )
    
    # Growth rate analysis
    growth_rates = np.diff(time_data)
    fig.add_trace(
        go.Scatter(x=n_values[1:], y=growth_rates, name='Growth Rate',
                  mode='lines+markers',
                  line=dict(color='green', width=2)),
        row=1, col=2
    )
    
    # Performance scaling bars
    input_sizes = [100, 1000, 10000]
    performance_times = [time_data[99], time_data[99]*10, time_data[99]*100]
    
    fig.add_trace(
        go.Bar(x=input_sizes, y=performance_times, name='Performance Scaling',
               marker_color='orange',
               hovertemplate='Input Size: %{x}<br>Time: %{y}<extra></extra>'),
        row=2, col=1
    )
    
    # Algorithm comparison radar chart
    categories = ['Time Efficiency', 'Space Efficiency', 'Implementation Ease', 
                 'Stability', 'Best Case Performance']
    
    # Score based on complexity (this would be dynamically calculated)
    if 'O(n²)' in current_complexity:
        values = [2, 4, 5, 3, 2]  # Bubble sort characteristics
    elif 'O(n log n)' in current_complexity:
        values = [4, 3, 3, 4, 4]  # Merge sort characteristics
    else:
        values = [5, 5, 4, 5, 5]  # Linear algorithm characteristics
    
    fig.add_trace(
        go.Scatterpolar(r=values, theta=categories, fill='toself',
                       name='Algorithm Profile'),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text="Interactive Algorithm Complexity Analysis",
        title_x=0.5,
        showlegend=True,
        height=800
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Input Size (n)", row=1, col=1)
    fig.update_yaxes(title_text="Time Units", row=1, col=1)
    fig.update_yaxes(title_text="Space Units", secondary_y=True, row=1, col=1)
    
    # Save as HTML for web integration
    pyo.plot(fig, filename='complexity_interactive.html', auto_open=False)
    
    # Also save as static image
    fig.write_image("complexity_interactive.png", width=1200, height=800)
    
    return fig
'''
    
    def _create_3d_complexity_landscape(self, complexity_data: Dict[str, Any]) -> str:
        """Create 3D complexity visualization"""
        return '''
def create_3d_complexity_landscape():
    """
    3D Complexity Landscape Visualization
    
    Frontend Integration:
    - Use Three.js or WebGL for web version
    - Provide rotation and zoom controls
    - Add interactive tooltips
    - Export as interactive 3D model
    """
    
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create 3D complexity landscape
    # X-axis: Input size, Y-axis: Algorithm variation, Z-axis: Operations
    
    x = np.linspace(1, 100, 50)
    y = np.linspace(1, 10, 20)  # Different algorithm parameters
    X, Y = np.meshgrid(x, y)
    
    # Z represents operations based on complexity
    current_complexity = complexity_data.get('time_complexity', 'O(n)')
    if 'O(n²)' in current_complexity:
        Z = X**2 * Y  # Quadratic with parameter variation
    elif 'O(n log n)' in current_complexity:
        Z = X * np.log2(X) * Y
    else:
        Z = X * Y
    
    # Create surface plot
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    
    # Add contour lines
    ax.contour(X, Y, Z, zdir='z', offset=Z.min(), cmap='viridis', alpha=0.5)
    ax.contour(X, Y, Z, zdir='x', offset=1, cmap='viridis', alpha=0.5)
    ax.contour(X, Y, Z, zdir='y', offset=1, cmap='viridis', alpha=0.5)
    
    # Labels and title
    ax.set_xlabel('Input Size (n)', fontsize=12)
    ax.set_ylabel('Algorithm Parameter', fontsize=12)
    ax.set_zlabel('Operations', fontsize=12)
    ax.set_title(f'3D Complexity Landscape: {current_complexity}', fontsize=14, fontweight='bold')
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    # Add annotation for optimal region
    if Z.min() < Z.max():
        min_idx = np.unravel_index(np.argmin(Z), Z.shape)
        ax.scatter([X[min_idx]], [Y[min_idx]], [Z[min_idx]], 
                  color='red', s=100, label='Optimal Point')
    
    ax.legend()
    plt.savefig('complexity_3d.png', dpi=300, bbox_inches='tight')
    plt.show()
'''
    
    def _combine_complexity_visualizations(self, visualizations: List[str]) -> str:
        """Combine all complexity visualizations into one comprehensive module"""
        header = '''
"""
COMPREHENSIVE COMPLEXITY VISUALIZATION MODULE
Generated by ComplexityVisualizationAgent

Frontend Integration Instructions:
==================================

1. WEB INTEGRATION:
   - Convert matplotlib figures to SVG for scalability
   - Use Plotly.js for interactive charts  
   - Implement responsive design with CSS Grid/Flexbox
   - Add loading states for complex visualizations

2. EXPORT CAPABILITIES:
   - PNG/SVG for static images
   - HTML for interactive charts
   - PDF for reports
   - JSON data for custom implementations

3. PERFORMANCE OPTIMIZATION:
   - Lazy load complex 3D visualizations
   - Use canvas rendering for animations
   - Implement progressive enhancement
   - Cache visualization data

4. ACCESSIBILITY:
   - Alt text for all charts
   - Keyboard navigation for interactive elements
   - High contrast mode support
   - Screen reader compatible

5. MOBILE CONSIDERATIONS:
   - Touch-friendly interactive elements
   - Responsive breakpoints
   - Simplified views for small screens
   - Swipe gestures for 3D navigation

Dependencies Required:
- matplotlib >= 3.5.0
- plotly >= 5.0.0
- numpy >= 1.21.0
- seaborn >= 0.11.0
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from mpl_toolkits.mplot3d import Axes3D

# Set style for consistent appearance
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

'''
        
        combined = header
        for viz in visualizations:
            combined += viz + "\n\n"
        
        combined += '''
def generate_all_complexity_visualizations(complexity_data):
    """
    Master function to generate all complexity visualizations
    
    Args:
        complexity_data (dict): Dictionary containing complexity analysis results
    
    Returns:
        dict: Paths to generated visualization files
    """
    
    print("🎨 Generating comprehensive complexity visualizations...")
    
    # Generate all visualizations
    create_big_o_comparison()
    create_complexity_curves() 
    create_interactive_complexity_explorer()
    create_3d_complexity_landscape()
    
    generated_files = {
        "static_charts": [
            "big_o_comparison.png",
            "big_o_comparison.svg", 
            "complexity_analysis.png",
            "complexity_3d.png"
        ],
        "interactive_charts": [
            "complexity_interactive.html"
        ],
        "data_exports": [
            "complexity_data.json"
        ]
    }
    
    print("✅ All complexity visualizations generated successfully!")
    return generated_files

if __name__ == "__main__":
    # Example usage
    sample_complexity_data = {
        "time_complexity": "O(n²)",
        "space_complexity": "O(1)", 
        "algorithm_name": "Bubble Sort"
    }
    
    generate_all_complexity_visualizations(sample_complexity_data)
'''
        
        return combined
    
    def _get_frontend_instructions(self) -> str:
        """Get detailed frontend integration instructions"""
        return """
FRONTEND INTEGRATION GUIDE FOR COMPLEXITY VISUALIZATIONS
=======================================================

1. STATIC CHARTS (PNG/SVG):
   - Use <img> tags with responsive CSS
   - Implement lazy loading for performance
   - Add zoom functionality with libraries like PhotoSwipe
   - Provide download buttons for exports

2. INTERACTIVE CHARTS (Plotly HTML):
   - Embed using <iframe> or integrate Plotly.js directly
   - Maintain responsive behavior with resizeobserver
   - Add custom styling to match application theme
   - Implement data update mechanisms

3. 3D VISUALIZATIONS:
   - Use Three.js for web-based 3D rendering
   - Provide fallback 2D views for mobile
   - Add touch controls for mobile devices
   - Implement progressive loading for complex scenes

4. DATA INTEGRATION:
   - Use WebSocket for real-time updates
   - Implement caching strategy for visualization data
   - Add loading states and error handling
   - Provide accessibility alternatives

5. RESPONSIVE DESIGN:
   - Mobile-first approach with progressive enhancement
   - Use CSS Grid for dashboard layouts
   - Implement touch gestures for interactions
   - Ensure readability across all screen sizes
"""

# Base visualizer class for other agents
class BaseVisualizationAgent:
    def __init__(self):
        self.name = "BaseVisualizer"
        self.output_format = "static"
    
    def _extract_complexity_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract complexity information from analysis data"""
        if isinstance(data, dict):
            # Try to extract from complexity analysis result
            if "complexity_analysis" in data:
                return data["complexity_analysis"]
            elif "agent_result" in data and "complexity_analysis" in data["agent_result"]:
                return data["agent_result"]["complexity_analysis"]
            else:
                return data
        return {}
    
    def _parse_recommendations(self, ai_response: str) -> List[str]:
        """Parse AI recommendations into actionable items"""
        lines = ai_response.split('\n')
        recommendations = []
        for line in lines:
            if line.strip() and (line.startswith('-') or line.startswith('•') or line.startswith('*')):
                recommendations.append(line.strip())
        return recommendations[:5]  # Top 5 recommendations
