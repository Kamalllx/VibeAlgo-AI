# backend/visualization/base_visualizer.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
from datetime import datetime
import json
import os

class BaseVisualizationAgent(ABC):
    """
    Base class for all visualization agents
    Provides common functionality and interface
    """
    
    def __init__(self):
        self.name = "BaseVisualizer"
        self.role = "Visualization Specialist"
        self.output_format = "static"
        self.specialties = []
        self.generated_files = []
        
        # Common visualization settings
        self.default_figsize = (12, 8)
        self.default_dpi = 300
        self.default_style = 'seaborn-v0_8'
        
        # Color palettes for different scenarios
        self.color_palettes = {
            "default": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
            "complexity": ["#e74c3c", "#f39c12", "#f1c40f", "#27ae60", "#3498db"],
            "performance": ["#2ecc71", "#f39c12", "#e74c3c", "#9b59b6", "#34495e"],
            "educational": ["#3498db", "#e67e22", "#e74c3c", "#f1c40f", "#9b59b6"],
            "comparison": ["#1abc9c", "#e74c3c", "#f39c12", "#9b59b6", "#34495e"]
        }
        
        # Set default style
        plt.style.use(self.default_style)
        
        print(f"🎨 [{self.name}] Base Visualization Agent initialized")
    
    @abstractmethod
    async def generate_visualization(self, request) -> Dict[str, Any]:
        """
        Abstract method that must be implemented by all visualization agents
        
        Args:
            request: VisualizationRequest object with scenario data
            
        Returns:
            Dict containing visualization results, files, and metadata
        """
        pass
    
    def set_style_theme(self, theme: str = "modern"):
        """Set visualization style theme"""
        style_map = {
            "modern": "seaborn-v0_8",
            "classic": "classic",
            "minimal": "seaborn-v0_8-whitegrid",
            "dark": "dark_background",
            "academic": "seaborn-v0_8-paper"
        }
        
        style = style_map.get(theme, self.default_style)
        plt.style.use(style)
        print(f"🎨 Style theme set to: {theme} ({style})")
    
    def get_color_palette(self, scenario_type: str = "default", n_colors: int = 5) -> List[str]:
        """Get appropriate color palette for scenario"""
        palette = self.color_palettes.get(scenario_type, self.color_palettes["default"])
        
        # Extend palette if more colors needed
        if len(palette) < n_colors:
            import seaborn as sns
            extended_palette = sns.color_palette("husl", n_colors)
            return [f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" 
                   for r, g, b in extended_palette]
        
        return palette[:n_colors]
    
    def create_figure(self, figsize: Tuple[int, int] = None, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """Create standardized figure with consistent styling"""
        if figsize is None:
            figsize = self.default_figsize
            
        fig, ax = plt.subplots(figsize=figsize, **kwargs)
        
        # Apply consistent styling
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        return fig, ax
    
    def create_subplots(self, rows: int, cols: int, figsize: Tuple[int, int] = None, **kwargs) -> Tuple[plt.Figure, np.ndarray]:
        """Create standardized subplots with consistent styling"""
        if figsize is None:
            figsize = (self.default_figsize[0] * cols * 0.8, self.default_figsize[1] * rows * 0.8)
            
        fig, axes = plt.subplots(rows, cols, figsize=figsize, **kwargs)
        
        # Ensure axes is always array
        if rows * cols == 1:
            axes = np.array([axes])
        elif rows == 1 or cols == 1:
            axes = np.array(axes)
        
        # Apply consistent styling to all subplots
        for ax in axes.flat:
            ax.grid(True, alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        return fig, axes
    
    def save_figure(self, fig: plt.Figure, filename: str, formats: List[str] = None) -> List[str]:
        """Save figure in multiple formats"""
        if formats is None:
            formats = ['png', 'svg']
        
        saved_files = []
        
        for fmt in formats:
            filepath = f"{filename}.{fmt}"
            
            if fmt == 'png':
                fig.savefig(filepath, dpi=self.default_dpi, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
            elif fmt == 'svg':
                fig.savefig(filepath, format='svg', bbox_inches='tight')
            elif fmt == 'pdf':
                fig.savefig(filepath, format='pdf', bbox_inches='tight')
            
            saved_files.append(filepath)
            self.generated_files.append(filepath)
        
        print(f"💾 Saved visualization: {saved_files}")
        return saved_files
    
    def add_educational_annotations(self, ax: plt.Axes, annotations: List[Dict[str, Any]]):
        """Add educational annotations to plots"""
        for annotation in annotations:
            ax.annotate(
                annotation.get('text', ''),
                xy=annotation.get('xy', (0.5, 0.5)),
                xytext=annotation.get('xytext', (0.6, 0.6)),
                arrowprops=annotation.get('arrowprops', dict(arrowstyle='->', color='red')),
                fontsize=annotation.get('fontsize', 10),
                bbox=annotation.get('bbox', dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
            )
    
    def create_legend(self, ax: plt.Axes, location: str = 'best', **kwargs):
        """Create standardized legend"""
        legend = ax.legend(loc=location, frameon=True, shadow=True, **kwargs)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)
        return legend
    
    def format_axes(self, ax: plt.Axes, title: str = None, xlabel: str = None, ylabel: str = None, **kwargs):
        """Apply consistent axis formatting"""
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=12)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=12)
        
        # Apply any additional formatting
        for key, value in kwargs.items():
            if hasattr(ax, f'set_{key}'):
                getattr(ax, f'set_{key}')(value)
    
    def _extract_complexity_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract complexity information from analysis data"""
        complexity_data = {}
        
        if isinstance(data, dict):
            # Try to extract from various possible sources
            if "complexity_analysis" in data:
                complexity_data = data["complexity_analysis"]
            elif "agent_result" in data and "complexity_analysis" in data["agent_result"]:
                complexity_data = data["agent_result"]["complexity_analysis"]
            elif "optimal_solution" in data and "complexity_analysis" in data["optimal_solution"]:
                complexity_data = data["optimal_solution"]["complexity_analysis"]
            else:
                complexity_data = data
        
        # Set defaults if not present
        complexity_data.setdefault("time_complexity", "O(n)")
        complexity_data.setdefault("space_complexity", "O(1)")
        complexity_data.setdefault("algorithm_name", "Unknown Algorithm")
        
        return complexity_data
    
    def _extract_algorithm_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract algorithm information from request data"""
        algorithm_data = {}
        
        if isinstance(data, dict):
            # Extract code
            if "optimal_solution" in data and "code" in data["optimal_solution"]:
                code_data = data["optimal_solution"]["code"]
                if isinstance(code_data, dict):
                    algorithm_data["code"] = code_data.get("code", "")
                else:
                    algorithm_data["code"] = str(code_data)
            elif "code" in data:
                algorithm_data["code"] = data["code"]
            
            # Extract complexity information
            if "complexity_analysis" in data:
                complexity = data["complexity_analysis"]
                algorithm_data["time_complexity"] = complexity.get("time_complexity", "Unknown")
                algorithm_data["space_complexity"] = complexity.get("space_complexity", "Unknown")
            
            # Try to detect algorithm type from code
            code = algorithm_data.get("code", "").lower()
            if "sort" in code:
                algorithm_data["algorithm_type"] = "sorting"
            elif "search" in code:
                algorithm_data["algorithm_type"] = "searching"
            elif "graph" in code or "bfs" in code or "dfs" in code:
                algorithm_data["algorithm_type"] = "graph_traversal"
            else:
                algorithm_data["algorithm_type"] = "general"
        
        return algorithm_data
    
    def _extract_performance_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract performance information from request data"""
        performance_data = {}
        
        if isinstance(data, dict):
            # Look for performance metrics
            performance_keys = ["execution_time", "memory_usage", "cpu_usage", "benchmark_results"]
            for key in performance_keys:
                if key in data:
                    performance_data[key] = data[key]
            
            # Extract from nested structures
            if "ai_processing" in data:
                ai_data = data["ai_processing"]
                performance_data["tokens_used"] = ai_data.get("total_tokens", 0)
                performance_data["processing_time"] = ai_data.get("processing_time", 0)
            
            # Set defaults
            performance_data.setdefault("algorithm_name", "Unknown Algorithm")
            performance_data.setdefault("dataset_size", "Unknown")
        
        return performance_data
    def create_visualization_result(self, agent_name: str, viz_type: str, code: str, 
                                  data: Dict[str, Any], file_paths: List[str], 
                                  metadata: Dict[str, Any], instructions: str) -> Dict[str, Any]:
        """Create standardized visualization result"""
        return {
            "agent_name": agent_name,
            "visualization_type": viz_type,
            "code": code,
            "data": data,
            "file_paths": file_paths,
            "metadata": metadata,
            "frontend_instructions": instructions
        }
    def export_visualization_data(self, data: Dict[str, Any], filename: str):
        """Export visualization data as JSON for frontend integration"""
        export_data = {
            "agent_name": self.name,
            "generated_at": datetime.now().isoformat(),
            "visualization_data": data,
            "generated_files": self.generated_files,
            "metadata": {
                "style_theme": self.default_style,
                "color_palette": "default",
                "specialties": self.specialties
            }
        }
        
        with open(f"{filename}.json", 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"📊 Exported visualization data: {filename}.json")
        return f"{filename}.json"
    
    def get_frontend_integration_code(self) -> str:
        """Generate frontend integration code template"""
        return f"""
/* 
Frontend Integration Template for {self.name}
Generated on: {datetime.now().isoformat()}
*/

class {self.name}Integration {{
    constructor(containerId) {{
        this.container = document.getElementById(containerId);
        this.visualizations = [];
    }}
    
    loadVisualization(dataUrl) {{
        fetch(dataUrl)
            .then(response => response.json())
            .then(data => {{
                this.renderVisualization(data);
            }});
    }}
    
    renderVisualization(data) {{
        // Implementation depends on visualization type
        // For static images: create <img> elements
        // For interactive charts: use Plotly.js or D3.js
        // For animations: use CSS animations or video elements
    }}
    
    exportVisualization(format = 'png') {{
        // Export functionality
    }}
    
    addInteractivity() {{
        // Add interactive controls
    }}
}}

// Usage example:
// const visualizer = new {self.name}Integration('visualization-container');
// visualizer.loadVisualization('visualization_data.json');
"""
    
    def cleanup(self):
        """Clean up resources and temporary files"""
        plt.close('all')  # Close all matplotlib figures
        print(f"🧹 Cleaned up resources for {self.name}")
    