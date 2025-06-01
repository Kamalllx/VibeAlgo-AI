# backend/visualization/performance_analyzer.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import psutil
import time
from typing import Dict, List, Any, Tuple
from ai.groq_client import groq_client
from visualization.base_visualizer import BaseVisualizationAgent

class PerformanceVisualizationAgent(BaseVisualizationAgent):
    def __init__(self):
        super().__init__()
        self.name = "PerformanceAnalyzer"
        self.role = "Performance Visualization Specialist"
        self.specialties = [
            "execution_time_analysis",
            "memory_usage_visualization",
            "cpu_performance_charts",
            "benchmark_comparisons",
            "resource_utilization_tracking",
            "scalability_analysis"
        ]
        
        print(f"📈 [{self.name}] Performance Visualization Agent initialized")
    
    async def generate_visualization(self, request) -> Dict[str, Any]:
        """Generate performance-focused visualizations"""
        print(f"\n📈 [{self.name}] GENERATING PERFORMANCE VISUALIZATIONS")
        
        # Extract performance data
        performance_data = self._extract_performance_data(request.data)
        
        # Get AI recommendations for performance visualization
        viz_recommendations = await self._get_performance_recommendations(performance_data, request)
        
        # Generate multiple performance visualization types
        visualizations = []
        
        # 1. Execution Time Analysis
        time_analysis = self._create_execution_time_analysis(performance_data)
        visualizations.append(time_analysis)
        
        # 2. Memory Usage Visualization
        if "memory" in request.requirements or "memory_usage" in performance_data:
            memory_viz = self._create_memory_usage_visualization(performance_data)
            visualizations.append(memory_viz)
        
        # 3. CPU Performance Charts
        if "cpu" in request.requirements or "cpu_usage" in performance_data:
            cpu_viz = self._create_cpu_performance_charts(performance_data)
            visualizations.append(cpu_viz)
        
        # 4. Algorithm Comparison Benchmarks
        if "comparison" in request.requirements:
            benchmark_viz = self._create_benchmark_comparison(performance_data)
            visualizations.append(benchmark_viz)
        
        # 5. Real-time Performance Monitor
        if request.output_format == "interactive":
            realtime_viz = self._create_realtime_performance_monitor(performance_data)
            visualizations.append(realtime_viz)
        
        # Combine all visualizations
        combined_code = self._combine_performance_visualizations(visualizations)
        
        return {
            "agent_name": self.name,
            "visualization_type": "performance_analysis",
            "code": combined_code,
            "data": performance_data,
            "file_paths": ["performance_analysis.png", "benchmark_comparison.html", "realtime_monitor.json"],
            "metadata": {
                "visualization_types": ["execution_time", "memory_usage", "cpu_performance", "benchmarks"],
                "performance_metrics": list(performance_data.keys()),
                "recommendations": viz_recommendations
            },
            "frontend_instructions": self._get_performance_frontend_instructions()
        }
    
    async def _get_performance_recommendations(self, performance_data: Dict[str, Any], request) -> List[str]:
        """Get AI recommendations for performance visualization"""
        recommendation_prompt = f"""
        As a performance analysis expert, recommend the best visualization approaches for this performance data:
        
        Performance Data: {performance_data}
        Scenario: {request.scenario_type}
        Requirements: {request.requirements}
        
        Suggest:
        1. Most effective chart types for performance metrics
        2. Key performance indicators to highlight
        3. Comparison strategies for multiple algorithms
        4. Real-time monitoring visualizations
        5. Bottleneck identification techniques
        
        Focus on actionable insights and optimization opportunities.
        """
        
        response = groq_client.chat_completion([
            {"role": "system", "content": "You are a performance analysis expert specializing in algorithm optimization and system monitoring."},
            {"role": "user", "content": recommendation_prompt}
        ])
        
        return self._parse_recommendations(response.content)
    
    def _create_execution_time_analysis(self, performance_data: Dict[str, Any]) -> str:
        """Create execution time analysis visualization"""
        return '''
def create_execution_time_analysis():
    """
    Execution Time Analysis Visualization
    
    Frontend Integration:
    - Use responsive charts for different screen sizes
    - Add zoom functionality for detailed analysis
    - Implement time range selection
    - Export data as CSV for further analysis
    """
    
    # Sample execution time data
    input_sizes = [10, 50, 100, 500, 1000, 5000, 10000]
    
    # Simulate execution times for different algorithms
    algorithms = {
        'Bubble Sort': [size**2 * 0.001 for size in input_sizes],
        'Quick Sort': [size * np.log2(size) * 0.0001 for size in input_sizes],
        'Merge Sort': [size * np.log2(size) * 0.00015 for size in input_sizes],
        'Binary Search': [np.log2(size) * 0.0001 for size in input_sizes if size > 0]
    }
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Linear scale comparison
    colors = self.get_color_palette("performance", len(algorithms))
    for i, (name, times) in enumerate(algorithms.items()):
        sizes_to_plot = input_sizes[:len(times)]
        ax1.plot(sizes_to_plot, times, label=name, color=colors[i], 
                linewidth=2, marker='o', markersize=6)
    
    ax1.set_xlabel('Input Size')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title('Algorithm Performance Comparison (Linear Scale)', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Log scale comparison
    for i, (name, times) in enumerate(algorithms.items()):
        sizes_to_plot = input_sizes[:len(times)]
        ax2.loglog(sizes_to_plot, times, label=name, color=colors[i], 
                  linewidth=2, marker='s', markersize=6)
    
    ax2.set_xlabel('Input Size (log scale)')
    ax2.set_ylabel('Execution Time (log scale)')
    ax2.set_title('Algorithm Performance Comparison (Log Scale)', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Performance ratio analysis
    base_algorithm = 'Quick Sort'
    if base_algorithm in algorithms:
        base_times = algorithms[base_algorithm]
        for i, (name, times) in enumerate(algorithms.items()):
            if name != base_algorithm:
                sizes_to_plot = input_sizes[:min(len(times), len(base_times))]
                ratios = [t1/t2 if t2 > 0 else 0 for t1, t2 in zip(times[:len(sizes_to_plot)], base_times[:len(sizes_to_plot)])]
                ax3.plot(sizes_to_plot, ratios, label=f'{name} / {base_algorithm}', 
                        color=colors[i], linewidth=2, marker='^')
    
    ax3.set_xlabel('Input Size')
    ax3.set_ylabel('Performance Ratio')
    ax3.set_title('Performance Ratio Comparison', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Equal Performance')
    
    # 4. Efficiency Analysis (Operations per second)
    for i, (name, times) in enumerate(algorithms.items()):
        sizes_to_plot = input_sizes[:len(times)]
        ops_per_sec = [size/time if time > 0 else 0 for size, time in zip(sizes_to_plot, times)]
        ax4.bar([x + i*0.15 for x in range(len(sizes_to_plot))], ops_per_sec, 
               width=0.15, label=name, color=colors[i], alpha=0.8)
    
    ax4.set_xlabel('Input Size Category')
    ax4.set_ylabel('Operations per Second')
    ax4.set_title('Algorithm Efficiency (Operations/Second)', fontweight='bold')
    ax4.set_xticks(range(len(input_sizes)))
    ax4.set_xticklabels([f'{size}' for size in input_sizes])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Add performance insights
    fig.suptitle('Comprehensive Execution Time Analysis', fontsize=16, fontweight='bold', y=1.02)
    
    # Save in multiple formats
    self.save_figure(fig, 'execution_time_analysis', ['png', 'svg', 'pdf'])
    plt.show()
    
    return fig
'''
    
    def _create_memory_usage_visualization(self, performance_data: Dict[str, Any]) -> str:
        """Create memory usage visualization"""
        return '''
def create_memory_usage_visualization():
    """
    Memory Usage Analysis Visualization
    
    Frontend Integration:
    - Real-time memory monitoring
    - Memory leak detection alerts
    - Interactive memory profiling
    - Export memory reports
    """
    
    # Simulate memory usage data
    time_points = np.arange(0, 100, 1)
    
    # Different memory patterns
    memory_patterns = {
        'Efficient Algorithm': 50 + 5 * np.sin(time_points * 0.1) + np.random.normal(0, 2, len(time_points)),
        'Memory Leak': 50 + time_points * 0.5 + np.random.normal(0, 3, len(time_points)),
        'Batch Processing': 50 + 30 * np.sin(time_points * 0.2)**2 + np.random.normal(0, 5, len(time_points)),
        'Optimized Version': 30 + 3 * np.sin(time_points * 0.15) + np.random.normal(0, 1, len(time_points))
    }
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Memory usage over time
    colors = self.get_color_palette("performance", len(memory_patterns))
    for i, (name, usage) in enumerate(memory_patterns.items()):
        ax1.plot(time_points, usage, label=name, color=colors[i], linewidth=2)
        ax1.fill_between(time_points, usage, alpha=0.2, color=colors[i])
    
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Memory Usage (MB)')
    ax1.set_title('Memory Usage Patterns Over Time', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Memory usage distribution
    all_usage = []
    labels = []
    for name, usage in memory_patterns.items():
        all_usage.extend(usage)
        labels.extend([name] * len(usage))
    
    df = pd.DataFrame({'Algorithm': labels, 'Memory_Usage': all_usage})
    
    # Box plot for memory distribution
    algorithms = list(memory_patterns.keys())
    usage_data = [memory_patterns[alg] for alg in algorithms]
    
    box_plot = ax2.boxplot(usage_data, labels=algorithms, patch_artist=True)
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('Memory Usage (MB)')
    ax2.set_title('Memory Usage Distribution by Algorithm', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # 3. Memory efficiency comparison
    avg_memory = [np.mean(usage) for usage in usage_data]
    max_memory = [np.max(usage) for usage in usage_data]
    min_memory = [np.min(usage) for usage in usage_data]
    
    x = np.arange(len(algorithms))
    width = 0.25
    
    ax3.bar(x - width, avg_memory, width, label='Average', color=colors[0], alpha=0.8)
    ax3.bar(x, max_memory, width, label='Maximum', color=colors[1], alpha=0.8)
    ax3.bar(x + width, min_memory, width, label='Minimum', color=colors[2], alpha=0.8)
    
    ax3.set_xlabel('Algorithm')
    ax3.set_ylabel('Memory Usage (MB)')
    ax3.set_title('Memory Statistics Comparison', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(algorithms)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    
    # 4. Memory vs Performance correlation
    # Simulate performance data
    performance_scores = [85, 60, 75, 95]  # Higher is better
    
    scatter = ax4.scatter(avg_memory, performance_scores, 
                         c=range(len(algorithms)), cmap='viridis', 
                         s=200, alpha=0.7, edgecolors='black')
    
    # Add algorithm labels
    for i, alg in enumerate(algorithms):
        ax4.annotate(alg, (avg_memory[i], performance_scores[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax4.set_xlabel('Average Memory Usage (MB)')
    ax4.set_ylabel('Performance Score')
    ax4.set_title('Memory vs Performance Trade-off', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(avg_memory, performance_scores, 1)
    p = np.poly1d(z)
    ax4.plot(sorted(avg_memory), p(sorted(avg_memory)), "r--", alpha=0.8, 
            label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
    ax4.legend()
    
    plt.tight_layout()
    
    # Add system memory info if available
    try:
        memory_info = psutil.virtual_memory()
        fig.suptitle(f'Memory Analysis - System: {memory_info.total/(1024**3):.1f}GB Total, '
                    f'{memory_info.percent}% Used', fontsize=14, y=1.02)
    except:
        fig.suptitle('Memory Usage Analysis', fontsize=14, y=1.02)
    
    self.save_figure(fig, 'memory_usage_analysis', ['png', 'svg'])
    plt.show()
    
    return fig
'''
    
    def _create_cpu_performance_charts(self, performance_data: Dict[str, Any]) -> str:
        """Create CPU performance visualization"""
        return '''
def create_cpu_performance_charts():
    """
    CPU Performance Analysis Charts
    
    Frontend Integration:
    - Real-time CPU monitoring
    - Core utilization breakdown
    - Thermal throttling detection
    - Performance scaling analysis
    """
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. CPU Utilization Over Time
    time_points = np.arange(0, 60, 1)  # 60 seconds
    
    # Simulate multi-core CPU usage
    cores = 4
    cpu_usage = {}
    for core in range(cores):
        base_usage = 20 + core * 10
        noise = np.random.normal(0, 5, len(time_points))
        trend = 10 * np.sin(time_points * 0.1 + core)
        cpu_usage[f'Core {core}'] = np.clip(base_usage + trend + noise, 0, 100)
    
    colors = self.get_color_palette("performance", cores)
    for i, (core, usage) in enumerate(cpu_usage.items()):
        ax1.plot(time_points, usage, label=core, color=colors[i], linewidth=2)
        ax1.fill_between(time_points, usage, alpha=0.3, color=colors[i])
    
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('CPU Usage (%)')
    ax1.set_title('Multi-Core CPU Utilization', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # 2. Algorithm CPU Efficiency
    algorithms = ['Bubble Sort', 'Quick Sort', 'Merge Sort', 'Heap Sort']
    cpu_efficiency = [25, 85, 80, 75]  # Efficiency percentage
    energy_consumption = [100, 40, 45, 50]  # Relative energy consumption
    
    x = np.arange(len(algorithms))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, cpu_efficiency, width, label='CPU Efficiency (%)', 
                   color=colors[0], alpha=0.8)
    bars2 = ax2.bar(x + width/2, energy_consumption, width, label='Energy Consumption', 
                   color=colors[1], alpha=0.8)
    
    ax2.set_xlabel('Algorithm')
    ax2.set_ylabel('Percentage')
    ax2.set_title('CPU Efficiency vs Energy Consumption', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(algorithms)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.0f}', ha='center', va='bottom')
    
    # 3. Performance Scaling
    thread_counts = [1, 2, 4, 8, 16]
    speedup_ideal = thread_counts  # Linear speedup
    speedup_actual = [1, 1.8, 3.2, 5.5, 7.8]  # Realistic speedup
    efficiency = [100, 90, 80, 68.75, 48.75]  # Efficiency percentage
    
    ax3_twin = ax3.twinx()
    
    line1 = ax3.plot(thread_counts, speedup_ideal, 'b--', linewidth=2, 
                    label='Ideal Speedup', marker='o')
    line2 = ax3.plot(thread_counts, speedup_actual, 'r-', linewidth=2, 
                    label='Actual Speedup', marker='s')
    line3 = ax3_twin.plot(thread_counts, efficiency, 'g-', linewidth=2, 
                         label='Efficiency (%)', marker='^')
    
    ax3.set_xlabel('Number of Threads')
    ax3.set_ylabel('Speedup Factor', color='black')
    ax3_twin.set_ylabel('Efficiency (%)', color='green')
    ax3.set_title('Parallel Performance Scaling', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper left')
    
    # 4. CPU Temperature and Throttling
    temp_time = np.arange(0, 120, 2)  # 2 minutes, every 2 seconds
    base_temp = 45
    cpu_temp = base_temp + 15 * np.sin(temp_time * 0.1) + np.random.normal(0, 2, len(temp_time))
    cpu_temp = np.clip(cpu_temp, 30, 85)
    
    # Thermal throttling threshold
    throttle_temp = 75
    
    ax4.plot(temp_time, cpu_temp, 'r-', linewidth=2, label='CPU Temperature')
    ax4.axhline(y=throttle_temp, color='orange', linestyle='--', linewidth=2, 
               label=f'Throttling Threshold ({throttle_temp}°C)')
    ax4.fill_between(temp_time, cpu_temp, throttle_temp, 
                    where=(cpu_temp >= throttle_temp), alpha=0.3, color='red', 
                    label='Throttling Zone')
    
    ax4.set_xlabel('Time (seconds)')
    ax4.set_ylabel('Temperature (°C)')
    ax4.set_title('CPU Temperature Monitoring', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Add system info if available
    try:
        cpu_info = psutil.cpu_percent(interval=1, percpu=True)
        avg_cpu = np.mean(cpu_info)
        fig.suptitle(f'CPU Performance Analysis - Current Usage: {avg_cpu:.1f}%', 
                    fontsize=14, y=1.02)
    except:
        fig.suptitle('CPU Performance Analysis', fontsize=14, y=1.02)
    
    self.save_figure(fig, 'cpu_performance_analysis', ['png', 'svg'])
    plt.show()
    
    return fig
'''
    
    def _create_benchmark_comparison(self, performance_data: Dict[str, Any]) -> str:
        """Create benchmark comparison visualization"""
        return '''
def create_benchmark_comparison():
    """
    Algorithm Benchmark Comparison
    
    Frontend Integration:
    - Interactive benchmark selection
    - Custom benchmark creation
    - Performance trend analysis
    - Export benchmark results
    """
    
    # Sample benchmark data
    algorithms = ['Bubble Sort', 'Selection Sort', 'Insertion Sort', 'Merge Sort', 'Quick Sort', 'Heap Sort']
    datasets = ['Random', 'Sorted', 'Reverse Sorted', 'Nearly Sorted']
    
    # Benchmark results (execution times in milliseconds)
    benchmark_data = {
        'Bubble Sort': [1000, 100, 1200, 800],
        'Selection Sort': [800, 750, 850, 780],
        'Insertion Sort': [900, 50, 950, 200],
        'Merge Sort': [120, 110, 125, 115],
        'Quick Sort': [100, 95, 180, 105],
        'Heap Sort': [140, 135, 145, 138]
    }
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Heatmap of benchmark results
    data_matrix = np.array([benchmark_data[alg] for alg in algorithms])
    
    im = ax1.imshow(data_matrix, cmap='RdYlGn_r', aspect='auto')
    ax1.set_xticks(range(len(datasets)))
    ax1.set_yticks(range(len(algorithms)))
    ax1.set_xticklabels(datasets)
    ax1.set_yticklabels(algorithms)
    ax1.set_title('Benchmark Results Heatmap (ms)', fontweight='bold')
    
    # Add text annotations
    for i in range(len(algorithms)):
        for j in range(len(datasets)):
            text = ax1.text(j, i, f'{data_matrix[i, j]:.0f}',
                           ha="center", va="center", color="black", fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Execution Time (ms)')
    
    # 2. Grouped bar chart
    x = np.arange(len(datasets))
    width = 0.12
    colors = self.get_color_palette("comparison", len(algorithms))
    
    for i, (alg, times) in enumerate(benchmark_data.items()):
        offset = (i - len(algorithms)/2) * width
        bars = ax2.bar(x + offset, times, width, label=alg, color=colors[i], alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 10,
                    f'{height:.0f}', ha='center', va='bottom', fontsize=8)
    
    ax2.set_xlabel('Dataset Type')
    ax2.set_ylabel('Execution Time (ms)')
    ax2.set_title('Benchmark Comparison by Dataset Type', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Performance ranking
    avg_performance = {alg: np.mean(times) for alg, times in benchmark_data.items()}
    sorted_algs = sorted(avg_performance.items(), key=lambda x: x[1])
    
    alg_names, avg_times = zip(*sorted_algs)
    ranks = range(1, len(alg_names) + 1)
    
    bars = ax3.barh(ranks, avg_times, color=colors[:len(alg_names)], alpha=0.8)
    ax3.set_yticks(ranks)
    ax3.set_yticklabels(alg_names)
    ax3.set_xlabel('Average Execution Time (ms)')
    ax3.set_title('Algorithm Performance Ranking', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Add ranking numbers
    for i, (rank, time) in enumerate(zip(ranks, avg_times)):
        ax3.text(time + 20, rank, f'#{i+1}', va='center', fontweight='bold')
    
    # 4. Scalability analysis
    input_sizes = [100, 500, 1000, 5000, 10000]
    
    # Simulate scalability data for top 3 algorithms
    top_3_algs = [alg for alg, _ in sorted_algs[:3]]
    scalability_data = {}
    
    for alg in top_3_algs:
        if 'Merge' in alg or 'Quick' in alg or 'Heap' in alg:
            # O(n log n) algorithms
            scalability_data[alg] = [size * np.log2(size) * 0.01 for size in input_sizes]
        else:
            # O(n²) algorithms
            scalability_data[alg] = [size**2 * 0.0001 for size in input_sizes]
    
    for i, (alg, times) in enumerate(scalability_data.items()):
        ax4.plot(input_sizes, times, label=alg, color=colors[i], 
                linewidth=3, marker='o', markersize=8)
    
    ax4.set_xlabel('Input Size')
    ax4.set_ylabel('Projected Execution Time (ms)')
    ax4.set_title('Scalability Analysis (Top 3 Algorithms)', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    
    plt.tight_layout()
    
    # Add benchmark summary
    best_overall = min(avg_performance.items(), key=lambda x: x[1])
    fig.suptitle(f'Algorithm Benchmark Analysis - Best Overall: {best_overall[0]} ({best_overall[1]:.0f}ms avg)', 
                fontsize=14, y=1.02)
    
    self.save_figure(fig, 'benchmark_comparison', ['png', 'svg'])
    plt.show()
    
    return fig
'''
    
    def _create_realtime_performance_monitor(self, performance_data: Dict[str, Any]) -> str:
        """Create real-time performance monitoring visualization"""
        return '''
def create_realtime_performance_monitor():
    """
    Real-time Performance Monitor
    
    Frontend Integration:
    - WebSocket for real-time data
    - Configurable refresh intervals
    - Alert thresholds and notifications
    - Historical data export
    """
    
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import time
    
    # Create subplot structure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('CPU Usage', 'Memory Usage', 'Execution Time', 'System Metrics'),
        specs=[[{"secondary_y": True}, {"secondary_y": False}],
               [{"secondary_y": False}, {"type": "indicator"}]]
    )
    
    # Generate sample real-time data
    timestamps = pd.date_range(start='now', periods=60, freq='1S')
    
    # CPU data
    cpu_usage = 30 + 20 * np.sin(np.arange(60) * 0.2) + np.random.normal(0, 5, 60)
    cpu_usage = np.clip(cpu_usage, 0, 100)
    
    fig.add_trace(
        go.Scatter(x=timestamps, y=cpu_usage, name='CPU Usage (%)',
                  line=dict(color='red', width=2),
                  hovertemplate='Time: %{x}<br>CPU: %{y:.1f}%<extra></extra>'),
        row=1, col=1
    )
    
    # Memory data
    memory_usage = 1000 + 200 * np.sin(np.arange(60) * 0.15) + np.random.normal(0, 50, 60)
    memory_usage = np.clip(memory_usage, 500, 2000)
    
    fig.add_trace(
        go.Scatter(x=timestamps, y=memory_usage, name='Memory Usage (MB)',
                  line=dict(color='blue', width=2),
                  hovertemplate='Time: %{x}<br>Memory: %{y:.0f}MB<extra></extra>'),
        row=1, col=2
    )
    
    # Execution time data
    algorithms = ['Algorithm A', 'Algorithm B', 'Algorithm C']
    exec_times = {
        'Algorithm A': 100 + 20 * np.sin(np.arange(60) * 0.1) + np.random.normal(0, 10, 60),
        'Algorithm B': 80 + 15 * np.sin(np.arange(60) * 0.12) + np.random.normal(0, 8, 60),
        'Algorithm C': 120 + 25 * np.sin(np.arange(60) * 0.08) + np.random.normal(0, 12, 60)
    }
    
    colors = ['green', 'orange', 'purple']
    for i, (alg, times) in enumerate(exec_times.items()):
        fig.add_trace(
            go.Scatter(x=timestamps, y=times, name=alg,
                      line=dict(color=colors[i], width=2),
                      hovertemplate=f'Time: %{{x}}<br>{alg}: %{{y:.1f}}ms<extra></extra>'),
            row=2, col=1
        )
    
    # System metrics gauge
    current_cpu = np.mean(cpu_usage[-10:])  # Last 10 readings average
    
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=current_cpu,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Current CPU Usage"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text="Real-time Performance Monitor",
        title_x=0.5,
        showlegend=True,
        height=800,
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(
                        args=[{"visible": [True, True, True, True, True, True]}],
                        label="All",
                        method="restyle"
                    ),
                    dict(
                        args=[{"visible": [True, False, False, False, False, False]}],
                        label="CPU Only",
                        method="restyle"
                    ),
                    dict(
                        args=[{"visible": [False, True, False, False, False, False]}],
                        label="Memory Only",
                        method="restyle"
                    )
                ]),
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.01,
                xanchor="left",
                y=1.02,
                yanchor="top"
            ),
        ]
    )
    
    # Save as interactive HTML
    fig.write_html('realtime_performance_monitor.html', include_plotlyjs=True)
    
    # Save configuration for WebSocket integration
    config = {
        "realtime_config": {
            "refresh_interval": 1000,  # milliseconds
            "data_points": 60,
            "alerts": {
                "cpu_threshold": 90,
                "memory_threshold": 1500,
                "execution_time_threshold": 200
            },
            "websocket_url": "ws://localhost:8080/performance",
            "data_retention": "1hour"
        },
        "chart_config": {
            "cpu_color": "red",
            "memory_color": "blue",
            "time_colors": ["green", "orange", "purple"]
        }
    }
    
    with open('realtime_monitor_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    self.plotly_fig = fig
    return fig
'''
    
    def _combine_performance_visualizations(self, visualizations: List[str]) -> str:
        """Combine all performance visualizations"""
        header = '''
"""
COMPREHENSIVE PERFORMANCE ANALYSIS MODULE
Generated by PerformanceVisualizationAgent

Frontend Integration Instructions:
================================

1. REAL-TIME MONITORING:
   - Implement WebSocket connections for live data
   - Use requestAnimationFrame for smooth updates
   - Add configurable refresh intervals
   - Implement data buffering and retention policies

2. ALERT SYSTEM:
   - Set up threshold-based alerts
   - Implement email/SMS notifications
   - Add visual alert indicators
   - Create alert history and management

3. EXPORT CAPABILITIES:
   - PDF reports with charts and analysis
   - CSV data export for spreadsheet analysis
   - JSON configuration export/import
   - API endpoints for data access

4. INTERACTIVE FEATURES:
   - Zoom and pan on time series charts
   - Filter by algorithm or time range
   - Custom benchmark creation
   - Performance comparison tools

5. MOBILE OPTIMIZATION:
   - Responsive chart sizing
   - Touch-friendly interactions
   - Simplified mobile dashboards
   - Offline data caching

Dependencies Required:
- matplotlib >= 3.5.0
- plotly >= 5.0.0
- pandas >= 1.3.0
- numpy >= 1.21.0
- psutil >= 5.8.0 (for system monitoring)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import psutil
import time
import json
from datetime import datetime

# Set performance monitoring style
plt.style.use('seaborn-v0_8')

'''
        
        combined = header
        for viz in visualizations:
            combined += viz + "\n\n"
        
        combined += '''
def generate_all_performance_visualizations(performance_data):
    """
    Master function to generate all performance visualizations
    
    Args:
        performance_data (dict): Dictionary containing performance metrics
    
    Returns:
        dict: Paths to generated visualization files
    """
    
    print("📈 Generating comprehensive performance visualizations...")
    
    # Generate all visualizations
    create_execution_time_analysis()
    create_memory_usage_visualization()
    create_cpu_performance_charts()
    create_benchmark_comparison()
    create_realtime_performance_monitor()
    
    generated_files = {
        "static_charts": [
            "execution_time_analysis.png",
            "memory_usage_analysis.png",
            "cpu_performance_analysis.png",
            "benchmark_comparison.png"
        ],
        "interactive_charts": [
            "realtime_performance_monitor.html"
        ],
        "configuration_files": [
            "realtime_monitor_config.json"
        ],
        "data_exports": [
            "performance_data.json"
        ]
    }
    
    print("✅ All performance visualizations generated successfully!")
    return generated_files

if __name__ == "__main__":
    # Example usage
    sample_performance_data = {
        "execution_time": 150.5,
        "memory_usage": 1024,
        "cpu_usage": 75.2,
        "algorithm_name": "Quick Sort"
    }
    
    generate_all_performance_visualizations(sample_performance_data)
'''
        
        return combined
    
    def _parse_recommendations(self, ai_response: str) -> List[str]:
        """Parse AI recommendations into actionable items"""
        lines = ai_response.split('\n')
        recommendations = []
        for line in lines:
            if line.strip() and (line.startswith('-') or line.startswith('•') or line.startswith('*')):
                recommendations.append(line.strip())
        return recommendations[:5]
    
    def _get_performance_frontend_instructions(self) -> str:
        """Get frontend integration instructions for performance visualizations"""
        return """
FRONTEND INTEGRATION GUIDE FOR PERFORMANCE VISUALIZATIONS
========================================================

1. REAL-TIME CHARTS:
   - Use WebSocket for live data streaming
   - Implement circular buffers for performance
   - Add play/pause controls for data collection
   - Use Canvas rendering for high-frequency updates

2. SYSTEM MONITORING:
   - Request appropriate permissions for system access
   - Implement fallback for restricted environments
   - Add privacy controls for system metrics
   - Cache data locally for offline analysis

3. ALERT MANAGEMENT:
   - Visual indicators for threshold violations
   - Sound notifications for critical alerts
   - Email/SMS integration for remote monitoring
   - Alert history and acknowledgment system

4. PERFORMANCE OPTIMIZATION:
   - Lazy load complex charts
   - Use web workers for data processing
   - Implement chart virtualization for large datasets
   - Add progressive enhancement for slower devices

5. DATA EXPORT:
   - Multiple format support (PNG, PDF, CSV, JSON)
   - Custom report generation
   - Scheduled export automation
   - Data sharing and collaboration features
"""

# Global performance visualizer instance
performance_visualizer = PerformanceVisualizationAgent()
