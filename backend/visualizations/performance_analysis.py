# backend/visualizations/performance_analysis.py
#!/usr/bin/env python3
"""
FIXED Performance Analysis and Complexity Visualization Module
Fixes division by zero errors and matplotlib threading issues
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import time
import warnings
warnings.filterwarnings('ignore')

def create_complexity_comparison_plots():
    """Create comprehensive complexity comparison visualizations with error handling"""
    
    try:
        # Input sizes - avoid zero
        n_values = np.array([10, 50, 100, 500, 1000, 5000, 10000])
        
        # Different complexity functions with safety checks
        complexities = {
            'O(1)': np.ones_like(n_values),
            'O(log n)': np.maximum(np.log2(n_values), 1),  # Ensure minimum value
            'O(n)': n_values,
            'O(n log n)': n_values * np.maximum(np.log2(n_values), 1),
            'O(n²)': n_values ** 2,
            'O(2^n)': np.minimum(2 ** (n_values // 1000), 1e6)  # Cap exponential growth
        }
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        # 1. Linear scale comparison
        ax1 = axes[0]
        colors = plt.cm.Set3(np.linspace(0, 1, len(complexities)))
        
        for i, (name, values) in enumerate(complexities.items()):
            if name != 'O(2^n)':  # Skip exponential for linear scale
                ax1.plot(n_values, values, 'o-', label=name, color=colors[i], linewidth=2, markersize=6)
        
        ax1.set_xlabel('Input Size (n)')
        ax1.set_ylabel('Operations Count')
        ax1.set_title('Time Complexity Comparison (Linear Scale)', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Logarithmic scale comparison
        ax2 = axes[1]
        for i, (name, values) in enumerate(complexities.items()):
            # Ensure positive values for log scale
            safe_values = np.maximum(values, 1e-10)
            ax2.loglog(n_values, safe_values, 'o-', label=name, color=colors[i], linewidth=2, markersize=6)
        
        ax2.set_xlabel('Input Size (n)')
        ax2.set_ylabel('Operations Count (log scale)')
        ax2.set_title('Time Complexity Comparison (Log Scale)', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Algorithm performance heatmap
        ax3 = axes[2]
        algorithms = ['Binary Search', 'Linear Search', 'Bubble Sort', 'Quick Sort', 'Merge Sort', 'Hash Lookup']
        scenarios = ['Best Case', 'Average Case', 'Worst Case']
        
        # Performance matrix (lower is better: 1=excellent, 5=poor)
        performance_matrix = np.array([
            [1, 1, 1],  # Binary Search
            [1, 3, 5],  # Linear Search
            [1, 5, 5],  # Bubble Sort
            [2, 2, 5],  # Quick Sort
            [2, 2, 2],  # Merge Sort
            [1, 1, 1],  # Hash Lookup
        ])
        
        sns.heatmap(performance_matrix, annot=True, cmap='RdYlGn_r', 
                    xticklabels=scenarios, yticklabels=algorithms,
                    cbar_kws={'label': 'Performance (1=Best, 5=Worst)'}, ax=ax3)
        ax3.set_title('Algorithm Performance Heatmap', fontweight='bold')
        
        # 4. Space complexity comparison
        ax4 = axes[3]
        space_complexities = {
            'Bubble Sort': 1,
            'Selection Sort': 1,
            'Insertion Sort': 1,
            'Merge Sort': 1000,  # O(n) for n=1000
            'Quick Sort': max(10, 1),    # O(log n) for n=1000, ensure positive
            'Heap Sort': 1
        }
        
        algorithms_list = list(space_complexities.keys())
        space_values = list(space_complexities.values())
        colors_space = plt.cm.viridis(np.linspace(0, 1, len(algorithms_list)))
        
        bars = ax4.bar(algorithms_list, space_values, color=colors_space, alpha=0.8)
        ax4.set_ylabel('Space Usage (relative)')
        ax4.set_title('Space Complexity Comparison', fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars with safety check
        for bar, value in zip(bars, space_values):
            height = bar.get_height()
            if height > 0:  # Ensure positive height
                ax4.text(bar.get_x() + bar.get_width()/2., height + max(space_values)*0.01,
                        f'{value}', ha='center', va='bottom', fontweight='bold')
        
        # 5. Algorithm efficiency radar chart
        ax5 = axes[4]
        ax5.remove()  # Remove to create polar subplot
        ax5 = fig.add_subplot(2, 3, 5, projection='polar')
        
        categories = ['Time Efficiency', 'Space Efficiency', 'Stability', 'Simplicity', 'Adaptability']
        
        # Scores for different algorithms (0-5 scale)
        binary_search_scores = [5, 5, 4, 4, 3]
        quick_sort_scores = [4, 4, 2, 3, 3]
        merge_sort_scores = [4, 2, 5, 3, 2]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Add algorithm scores
        for scores, label, color in [(binary_search_scores, 'Binary Search', 'blue'),
                                    (quick_sort_scores, 'Quick Sort', 'red'),
                                    (merge_sort_scores, 'Merge Sort', 'green')]:
            scores += scores[:1]  # Complete the circle
            ax5.plot(angles, scores, 'o-', linewidth=2, label=label, color=color)
            ax5.fill(angles, scores, alpha=0.25, color=color)
        
        ax5.set_xticks(angles[:-1])
        ax5.set_xticklabels(categories)
        ax5.set_ylim(0, 5)
        ax5.set_title('Algorithm Efficiency Radar Chart', fontweight='bold', pad=20)
        ax5.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        
        # 6. Big O growth visualization
        ax6 = axes[5]
        n_range = np.linspace(1, 100, 1000)
        
        growth_functions = {
            'O(1)': np.ones_like(n_range),
            'O(log n)': np.maximum(np.log2(n_range + 1), 1),  # Ensure positive
            'O(n)': n_range,
            'O(n log n)': n_range * np.maximum(np.log2(n_range + 1), 1),
            'O(n²)': n_range ** 2,
        }
        
        for name, values in growth_functions.items():
            ax6.plot(n_range, values, label=name, linewidth=2)
        
        ax6.set_xlabel('Input Size (n)')
        ax6.set_ylabel('Time/Operations')
        ax6.set_title('Big O Notation Growth Rates', fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim(0, 1000)
        
        plt.tight_layout()
        
        filename = 'complexity_analysis_comprehensive.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()  # Close figure to free memory
        
        print(f"SUCCESS: Comprehensive complexity analysis saved as {filename}")
        return filename
        
    except Exception as e:
        print(f"ERROR: Creating complexity plots failed: {e}")
        return None

def create_algorithm_performance_benchmark():
    """Create real-time performance benchmarking visualization with error handling"""
    
    try:
        def benchmark_algorithm(func, data_sizes, iterations=3):
            """Benchmark an algorithm across different data sizes"""
            times = []
            for size in data_sizes:
                data = list(range(size))
                np.random.shuffle(data)
                
                total_time = 0
                for _ in range(iterations):
                    start_time = time.time()
                    try:
                        func(data)
                    except:
                        pass  # Handle any function errors
                    total_time += time.time() - start_time
                
                times.append(max(total_time / iterations, 1e-6))  # Ensure positive time
            return times
        
        # Simple sorting algorithms for benchmarking
        def bubble_sort(arr):
            arr = arr.copy()
            n = len(arr)
            for i in range(n):
                for j in range(0, n-i-1):
                    if arr[j] > arr[j+1]:
                        arr[j], arr[j+1] = arr[j+1], arr[j]
            return arr
        
        def selection_sort(arr):
            arr = arr.copy()
            for i in range(len(arr)):
                min_idx = i
                for j in range(i+1, len(arr)):
                    if arr[min_idx] > arr[j]:
                        min_idx = j
                arr[i], arr[min_idx] = arr[min_idx], arr[i]
            return arr
        
        # Benchmark data
        data_sizes = [50, 100, 200, 500]
        
        bubble_times = benchmark_algorithm(bubble_sort, data_sizes)
        selection_times = benchmark_algorithm(selection_sort, data_sizes)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Performance comparison
        ax1.plot(data_sizes, bubble_times, 'ro-', label='Bubble Sort', linewidth=2, markersize=8)
        ax1.plot(data_sizes, selection_times, 'go-', label='Selection Sort', linewidth=2, markersize=8)
        
        ax1.set_xlabel('Input Size')
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.set_title('Real-Time Performance Benchmark', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Performance ratio analysis with safety check
        ratios = []
        for b, s in zip(bubble_times, selection_times):
            if s > 0:  # Avoid division by zero
                ratios.append(b / s)
            else:
                ratios.append(1.0)  # Default ratio if selection time is zero
        
        ax2.bar(range(len(data_sizes)), ratios, color='skyblue', alpha=0.7)
        ax2.set_xlabel('Data Size Index')
        ax2.set_ylabel('Bubble Sort / Selection Sort Time Ratio')
        ax2.set_title('Performance Ratio Analysis', fontweight='bold')
        ax2.set_xticks(range(len(data_sizes)))
        ax2.set_xticklabels(data_sizes)
        
        plt.tight_layout()
        
        filename = 'performance_benchmark.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()  # Close figure to free memory
        
        print(f"SUCCESS: Performance benchmark saved as {filename}")
        return filename
        
    except Exception as e:
        print(f"ERROR: Creating benchmark failed: {e}")
        return None

if __name__ == "__main__":
    create_complexity_comparison_plots()
    create_algorithm_performance_benchmark()
