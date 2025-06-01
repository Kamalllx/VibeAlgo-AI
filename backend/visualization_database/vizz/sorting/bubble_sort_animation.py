# backend/visualization_database/visualizations/sorting/bubble_sort_animation.py
#!/usr/bin/env python3
"""
Bubble Sort - Step-by-Step Animation
Shows element comparisons and swaps
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

def create_bubble_sort_visualization():
    """Create comprehensive bubble sort visualization"""
    
    # Sample data
    arr = [64, 34, 25, 12, 22, 11, 90, 5]
    original_arr = arr.copy()
    
    # Track all bubble sort steps
    steps = []
    n = len(arr)
    
    for i in range(n):
        for j in range(0, n - i - 1):
            # Record comparison step
            steps.append({
                'array': arr.copy(),
                'comparing': [j, j + 1],
                'swapped': False,
                'pass': i + 1,
                'step_type': 'compare'
            })
            
            # If swap needed
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                steps.append({
                    'array': arr.copy(),
                    'comparing': [j, j + 1],
                    'swapped': True,
                    'pass': i + 1,
                    'step_type': 'swap'
                })
    
    # Create visualization showing key steps
    key_steps = []
    step_interval = max(1, len(steps) // 12)  # Show ~12 key steps
    for i in range(0, len(steps), step_interval):
        key_steps.append(steps[i])
    if steps[-1] not in key_steps:
        key_steps.append(steps[-1])  # Always include final step
    
    # Create subplots
    rows = 3
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
    axes = axes.flatten()
    
    for idx, step in enumerate(key_steps[:12]):
        ax = axes[idx]
        
        # Create bars
        bars = ax.bar(range(len(step['array'])), step['array'], 
                     color='lightblue', alpha=0.7, edgecolor='black')
        
        # Color the comparing elements
        if 'comparing' in step:
            for comp_idx in step['comparing']:
                if step['step_type'] == 'swap' and step['swapped']:
                    bars[comp_idx].set_color('red')  # Red for swapped
                else:
                    bars[comp_idx].set_color('orange')  # Orange for comparing
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   str(step['array'][i]), ha='center', va='bottom', fontweight='bold')
        
        # Add arrows for comparison
        if 'comparing' in step and len(step['comparing']) == 2:
            idx1, idx2 = step['comparing']
            y_arrow = max(step['array']) + 15
            
            if step['step_type'] == 'swap' and step['swapped']:
                # Show swap arrows
                ax.annotate('', xy=(idx2, y_arrow), xytext=(idx1, y_arrow),
                           arrowprops=dict(arrowstyle='<->', color='red', lw=2))
                ax.text((idx1 + idx2) / 2, y_arrow + 5, 'SWAP', ha='center', 
                       fontsize=10, fontweight='bold', color='red')
            else:
                # Show comparison arrows
                ax.annotate('', xy=(idx2, y_arrow), xytext=(idx1, y_arrow),
                           arrowprops=dict(arrowstyle='<->', color='orange', lw=2))
                ax.text((idx1 + idx2) / 2, y_arrow + 5, 'COMPARE', ha='center',
                       fontsize=10, fontweight='bold', color='orange')
        
        # Title with step info
        title = f"Pass {step['pass']}: "
        if step['step_type'] == 'swap' and step['swapped']:
            title += f"Swap {step['array'][step['comparing'][1]]} and {step['array'][step['comparing'][0]]}"
        else:
            comp_vals = [step['array'][i] for i in step['comparing']]
            title += f"Compare {comp_vals[0]} and {comp_vals[1]}"
        
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylim(0, max(original_arr) + 30)
        ax.set_xlabel('Array Index')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(key_steps), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Bubble Sort: Step-by-Step Animation', fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    # Save visualization
    filename = 'bubble_sort_animation.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f" Bubble sort visualization saved as {filename}")
    
    # Also create complexity comparison
    create_sorting_complexity_comparison()
    
    return filename

def create_sorting_complexity_comparison():
    """Create sorting algorithms complexity comparison"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Input sizes
    n = np.logspace(1, 4, 100)
    
    # Sorting complexities
    algorithms = {
        'Bubble Sort: O(n²)': n**2 / 1000,
        'Selection Sort: O(n²)': n**2 / 1000,
        'Insertion Sort: O(n²)': n**2 / 1200,  # Slightly better
        'Merge Sort: O(n log n)': n * np.log2(n),
        'Quick Sort: O(n log n)': n * np.log2(n) * 0.9,  # Slightly better
        'Heap Sort: O(n log n)': n * np.log2(n) * 1.1   # Slightly worse
    }
    
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
    
    # Linear scale
    for i, (name, values) in enumerate(algorithms.items()):
        style = '--' if 'O(n²)' in name else '-'
        width = 3 if 'Bubble' in name else 2
        ax1.plot(n, values, label=name, color=colors[i], linestyle=style, linewidth=width)
    
    ax1.set_xlabel('Input Size (n)')
    ax1.set_ylabel('Operations')
    ax1.set_title('Sorting Algorithms Complexity (Linear Scale)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Log scale
    for i, (name, values) in enumerate(algorithms.items()):
        style = '--' if 'O(n²)' in name else '-'
        width = 3 if 'Bubble' in name else 2
        ax2.loglog(n, values, label=name, color=colors[i], linestyle=style, linewidth=width)
    
    ax2.set_xlabel('Input Size (n) - Log Scale')
    ax2.set_ylabel('Operations - Log Scale')
    ax2.set_title('Sorting Algorithms Complexity (Log Scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sorting_complexity_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Sorting complexity comparison saved")

if __name__ == "__main__":
    create_bubble_sort_visualization()
