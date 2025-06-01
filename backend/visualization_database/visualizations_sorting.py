# backend/visualization_database/visualizations_sorting.py
#!/usr/bin/env python3
"""
Sorting Algorithm Visualizations
All sorting algorithm visualization implementations
"""

import os
from pathlib import Path

def create_sorting_visualizations():
    """Create all sorting algorithm visualization files"""
    
    base_dir = Path("visualizations/sorting")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Selection Sort Animation
    selection_sort_code = '''#!/usr/bin/env python3
"""
Selection Sort - Step-by-Step Animation
Shows minimum element selection and placement
"""

import matplotlib.pyplot as plt
import numpy as np

def create_selection_sort_visualization():
    """Create comprehensive selection sort visualization"""
    
    # Sample data
    arr = [64, 25, 12, 22, 11, 90, 76, 50]
    original_arr = arr.copy()
    
    # Track selection sort steps
    steps = []
    n = len(arr)
    
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            steps.append({
                'array': arr.copy(),
                'sorted_portion': i,
                'comparing': [min_idx, j],
                'current_min': min_idx,
                'step_type': 'compare',
                'pass': i + 1
            })
            
            if arr[j] < arr[min_idx]:
                min_idx = j
                steps.append({
                    'array': arr.copy(),
                    'sorted_portion': i,
                    'comparing': [min_idx, j],
                    'current_min': min_idx,
                    'step_type': 'new_min',
                    'pass': i + 1
                })
        
        # Swap
        if min_idx != i:
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
            steps.append({
                'array': arr.copy(),
                'sorted_portion': i + 1,
                'comparing': [i, min_idx],
                'current_min': i,
                'step_type': 'swap',
                'pass': i + 1
            })
    
    # Create visualization
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    key_steps = steps[::max(1, len(steps)//12)][:12]
    
    for idx, step in enumerate(key_steps):
        ax = axes[idx]
        
        # Create bars
        bars = ax.bar(range(len(step['array'])), step['array'], 
                     color='lightblue', alpha=0.7, edgecolor='black')
        
        # Color sorted portion
        for i in range(step['sorted_portion']):
            bars[i].set_color('lightgreen')
        
        # Color comparing elements
        if 'comparing' in step:
            for comp_idx in step['comparing']:
                if step['step_type'] == 'new_min':
                    bars[comp_idx].set_color('orange')
                elif step['step_type'] == 'swap':
                    bars[comp_idx].set_color('red')
                else:
                    bars[comp_idx].set_color('yellow')
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   str(step['array'][i]), ha='center', va='bottom', fontweight='bold')
        
        # Title
        title = f"Pass {step['pass']}: {step['step_type'].replace('_', ' ').title()}"
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylim(0, max(original_arr) + 10)
        ax.set_xlabel('Array Index')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(key_steps), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Selection Sort: Find Minimum and Place at Beginning', fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    filename = 'selection_sort_animation.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f" Selection sort visualization saved as {filename}")
    return filename

if __name__ == "__main__":
    create_selection_sort_visualization()
'''
    
    with open(base_dir / "selection_sort_animation.py", "w") as f:
        f.write(selection_sort_code)
    
    print(" Created sorting visualization files")
