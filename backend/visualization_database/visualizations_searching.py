# backend/visualization_database/visualizations_searching.py
#!/usr/bin/env python3
"""
Searching Algorithm Visualizations
All searching algorithm visualization implementations
"""

import os
from pathlib import Path

def create_searching_visualizations():
    """Create all searching algorithm visualization files"""
    
    base_dir = Path("visualizations/searching")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Linear Search Animation
    linear_search_code = '''#!/usr/bin/env python3
"""
Linear Search - Step-by-Step Animation
Shows sequential search through array elements
"""

import matplotlib.pyplot as plt
import numpy as np

def create_linear_search_visualization():
    """Create comprehensive linear search visualization"""
    
    # Sample data
    arr = [23, 45, 12, 67, 34, 89, 56, 78, 90, 43]
    target = 67
    
    # Linear search steps
    steps = []
    found_index = -1
    
    for i in range(len(arr)):
        steps.append({
            'array': arr.copy(),
            'current_index': i,
            'target': target,
            'found': arr[i] == target,
            'comparisons': i + 1
        })
        
        if arr[i] == target:
            found_index = i
            break
    
    # Create visualization
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, step in enumerate(steps[:10]):
        ax = axes[idx]
        
        # Create bars
        bars = ax.bar(range(len(step['array'])), step['array'], 
                     color='lightblue', alpha=0.7, edgecolor='black')
        
        # Highlight current element
        current_idx = step['current_index']
        if step['found']:
            bars[current_idx].set_color('green')
        else:
            bars[current_idx].set_color('orange')
        
        # Highlight searched elements
        for i in range(current_idx):
            bars[i].set_color('lightgray')
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   str(step['array'][i]), ha='center', va='bottom', fontweight='bold')
        
        # Add current comparison
        ax.text(0.5, 0.95, f'Searching for: {target}', transform=ax.transAxes,
                ha='center', va='top', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow'))
        
        # Title with step info
        if step['found']:
            title = f"Step {idx + 1}: FOUND at index {current_idx}!"
        else:
            title = f"Step {idx + 1}: Check index {current_idx} = {step['array'][current_idx]}"
        
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylim(0, max(arr) + 10)
        ax.set_xlabel('Array Index')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(steps), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Linear Search: Find {target} in Array', fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    filename = 'linear_search_animation.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Linear search visualization saved as {filename}")
    return filename

if __name__ == "__main__":
    create_linear_search_visualization()
'''
    
    with open(base_dir / "linear_search_animation.py", "w") as f:
        f.write(linear_search_code)
    
    # Ternary Search Visualization
    ternary_search_code = '''#!/usr/bin/env python3
"""
Ternary Search - Find Maximum in Unimodal Function
Shows divide by 3 approach for optimization
"""

import matplotlib.pyplot as plt
import numpy as np

def create_ternary_search_visualization():
    """Create ternary search visualization for unimodal function"""
    
    # Unimodal function: -(x-5)^2 + 25
    def unimodal_function(x):
        return -(x - 5)**2 + 25
    
    # Ternary search to find maximum
    left, right = 0, 10
    steps = []
    
    while right - left > 0.01:
        m1 = left + (right - left) / 3
        m2 = right - (right - left) / 3
        
        f_m1 = unimodal_function(m1)
        f_m2 = unimodal_function(m2)
        
        steps.append({
            'left': left,
            'right': right,
            'm1': m1,
            'm2': m2,
            'f_m1': f_m1,
            'f_m2': f_m2,
            'search_space': right - left
        })
        
        if f_m1 < f_m2:
            left = m1
        else:
            right = m2
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    x_full = np.linspace(0, 10, 1000)
    y_full = unimodal_function(x_full)
    
    for idx, step in enumerate(steps[:6]):
        ax = axes[idx]
        
        # Plot full function
        ax.plot(x_full, y_full, 'b-', alpha=0.3, linewidth=1, label='f(x)')
        
        # Highlight search space
        search_x = np.linspace(step['left'], step['right'], 100)
        search_y = unimodal_function(search_x)
        ax.plot(search_x, search_y, 'b-', linewidth=3, label='Search Space')
        
        # Mark evaluation points
        ax.plot(step['m1'], step['f_m1'], 'ro', markersize=10, label=f'm1 = {step["m1"]:.2f}')
        ax.plot(step['m2'], step['f_m2'], 'go', markersize=10, label=f'm2 = {step["m2"]:.2f}')
        
        # Mark boundaries
        ax.axvline(step['left'], color='orange', linestyle='--', alpha=0.7)
        ax.axvline(step['right'], color='orange', linestyle='--', alpha=0.7)
        
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 30)
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title(f'Step {idx + 1}: Space = {step["search_space"]:.3f}', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Ternary Search: Find Maximum of Unimodal Function', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    filename = 'ternary_search_function.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f" Ternary search visualization saved as {filename}")
    return filename

if __name__ == "__main__":
    create_ternary_search_visualization()
'''
    
    with open(base_dir / "ternary_search_function.py", "w") as f:
        f.write(ternary_search_code)
    
    print("Created searching visualization files")
