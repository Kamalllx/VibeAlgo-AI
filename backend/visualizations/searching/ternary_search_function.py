#!/usr/bin/env python3
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
