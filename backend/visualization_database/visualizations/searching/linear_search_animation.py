#!/usr/bin/env python3
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
