# backend/visualization_database/visualizations/searching/binary_search_animation.py
#!/usr/bin/env python3
"""
Binary Search - Step-by-Step Animation
Shows search space reduction and pointer movement
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

def create_binary_search_visualization():
    """Create comprehensive binary search visualization"""
    
    # Sample data
    arr = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
    target = 17
    
    # Binary search steps
    steps = []
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        steps.append({
            'left': left,
            'right': right,
            'mid': mid,
            'value': arr[mid],
            'comparison': arr[mid],
            'action': 'found' if arr[mid] == target else ('search_right' if arr[mid] < target else 'search_left')
        })
        
        if arr[mid] == target:
            break
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    # Create visualization
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    axes = axes.flatten()
    
    for i, step in enumerate(steps[:9]):  # Show up to 9 steps
        ax = axes[i]
        
        # Create bars for array elements
        bars = ax.bar(range(len(arr)), [1] * len(arr), color='lightgray', alpha=0.7, edgecolor='black')
        
        # Color the search space
        for j in range(step['left'], step['right'] + 1):
            bars[j].set_color('lightblue')
            bars[j].set_alpha(0.8)
        
        # Highlight mid element
        if step['action'] == 'found':
            bars[step['mid']].set_color('green')
        else:
            bars[step['mid']].set_color('orange')
        
        # Add array values as labels
        for j, val in enumerate(arr):
            ax.text(j, 0.5, str(val), ha='center', va='center', fontweight='bold', fontsize=10)
        
        # Add pointers with arrows
        ax.annotate('L', xy=(step['left'], 1.2), ha='center', fontsize=14, fontweight='bold', color='blue')
        ax.annotate('R', xy=(step['right'], 1.2), ha='center', fontsize=14, fontweight='bold', color='red')
        ax.annotate('M', xy=(step['mid'], 1.4), ha='center', fontsize=14, fontweight='bold', color='orange',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow'))
        
        # Add comparison info
        comparison_text = f"arr[{step['mid']}] = {step['value']} vs target = {target}"
        if step['action'] == 'found':
            comparison_text += " ✓ FOUND!"
        elif step['action'] == 'search_right':
            comparison_text += " → Search right half"
        else:
            comparison_text += " ← Search left half"
        
        ax.set_title(f'Step {i+1}: {comparison_text}', fontweight='bold', fontsize=12)
        ax.set_ylim(0, 1.6)
        ax.set_xlim(-0.5, len(arr) - 0.5)
        ax.set_xticks(range(len(arr)))
        ax.set_xticklabels(range(len(arr)))
        ax.set_xlabel('Array Index')
        ax.set_ylabel('Element')
        
        # Add search space size
        search_space = step['right'] - step['left'] + 1
        ax.text(0.02, 0.98, f'Search space: {search_space} elements', transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcyan'))
    
    # Hide unused subplots
    for i in range(len(steps), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Binary Search: Finding {target} in Sorted Array', fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    # Save visualization
    filename = 'binary_search_animation.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f" Binary search visualization saved as {filename}")
    return filename

if __name__ == "__main__":
    create_binary_search_visualization()
