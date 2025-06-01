# backend/visualizations/searching/binary_search_animation.py
#!/usr/bin/env python3
"""
Binary Search Algorithm - Step-by-Step Animation
Shows divide and conquer search in sorted arrays
"""

import matplotlib.pyplot as plt
import numpy as np

def create_binary_search_visualization():
    """Create comprehensive binary search visualization"""
    
    # Sample sorted array and target
    arr = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
    target = 15
    
    # Binary search with step tracking
    def binary_search_steps(arr, target):
        steps = []
        left, right = 0, len(arr) - 1
        step_count = 0
        
        while left <= right:
            mid = (left + right) // 2
            step_count += 1
            
            steps.append({
                'array': arr.copy(),
                'left': left,
                'right': right,
                'mid': mid,
                'target': target,
                'comparing': arr[mid],
                'step': step_count,
                'found': arr[mid] == target,
                'search_space': right - left + 1
            })
            
            if arr[mid] == target:
                return mid, steps
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return -1, steps
    
    result, steps = binary_search_steps(arr, target)
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 16))
    
    # Main algorithm steps
    for step_idx, step in enumerate(steps[:6]):
        ax = plt.subplot(3, 2, step_idx + 1)
        
        # Create bars for array elements
        bars = ax.bar(range(len(arr)), arr, color='lightgray', alpha=0.7, edgecolor='black')
        
        # Color coding for search space
        for i in range(step['left'], step['right'] + 1):
            bars[i].set_color('lightblue')
        
        # Highlight mid element
        bars[step['mid']].set_color('orange')
        
        # Highlight target if found
        if step['found']:
            bars[step['mid']].set_color('green')
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   str(arr[i]), ha='center', va='bottom', fontweight='bold')
        
        # Add search boundaries
        ax.axvline(step['left'] - 0.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.axvline(step['right'] + 0.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        # Add step information
        status = "FOUND!" if step['found'] else f"Target {target} {'>' if target > step['comparing'] else '<'} Mid {step['comparing']}"
        ax.text(0.5, 0.95, f"Step {step['step']}: {status}", transform=ax.transAxes,
                ha='center', va='top', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow'))
        
        ax.text(0.5, 0.05, f"Search Space: {step['search_space']} elements", transform=ax.transAxes,
                ha='center', va='bottom', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.2", facecolor='lightcyan'))
        
        ax.set_ylim(0, max(arr) + 3)
        ax.set_xlabel('Array Index')
        ax.set_ylabel('Value')
        ax.set_title(f'Binary Search Step {step["step"]}', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Binary Search: Find {target} in Sorted Array', fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    filename = 'binary_search_animation.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"SUCCESS: Binary search visualization saved as {filename}")
    return filename

if __name__ == "__main__":
    create_binary_search_visualization()
