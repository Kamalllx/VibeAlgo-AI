#!/usr/bin/env python3
"""
Next Greater Element - Stack-based Solution
Shows monotonic stack technique
"""

import matplotlib.pyplot as plt
import numpy as np

def create_next_greater_visualization():
    """Create next greater element visualization"""
    
    arr = [4, 5, 2, 25, 3, 1, 8]
    n = len(arr)
    
    # Next greater element algorithm using stack
    stack = []
    result = [-1] * n
    steps = []
    
    for i in range(n):
        # Pop elements smaller than current
        while stack and arr[stack[-1]] < arr[i]:
            idx = stack.pop()
            result[idx] = arr[i]
            steps.append({
                'array': arr.copy(),
                'current_index': i,
                'current_value': arr[i],
                'stack': stack.copy(),
                'result': result.copy(),
                'action': f'Found NGE for {arr[idx]} is {arr[i]}',
                'processing_index': idx
            })
        
        # Push current element
        stack.append(i)
        steps.append({
            'array': arr.copy(),
            'current_index': i,
            'current_value': arr[i],
            'stack': stack.copy(),
            'result': result.copy(),
            'action': f'Push {arr[i]} to stack',
            'processing_index': i
        })
    
    # Create visualization
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    for step_idx, step in enumerate(steps[:9]):
        ax = axes[step_idx]
        
        # Draw array
        bars = ax.bar(range(n), step['array'], color='lightblue', alpha=0.7, edgecolor='black')
        
        # Highlight current element
        if 'current_index' in step:
            bars[step['current_index']].set_color('orange')
        
        # Highlight elements in stack
        for stack_idx in step['stack']:
            bars[stack_idx].set_color('lightgreen')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   str(step['array'][i]), ha='center', va='bottom', fontweight='bold')
            
            # Add NGE result below
            nge_val = step['result'][i]
            nge_text = str(nge_val) if nge_val != -1 else '-'
            ax.text(bar.get_x() + bar.get_width()/2., -2,
                   f'NGE: {nge_text}', ha='center', va='top', fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow'))
        
        # Show stack state
        stack_text = f"Stack: {[step['array'][i] for i in step['stack']]}"
        ax.text(0.02, 0.98, stack_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
        
        # Show action
        ax.text(0.02, 0.85, step['action'], transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
        
        ax.set_ylim(-3, max(step['array']) + 3)
        ax.set_xlabel('Array Index')
        ax.set_ylabel('Value')
        ax.set_title(f'Step {step_idx + 1}', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Next Greater Element using Monotonic Stack', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    filename = 'next_greater_element.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f" Next greater element visualization saved as {filename}")
    return filename

if __name__ == "__main__":
    create_next_greater_visualization()
