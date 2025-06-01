#!/usr/bin/env python3
"""
Maximum Sum Subarray of Size K - Sliding Window
Shows fixed-size sliding window technique
"""

import matplotlib.pyplot as plt
import numpy as np

def create_max_sum_subarray_visualization():
    """Create comprehensive sliding window visualization"""
    
    # Sample array
    arr = [2, 1, 5, 1, 3, 2, 7, 4, 1]
    k = 3  # Window size
    n = len(arr)
    
    # Naive approach (for comparison)
    def max_sum_naive(arr, k):
        max_sum = float('-inf')
        steps = []
        
        for i in range(n - k + 1):
            current_sum = sum(arr[i:i+k])
            steps.append({
                'window_start': i,
                'window_end': i + k - 1,
                'window': arr[i:i+k],
                'sum': current_sum,
                'is_max': current_sum > max_sum
            })
            max_sum = max(max_sum, current_sum)
        
        return max_sum, steps
    
    # Sliding window approach
    def max_sum_sliding_window(arr, k):
        if n < k:
            return -1, []
        
        steps = []
        
        # Calculate sum of first window
        window_sum = sum(arr[:k])
        max_sum = window_sum
        
        steps.append({
            'window_start': 0,
            'window_end': k - 1,
            'window': arr[:k],
            'sum': window_sum,
            'action': 'Initial window',
            'is_max': True
        })
        
        # Slide the window
        for i in range(1, n - k + 1):
            # Remove first element of previous window and add last element of new window
            removed = arr[i - 1]
            added = arr[i + k - 1]
            window_sum = window_sum - removed + added
            
            is_new_max = window_sum > max_sum
            if is_new_max:
                max_sum = window_sum
            
            steps.append({
                'window_start': i,
                'window_end': i + k - 1,
                'window': arr[i:i+k],
                'sum': window_sum,
                'removed': removed,
                'added': added,
                'action': f'Remove {removed}, Add {added}',
                'is_max': is_new_max
            })
        
        return max_sum, steps
    
    # Get results from both approaches
    naive_max, naive_steps = max_sum_naive(arr, k)
    sliding_max, sliding_steps = max_sum_sliding_window(arr, k)
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 16))
    
    # Array visualization
    ax1 = plt.subplot(4, 2, (1, 2))
    
    # Draw array with indices
    bars = ax1.bar(range(n), arr, color='lightblue', alpha=0.7, edgecolor='black')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, arr)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
               str(val), ha='center', va='bottom', fontweight='bold')
        ax1.text(bar.get_x() + bar.get_width()/2., -0.3,
               str(i), ha='center', va='top', fontsize=10)
    
    ax1.set_xlabel('Array Index')
    ax1.set_ylabel('Value')
    ax1.set_title(f'Array: {arr}, Window Size: {k}', fontsize=14, fontweight='bold')
    ax1.set_ylim(-0.5, max(arr) + 1)
    ax1.grid(True, alpha=0.3)
    
    # Naive approach steps
    ax2 = plt.subplot(4, 2, 3)
    ax2.set_title('Naive Approach: Calculate Sum for Each Window', fontweight='bold')
    
    naive_data = []
    for step in naive_steps:
        naive_data.append([
            f"[{step['window_start']}:{step['window_end']}]",
            str(step['window']),
            str(step['sum']),
            "Success" if step['is_max'] else "failure"
        ])
    
    ax2.axis('tight')
    ax2.axis('off')
    naive_table = ax2.table(cellText=naive_data,
                           colLabels=['Window', 'Elements', 'Sum', 'Max?'],
                           cellLoc='center', loc='center')
    naive_table.auto_set_font_size(False)
    naive_table.set_fontsize(10)
    naive_table.scale(1, 1.5)
    
    # Color header and max rows
    for i in range(4):
        naive_table[(0, i)].set_facecolor('#FF5722')
        naive_table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i, step in enumerate(naive_steps):
        if step['is_max']:
            for j in range(4):
                naive_table[(i + 1, j)].set_facecolor('#FFE0B2')
    
    # Sliding window approach steps
    ax3 = plt.subplot(4, 2, 4)
    ax3.set_title('Sliding Window: O(n) Optimization', fontweight='bold')
    
    sliding_data = []
    for step in sliding_steps:
        if 'removed' in step:
            sliding_data.append([
                f"[{step['window_start']}:{step['window_end']}]",
                str(step['window']),
                f"-{step['removed']} +{step['added']}",
                str(step['sum']),
                "Success" if step['is_max'] else ""
            ])
        else:
            sliding_data.append([
                f"[{step['window_start']}:{step['window_end']}]",
                str(step['window']),
                step['action'],
                str(step['sum']),
                "Success"
            ])
    
    ax3.axis('tight')
    ax3.axis('off')
    sliding_table = ax3.table(cellText=sliding_data,
                             colLabels=['Window', 'Elements', 'Operation', 'Sum', 'Max?'],
                             cellLoc='center', loc='center')
    sliding_table.auto_set_font_size(False)
    sliding_table.set_fontsize(9)
    sliding_table.scale(1, 1.5)
    
    # Color header and max rows
    for i in range(5):
        sliding_table[(0, i)].set_facecolor('#4CAF50')
        sliding_table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i, step in enumerate(sliding_steps):
        if step['is_max']:
            for j in range(5):
                sliding_table[(i + 1, j)].set_facecolor('#E8F5E8')
    
    # Visual sliding window animation (show key steps)
    key_steps = [0, 2, 4, len(sliding_steps)-1]  # Show first, middle, and last steps
    
    for subplot_idx, step_idx in enumerate(key_steps):
        ax = plt.subplot(4, 4, 9 + subplot_idx)
        step = sliding_steps[step_idx]
        
        # Draw array
        bars = ax.bar(range(n), arr, color='lightgray', alpha=0.5, edgecolor='black')
        
        # Highlight current window
        for i in range(step['window_start'], step['window_end'] + 1):
            bars[i].set_color('orange')
            bars[i].set_alpha(0.8)
        
        # Add window bracket
        window_x = [step['window_start'] - 0.4, step['window_start'] - 0.4, 
                   step['window_end'] + 0.4, step['window_end'] + 0.4]
        window_y = [max(arr) + 0.5, max(arr) + 0.8, max(arr) + 0.8, max(arr) + 0.5]
        ax.plot(window_x, window_y, 'r-', linewidth=2)
        
        # Add sum label
        ax.text((step['window_start'] + step['window_end']) / 2, max(arr) + 1,
               f"Sum: {step['sum']}", ha='center', va='bottom', fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow'))
        
        # Add operation label for sliding steps
        if 'removed' in step:
            ax.text(0.5, 0.05, step['action'], transform=ax.transAxes,
                   ha='center', va='bottom', fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='lightblue'))
        
        ax.set_title(f'Step {step_idx + 1}', fontsize=10, fontweight='bold')
        ax.set_ylim(0, max(arr) + 1.5)
        ax.set_xticks(range(n))
        ax.set_xticklabels(range(n), fontsize=8)
        
        # Hide y-axis for cleaner look
        ax.set_yticks([])
    
    # Complexity comparison
    ax5 = plt.subplot(4, 2, (7, 8))
    
    # Compare time complexities for different array sizes
    sizes = [10, 50, 100, 500, 1000]
    naive_complexity = [size * 3 for size in sizes]  # O(n*k) approximation
    sliding_complexity = sizes  # O(n)
    
    ax5.plot(sizes, naive_complexity, 'r-o', label='Naive: O(n×k)', linewidth=2, markersize=6)
    ax5.plot(sizes, sliding_complexity, 'g-o', label='Sliding Window: O(n)', linewidth=2, markersize=6)
    
    ax5.set_xlabel('Array Size (n)')
    ax5.set_ylabel('Operations (approximate)')
    ax5.set_title('Time Complexity Comparison', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_yscale('log')
    
    # Algorithm summary
    summary_text = f"""
SLIDING WINDOW TECHNIQUE

PROBLEM: Find maximum sum of subarray of size {k}
ARRAY: {arr}
RESULT: Maximum sum = {sliding_max}

NAIVE APPROACH:
• Calculate sum for each window separately
• Time Complexity: O(n × k)
• Space Complexity: O(1)
• Total operations: {len(naive_steps) * k}

SLIDING WINDOW APPROACH:
• Calculate first window sum
• For subsequent windows: remove first, add last
• Time Complexity: O(n)
• Space Complexity: O(1)  
• Total operations: {len(sliding_steps) + k - 1}

EFFICIENCY GAIN: {((len(naive_steps) * k - len(sliding_steps) - k + 1) / (len(naive_steps) * k) * 100):.1f}%

KEY INSIGHT:
Instead of recalculating the entire sum,
we can reuse the previous sum by:
new_sum = old_sum - removed_element + new_element
"""
    
    # Add summary as text box
    props = dict(boxstyle='round', facecolor='lightcyan', alpha=0.8)
    fig.text(0.02, 0.02, summary_text, fontsize=10, fontfamily='monospace',
            verticalalignment='bottom', bbox=props)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # Make room for summary
    
    filename = 'max_sum_subarray.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f" Maximum sum subarray visualization saved as {filename}")
    return filename

if __name__ == "__main__":
    create_max_sum_subarray_visualization()
