#!/usr/bin/env python3
"""
Two Sum Problem - Hash Table Approach
Shows efficient pair finding using hash map
"""

import matplotlib.pyplot as plt
import numpy as np

def create_two_sum_visualization():
    """Create comprehensive two sum visualization"""
    
    # Sample array and target
    arr = [2, 7, 11, 15, 3, 6]
    target = 9
    
    def two_sum_brute_force(arr, target):
        """Brute force approach: O(n²)"""
        steps = []
        n = len(arr)
        
        for i in range(n):
            for j in range(i + 1, n):
                current_sum = arr[i] + arr[j]
                found = current_sum == target
                
                steps.append({
                    'i': i,
                    'j': j,
                    'arr_i': arr[i],
                    'arr_j': arr[j],
                    'sum': current_sum,
                    'found': found,
                    'comparisons': len(steps) + 1
                })
                
                if found:
                    return [i, j], steps
        
        return None, steps
    
    def two_sum_hash_map(arr, target):
        """Hash map approach: O(n)"""
        hash_map = {}
        steps = []
        
        for i, num in enumerate(arr):
            complement = target - num
            
            if complement in hash_map:
                steps.append({
                    'index': i,
                    'value': num,
                    'complement': complement,
                    'hash_map': hash_map.copy(),
                    'found': True,
                    'result_indices': [hash_map[complement], i],
                    'action': f'Found: {complement} at index {hash_map[complement]}'
                })
                return [hash_map[complement], i], steps
            else:
                hash_map[num] = i
                steps.append({
                    'index': i,
                    'value': num,
                    'complement': complement,
                    'hash_map': hash_map.copy(),
                    'found': False,
                    'action': f'Store {num} at index {i}'
                })
        
        return None, steps
    
    # Get results from both approaches
    brute_result, brute_steps = two_sum_brute_force(arr, target)
    hash_result, hash_steps = two_sum_hash_map(arr, target)
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 16))
    
    # Original array visualization
    ax1 = plt.subplot(4, 2, (1, 2))
    
    bars = ax1.bar(range(len(arr)), arr, color='lightblue', alpha=0.7, edgecolor='black')
    
    # Highlight the solution if found
    if hash_result:
        for idx in hash_result:
            bars[idx].set_color('red')
            bars[idx].set_alpha(0.9)
    
    # Add value and index labels
    for i, (bar, val) in enumerate(zip(bars, arr)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.2,
               str(val), ha='center', va='bottom', fontweight='bold', fontsize=12)
        ax1.text(bar.get_x() + bar.get_width()/2., -0.5,
               f'[{i}]', ha='center', va='top', fontsize=10)
    
    ax1.set_xlabel('Array Index')
    ax1.set_ylabel('Value')
    ax1.set_title(f'Two Sum Problem: Find indices where arr[i] + arr[j] = {target}\nArray: {arr}', 
                 fontsize=14, fontweight='bold')
    ax1.set_ylim(-1, max(arr) + 2)
    ax1.grid(True, alpha=0.3)
    
    # Brute force approach
    ax2 = plt.subplot(4, 2, 3)
    ax2.set_title('Brute Force Approach: O(n²) Time', fontweight='bold')
    
    # Show key steps from brute force
    brute_data = []
    for step in brute_steps[:8]:  # Show first 8 steps
        brute_data.append([
            f'({step["i"]}, {step["j"]})',
            f'{step["arr_i"]} + {step["arr_j"]}',
            str(step['sum']),
            'SUccess' if step['found'] else 'Failure',
            str(step['comparisons'])
        ])
    
    ax2.axis('tight')
    ax2.axis('off')
    brute_table = ax2.table(cellText=brute_data,
                           colLabels=['Indices', 'Calculation', 'Sum', 'Target?', 'Total Ops'],
                           cellLoc='center', loc='center')
    brute_table.auto_set_font_size(False)
    brute_table.set_fontsize(9)
    brute_table.scale(1, 1.5)
    
    # Color header and solution row
    for i in range(5):
        brute_table[(0, i)].set_facecolor('#F44336')
        brute_table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i, step in enumerate(brute_steps[:8]):
        if step['found']:
            for j in range(5):
                brute_table[(i + 1, j)].set_facecolor('#FFCDD2')
    
    # Hash map approach
    ax3 = plt.subplot(4, 2, 4)
    ax3.set_title('Hash Map Approach: O(n) Time', fontweight='bold')
    
    hash_data = []
    for step in hash_steps:
        complement_status = f"Success Found at [{step.get('result_indices', ['?', '?'])[0]}]" if step['found'] else f"Fail Not found"
        hash_data.append([
            f'[{step["index"]}]',
            str(step['value']),
            str(step['complement']),
            complement_status,
            step['action']
        ])
    
    ax3.axis('tight')
    ax3.axis('off')
    hash_table = ax3.table(cellText=hash_data,
                          colLabels=['Index', 'Value', 'Complement', 'In HashMap?', 'Action'],
                          cellLoc='center', loc='center')
    hash_table.auto_set_font_size(False)
    hash_table.set_fontsize(9)
    hash_table.scale(1, 1.5)
    
    # Color header and solution row
    for i in range(5):
        hash_table[(0, i)].set_facecolor('#4CAF50')
        hash_table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i, step in enumerate(hash_steps):
        if step['found']:
            for j in range(5):
                hash_table[(i + 1, j)].set_facecolor('#E8F5E8')
    
    # Hash map state visualization
    ax4 = plt.subplot(4, 2, (5, 6))
    
    # Show hash map building process
    max_steps = len(hash_steps)
    subplot_positions = [(5, 6), (7, 8)]  # We'll use the space for 2 hash map states
    
    # Show initial and final hash map states
    states_to_show = [0, len(hash_steps) - 1] if len(hash_steps) > 1 else [0]
    
    for state_idx, step_idx in enumerate(states_to_show):
        if state_idx >= len(subplot_positions):
            break
            
        step = hash_steps[step_idx]
        hash_map = step['hash_map']
        
        # Create mini subplot for hash map
        y_base = 0.6 - state_idx * 0.4
        
        # Draw hash table buckets
        bucket_width = 0.8 / max(len(hash_map), 1)
        
        for i, (key, value) in enumerate(hash_map.items()):
            x = i * bucket_width
            
            # Draw bucket
            rect = plt.Rectangle((x, y_base), bucket_width * 0.9, 0.15, 
                               facecolor='lightgreen', edgecolor='black', linewidth=1)
            ax4.add_patch(rect)
            
            # Add key-value pair
            ax4.text(x + bucket_width * 0.45, y_base + 0.075, 
                    f'{key}->{value}', ha='center', va='center', 
                    fontsize=10, fontweight='bold')
        
        # Add title for this state
        title = f'Step {step_idx + 1}: ' + ('Initial' if state_idx == 0 else 'Final') + ' HashMap'
        ax4.text(0.4, y_base + 0.2, title, ha='center', va='center', 
                fontsize=12, fontweight='bold')
    
    ax4.set_xlim(0, 0.8)
    ax4.set_ylim(0, 1)
    ax4.set_title('Hash Map Evolution', fontweight='bold')
    ax4.axis('off')
    
    # Algorithm complexity comparison
    ax5 = plt.subplot(4, 2, 7)
    
    # Compare for different array sizes
    sizes = [10, 50, 100, 500, 1000]
    brute_force_ops = [n * (n - 1) // 2 for n in sizes]  # O(n²) combinations
    hash_map_ops = sizes  # O(n) operations
    
    ax5.plot(sizes, brute_force_ops, 'r-o', label='Brute Force: O(n²)', linewidth=2)
    ax5.plot(sizes, hash_map_ops, 'g-o', label='Hash Map: O(n)', linewidth=2)
    
    ax5.set_xlabel('Array Size (n)')
    ax5.set_ylabel('Number of Operations')
    ax5.set_title('Time Complexity Comparison', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_yscale('log')
    
    # Space complexity comparison
    ax6 = plt.subplot(4, 2, 8)
    
    brute_force_space = [1] * len(sizes)  # O(1) space
    hash_map_space = sizes  # O(n) space for hash map
    
    ax6.plot(sizes, brute_force_space, 'r-o', label='Brute Force: O(1)', linewidth=2)
    ax6.plot(sizes, hash_map_space, 'g-o', label='Hash Map: O(n)', linewidth=2)
    
    ax6.set_xlabel('Array Size (n)')
    ax6.set_ylabel('Space Usage (relative)')
    ax6.set_title('Space Complexity Comparison', fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Algorithm summary
    solution_indices = hash_result if hash_result else "No solution"
    solution_values = f"[{arr[hash_result[0]]}, {arr[hash_result[1]]}]" if hash_result else "N/A"
    
    summary_text = f"""
TWO SUM PROBLEM ANALYSIS

INPUT: Array = {arr}, Target = {target}
SOLUTION: Indices = {solution_indices}, Values = {solution_values}

BRUTE FORCE APPROACH:
• Check all pairs (i, j) where i < j
• Time Complexity: O(n²)
• Space Complexity: O(1)
• Operations for n={len(arr)}: {len(brute_steps)}

HASH MAP APPROACH:
• For each element x, look for (target - x) in hash map
• Store seen elements with their indices
• Time Complexity: O(n)
• Space Complexity: O(n)
• Operations for n={len(arr)}: {len(hash_steps)}

EFFICIENCY GAIN:
• Time: {len(brute_steps) / len(hash_steps):.1f}x faster
• Trade-off: Uses O(n) extra space

HASH MAP PRINCIPLE:
1. As we iterate, store: value -> index
2. For current element x, check if (target - x) exists
3. If found, return stored index and current index
4. Hash table provides O(1) average lookup time

KEY INSIGHT:
Instead of checking all pairs, we complement search:
For each number, ask "What number would complete the sum?"
"""
    
    # Add summary
    plt.figtext(0.02, 0.02, summary_text, fontsize=10, fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)
    
    filename = 'two_sum.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Two sum visualization saved as {filename}")
    return filename

if __name__ == "__main__":
    create_two_sum_visualization()
