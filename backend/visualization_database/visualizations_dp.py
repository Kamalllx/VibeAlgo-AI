# backend/visualization_database/visualizations_dp.py
#!/usr/bin/env python3
"""
Dynamic Programming Algorithm Visualizations
All DP algorithm visualization implementations
"""

import os
from pathlib import Path

def create_dp_visualizations():
    """Create all DP algorithm visualization files"""
    
    base_dir = Path("visualizations/dynamic_programming")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Fibonacci with Memoization
    fibonacci_code = '''#!/usr/bin/env python3
"""
Fibonacci with Memoization - DP Table Visualization
Shows recursive tree vs memoized approach
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

def create_fibonacci_visualization():
    """Create Fibonacci memoization visualization"""
    
    n = 6  # Calculate fib(6)
    
    # Regular recursive calls (for comparison)
    def fib_calls(n, depth=0, calls=None):
        if calls is None:
            calls = []
        
        calls.append((n, depth))
        
        if n <= 1:
            return 1, calls
        
        fib_calls(n-1, depth+1, calls)
        fib_calls(n-2, depth+1, calls)
        
        return n, calls
    
    _, recursive_calls = fib_calls(n)
    
    # Memoized approach
    memo = {}
    memo_steps = []
    
    def fib_memo(n):
        if n in memo:
            memo_steps.append({
                'n': n,
                'memo': memo.copy(),
                'action': 'cache_hit',
                'result': memo[n]
            })
            return memo[n]
        
        if n <= 1:
            memo[n] = n
            memo_steps.append({
                'n': n,
                'memo': memo.copy(),
                'action': 'base_case',
                'result': n
            })
            return n
        
        memo_steps.append({
            'n': n,
            'memo': memo.copy(),
            'action': 'computing',
            'result': None
        })
        
        result = fib_memo(n-1) + fib_memo(n-2)
        memo[n] = result
        
        memo_steps.append({
            'n': n,
            'memo': memo.copy(),
            'action': 'stored',
            'result': result
        })
        
        return result
    
    fib_memo(n)
    
    # Create visualization
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 15))
    
    # Recursive call tree
    ax1.set_title('Recursive Fibonacci: Exponential Calls', fontsize=14, fontweight='bold')
    
    # Draw recursive tree
    positions = {}
    level_counts = {}
    
    for call_n, depth in recursive_calls:
        if depth not in level_counts:
            level_counts[depth] = 0
        
        x = level_counts[depth]
        y = -depth
        positions[(call_n, depth, level_counts[depth])] = (x, y)
        level_counts[depth] += 1
        
        # Draw node
        circle = plt.Circle((x, y), 0.3, color='lightcoral', alpha=0.7)
        ax1.add_patch(circle)
        ax1.text(x, y, str(call_n), ha='center', va='center', fontweight='bold')
    
    ax1.set_xlim(-1, max(level_counts.values()) + 1)
    ax1.set_ylim(-max(level_counts.keys()) - 1, 1)
    ax1.set_xlabel('Call Order')
    ax1.set_ylabel('Recursion Depth')
    ax1.grid(True, alpha=0.3)
    
    # Memoization table
    ax2.set_title('Memoization Table: O(n) Space, O(n) Time', fontsize=14, fontweight='bold')
    
    # Show final memo table
    memo_keys = sorted(memo.keys())
    memo_values = [memo[k] for k in memo_keys]
    
    bars = ax2.bar(memo_keys, memo_values, color='lightgreen', alpha=0.7, edgecolor='black')
    
    for i, (key, value) in enumerate(zip(memo_keys, memo_values)):
        ax2.text(key, value + 0.5, f'fib({key})={value}', 
                ha='center', va='bottom', fontweight='bold')
    
    ax2.set_xlabel('n')
    ax2.set_ylabel('fib(n)')
    ax2.set_title('Final Memoization Table')
    ax2.grid(True, alpha=0.3)
    
    # Comparison
    ax3.set_title('Complexity Comparison', fontsize=14, fontweight='bold')
    
    ns = range(1, 11)
    recursive_calls_count = [2**n - 1 for n in ns]  # Approximate
    memo_calls_count = ns  # Linear
    
    ax3.plot(ns, recursive_calls_count, 'r-o', label='Recursive: O(2^n)', linewidth=2)
    ax3.plot(ns, memo_calls_count, 'g-o', label='Memoized: O(n)', linewidth=2)
    
    ax3.set_xlabel('n')
    ax3.set_ylabel('Number of Calls')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filename = 'fibonacci_tree.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f" Fibonacci visualization saved as {filename}")
    return filename

if __name__ == "__main__":
    create_fibonacci_visualization()
'''
    
    with open(base_dir / "fibonacci_tree.py", "w") as f:
        f.write(fibonacci_code)
    
    # 0/1 Knapsack
    knapsack_code = '''#!/usr/bin/env python3
"""
0/1 Knapsack Problem - DP Table Visualization
Shows weight vs value optimization
"""

import matplotlib.pyplot as plt
import numpy as np

def create_knapsack_visualization():
    """Create 0/1 Knapsack visualization"""
    
    # Sample items: (weight, value)
    items = [(2, 3), (3, 4), (4, 5), (5, 6)]
    weights = [item[0] for item in items]
    values = [item[1] for item in items]
    capacity = 8
    n = len(items)
    
    # DP table
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
    
    # Fill DP table
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(
                    dp[i-1][w],  # Don't take item
                    dp[i-1][w - weights[i-1]] + values[i-1]  # Take item
                )
            else:
                dp[i][w] = dp[i-1][w]
    
    # Create visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Items visualization
    ax1.bar(range(len(items)), values, color='lightblue', alpha=0.7, label='Value')
    ax1_twin = ax1.twinx()
    ax1_twin.bar([x + 0.3 for x in range(len(items))], weights, 
                color='orange', alpha=0.7, width=0.3, label='Weight')
    
    ax1.set_xlabel('Item Index')
    ax1.set_ylabel('Value', color='blue')
    ax1_twin.set_ylabel('Weight', color='orange')
    ax1.set_title('Items: Weight vs Value')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    
    # DP table heatmap
    im = ax2.imshow(dp, cmap='YlOrRd', aspect='auto')
    ax2.set_xlabel('Capacity')
    ax2.set_ylabel('Items (0 to n)')
    ax2.set_title(f'DP Table: Max Value for Each (Items, Capacity)')
    
    # Add text annotations
    for i in range(n + 1):
        for j in range(capacity + 1):
            ax2.text(j, i, str(dp[i][j]), ha='center', va='center', 
                    color='white' if dp[i][j] > 5 else 'black', fontweight='bold')
    
    # Colorbar
    plt.colorbar(im, ax=ax2)
    
    # Solution tracing
    ax3.set_title('Optimal Solution Trace')
    
    # Trace back solution
    w = capacity
    selected_items = []
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            selected_items.append(i-1)
            w -= weights[i-1]
    
    selected_items.reverse()
    
    # Show selected items
    colors = ['red' if i in selected_items else 'lightgray' for i in range(len(items))]
    bars = ax3.bar(range(len(items)), values, color=colors, alpha=0.7, edgecolor='black')
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        weight = weights[i]
        status = "SELECTED" if i in selected_items else "NOT SELECTED"
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'W:{weight}\\nV:{height}\\n{status}', 
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    total_weight = sum(weights[i] for i in selected_items)
    total_value = sum(values[i] for i in selected_items)
    
    ax3.text(0.5, 0.95, f'Total Weight: {total_weight}/{capacity}\\nTotal Value: {total_value}', 
            transform=ax3.transAxes, ha='center', va='top', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow'))
    
    ax3.set_xlabel('Item Index')
    ax3.set_ylabel('Value')
    
    plt.tight_layout()
    
    filename = 'knapsack_table.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f" Knapsack visualization saved as {filename}")
    return filename

if __name__ == "__main__":
    create_knapsack_visualization()
'''
    
    with open(base_dir / "knapsack_table.py", "w") as f:
        f.write(knapsack_code)
    
    print(" Created DP visualization files")
