#!/usr/bin/env python3
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
