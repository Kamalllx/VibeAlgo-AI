# backend/visualization_database/visualizations_heap.py
#!/usr/bin/env python3
"""
Heap Algorithm Visualizations
All heap algorithm visualization implementations
"""

import os
from pathlib import Path

def create_heap_visualizations():
    """Create all heap algorithm visualization files"""
    
    base_dir = Path("visualizations/heap")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Kth Largest Element
    kth_largest_code = '''#!/usr/bin/env python3
"""
Kth Largest Element - Min Heap Approach
Shows efficient selection using heap data structure
"""

import matplotlib.pyplot as plt
import heapq
import numpy as np

def create_kth_largest_visualization():
    """Create comprehensive Kth largest element visualization"""
    
    # Sample array and k
    arr = [3, 2, 1, 5, 6, 4, 8, 7]
    k = 3
    
    def kth_largest_sorting(arr, k):
        """Approach 1: Sorting"""
        sorted_arr = sorted(arr, reverse=True)
        return sorted_arr[k-1], sorted_arr
    
    def kth_largest_heap(arr, k):
        """Approach 2: Min Heap of size k"""
        heap = []
        steps = []
        
        for i, num in enumerate(arr):
            if len(heap) < k:
                heapq.heappush(heap, num)
                steps.append({
                    'element': num,
                    'heap': heap.copy(),
                    'action': f'Add {num} to heap',
                    'heap_size': len(heap)
                })
            elif num > heap[0]:
                removed = heapq.heapreplace(heap, num)
                steps.append({
                    'element': num,
                    'heap': heap.copy(),
                    'action': f'Replace {removed} with {num}',
                    'heap_size': len(heap),
                    'removed': removed
                })
            else:
                steps.append({
                    'element': num,
                    'heap': heap.copy(),
                    'action': f'Skip {num} (smaller than {heap[0]})',
                    'heap_size': len(heap)
                })
        
        return heap[0] if heap else None, steps
    
    # Get results from both approaches
    sorting_result, sorted_arr = kth_largest_sorting(arr, k)
    heap_result, heap_steps = kth_largest_heap(arr, k)
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 16))
    
    # Original array
    ax1 = plt.subplot(4, 2, (1, 2))
    
    bars = ax1.bar(range(len(arr)), arr, color='lightblue', alpha=0.7, edgecolor='black')
    
    # Highlight kth largest elements
    sorted_indices = sorted(range(len(arr)), key=lambda i: arr[i], reverse=True)
    for i in range(k):
        bars[sorted_indices[i]].set_color('red')
        bars[sorted_indices[i]].set_alpha(0.8)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, arr)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
               str(val), ha='center', va='bottom', fontweight='bold')
    
    ax1.set_xlabel('Array Index')
    ax1.set_ylabel('Value')
    ax1.set_title(f'Original Array: {arr}, Find {k}rd Largest Element', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Sorting approach
    ax2 = plt.subplot(4, 2, 3)
    
    sorted_bars = ax2.bar(range(len(sorted_arr)), sorted_arr, color='lightgreen', alpha=0.7, edgecolor='black')
    
    # Highlight the kth largest
    sorted_bars[k-1].set_color('red')
    sorted_bars[k-1].set_alpha(0.9)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(sorted_bars, sorted_arr)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
               str(val), ha='center', va='bottom', fontweight='bold')
        
        # Mark the kth position
        if i == k-1:
            ax2.text(bar.get_x() + bar.get_width()/2., height/2,
                   f'{k}rd\\nLargest', ha='center', va='center', 
                   fontweight='bold', color='white')
    
    ax2.set_xlabel('Sorted Position')
    ax2.set_ylabel('Value')
    ax2.set_title(f'Sorting Approach: O(n log n)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Heap approach steps
    ax3 = plt.subplot(4, 2, 4)
    ax3.set_title(f'Min Heap Approach: O(n log k)', fontweight='bold')
    
    heap_data = []
    for step in heap_steps:
        heap_data.append([
            str(step['element']),
            str(step['heap']),
            str(step['heap_size']),
            step['action']
        ])
    
    ax3.axis('tight')
    ax3.axis('off')
    heap_table = ax3.table(cellText=heap_data,
                          colLabels=['Element', 'Heap State', 'Size', 'Action'],
                          cellLoc='center', loc='center')
    heap_table.auto_set_font_size(False)
    heap_table.set_fontsize(9)
    heap_table.scale(1, 1.5)
    
    # Color header
    for i in range(4):
        heap_table[(0, i)].set_facecolor('#FF9800')
        heap_table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Visual heap representation for final state
    ax4 = plt.subplot(4, 2, (5, 6))
    
    final_heap = heap_steps[-1]['heap']
    
    # Draw heap as binary tree
    def draw_heap_tree(heap, ax):
        if not heap:
            return
        
        n = len(heap)
        levels = int(np.log2(n)) + 1
        
        # Position nodes
        positions = {}
        for i in range(n):
            level = int(np.log2(i + 1))
            position_in_level = i - (2**level - 1)
            max_positions_in_level = 2**level
            
            x = (position_in_level - max_positions_in_level/2 + 0.5) * (4 / max_positions_in_level)
            y = levels - level - 1
            positions[i] = (x, y)
        
        # Draw edges
        for i in range(n):
            left_child = 2 * i + 1
            right_child = 2 * i + 2
            
            if left_child < n:
                x1, y1 = positions[i]
                x2, y2 = positions[left_child]
                ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2, alpha=0.6)
            
            if right_child < n:
                x1, y1 = positions[i]
                x2, y2 = positions[right_child]
                ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2, alpha=0.6)
        
        # Draw nodes
        for i in range(n):
            x, y = positions[i]
            
            # Root is the minimum (kth largest overall)
            color = 'red' if i == 0 else 'lightblue'
            
            circle = plt.Circle((x, y), 0.3, color=color, ec='black', linewidth=2)
            ax.add_patch(circle)
            ax.text(x, y, str(heap[i]), ha='center', va='center', 
                   fontsize=12, fontweight='bold', color='white' if i == 0 else 'black')
        
        ax.set_xlim(-3, 3)
        ax.set_ylim(-0.5, levels)
        ax.set_aspect('equal')
        ax.axis('off')
    
    draw_heap_tree(final_heap, ax4)
    ax4.set_title(f'Final Min Heap (size {k}): Root = {k}rd Largest = {final_heap[0]}', 
                 fontweight='bold')
    
    # Algorithm comparison
    ax5 = plt.subplot(4, 2, 7)
    
    # Compare time complexities for different array sizes
    sizes = [100, 500, 1000, 5000, 10000]
    k_values = [10, 10, 10, 10, 10]  # Fixed k for comparison
    
    sorting_times = [n * np.log2(n) for n in sizes]
    heap_times = [n * np.log2(k) for n, k in zip(sizes, k_values)]
    
    ax5.plot(sizes, sorting_times, 'r-o', label='Sorting: O(n log n)', linewidth=2)
    ax5.plot(sizes, heap_times, 'g-o', label='Min Heap: O(n log k)', linewidth=2)
    
    ax5.set_xlabel('Array Size (n)')
    ax5.set_ylabel('Time Complexity (relative)')
    ax5.set_title(f'Algorithm Comparison (k = {k})', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_yscale('log')
    
    # Space comparison
    ax6 = plt.subplot(4, 2, 8)
    
    space_sorting = sizes  # O(n) for storing sorted array
    space_heap = [k] * len(sizes)  # O(k) for heap
    
    ax6.plot(sizes, space_sorting, 'r-o', label='Sorting: O(n)', linewidth=2)
    ax6.plot(sizes, space_heap, 'g-o', label='Min Heap: O(k)', linewidth=2)
    
    ax6.set_xlabel('Array Size (n)')
    ax6.set_ylabel('Space Complexity (relative)')
    ax6.set_title(f'Space Comparison (k = {k})', fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Add algorithm summary
    summary_text = f"""
KTH LARGEST ELEMENT PROBLEM

INPUT: Array = {arr}, k = {k}
RESULT: {k}rd largest element = {heap_result}

APPROACH 1 - SORTING:
• Sort array in descending order
• Return element at index k-1
• Time: O(n log n), Space: O(n)

APPROACH 2 - MIN HEAP:
• Maintain min heap of size k
• For each element:
  - If heap size < k: add to heap
  - If element > heap root: replace root
• Heap root is kth largest element
• Time: O(n log k), Space: O(k)

WHEN TO USE HEAP APPROACH:
• When k << n (k much smaller than n)
• When you need streaming/online processing
• When space is constrained

HEAP ADVANTAGE:
For k = {k}, n = {len(arr)}:
Time improvement: {len(arr) * np.log2(len(arr)) / (len(arr) * np.log2(k)):.1f}x faster
Space improvement: {len(arr) / k:.1f}x less space
"""
    
    # Add summary
    plt.figtext(0.02, 0.02, summary_text, fontsize=10, fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    
    filename = 'kth_largest.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f" Kth largest element visualization saved as {filename}")
    return filename

if __name__ == "__main__":
    create_kth_largest_visualization()
'''
    
    with open(base_dir / "kth_largest.py", "w") as f:
        f.write(kth_largest_code)
    
    print("Created heap visualization files")
