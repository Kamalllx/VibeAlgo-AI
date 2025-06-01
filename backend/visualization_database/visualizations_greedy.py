# backend/visualization_database/visualizations_greedy.py
#!/usr/bin/env python3
"""
Greedy Algorithm Visualizations
All greedy algorithm visualization implementations
"""

import os
from pathlib import Path

def create_greedy_visualizations():
    """Create all greedy algorithm visualization files"""
    
    base_dir = Path("visualizations/greedy")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Activity Selection Problem
    activity_selection_code = '''#!/usr/bin/env python3
"""
Activity Selection Problem - Greedy Algorithm
Shows optimal activity scheduling with timeline visualization
"""

import matplotlib.pyplot as plt
import numpy as np

def create_activity_selection_visualization():
    """Create comprehensive activity selection visualization"""
    
    # Activities: (start_time, end_time, name)
    activities = [
        (1, 4, 'A1'), (3, 5, 'A2'), (0, 6, 'A3'), (5, 7, 'A4'), 
        (3, 9, 'A5'), (5, 9, 'A6'), (6, 10, 'A7'), (8, 11, 'A8'), 
        (8, 12, 'A9'), (2, 14, 'A10'), (12, 16, 'A11')
    ]
    
    # Sort by end time (greedy choice)
    sorted_activities = sorted(enumerate(activities), key=lambda x: x[1][1])
    
    selected = []
    last_end_time = 0
    steps = []
    
    for i, (orig_idx, (start, end, name)) in enumerate(sorted_activities):
        compatible = start >= last_end_time
        
        steps.append({
            'sorted_activities': sorted_activities[:i+1],
            'current_activity': (orig_idx, start, end, name),
            'selected': selected.copy(),
            'last_end_time': last_end_time,
            'compatible': compatible,
            'step': i + 1
        })
        
        if compatible:
            selected.append(orig_idx)
            last_end_time = end
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 16))
    
    # Main timeline visualization
    ax1 = plt.subplot(3, 2, (1, 2))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(activities)))
    
    for i, (start, end, name) in enumerate(activities):
        color = colors[i]
        alpha = 0.9 if i in selected else 0.3
        linewidth = 4 if i in selected else 1
        edgecolor = 'red' if i in selected else 'black'
        
        # Draw activity bar
        ax1.barh(i, end - start, left=start, height=0.6, 
                color=color, alpha=alpha, edgecolor=edgecolor, linewidth=linewidth)
        
        # Add activity label
        ax1.text(start + (end - start)/2, i, name, ha='center', va='center', 
                fontweight='bold', fontsize=10, 
                color='white' if i in selected else 'black')
        
        # Add time labels
        ax1.text(start - 0.3, i, str(start), ha='right', va='center', fontsize=8)
        ax1.text(end + 0.3, i, str(end), ha='left', va='center', fontsize=8)
    
    ax1.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Activities', fontsize=12, fontweight='bold')
    ax1.set_title('Activity Selection Problem: Greedy Solution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-1, 17)
    
    # Step-by-step selection process
    ax2 = plt.subplot(3, 2, (3, 4))
    
    # Create table showing selection process
    table_data = []
    headers = ['Step', 'Activity', 'Start', 'End', 'Compatible?', 'Selected?', 'Reason']
    
    current_end = 0
    for i, (orig_idx, (start, end, name)) in enumerate(sorted_activities):
        compatible = start >= current_end
        selected_status = "Success" if compatible else "Failed"
        reason = "Non-overlapping" if compatible else f"Overlaps with prev (end={current_end})"
        
        table_data.append([
            str(i + 1),
            name,
            str(start),
            str(end),
            "Yes" if compatible else "No",
            selected_status,
            reason
        ])
        
        if compatible:
            current_end = end
    
    ax2.axis('tight')
    ax2.axis('off')
    table = ax2.table(cellText=table_data, colLabels=headers,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Color the header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color selected rows
    current_end = 0
    for i, (orig_idx, (start, end, name)) in enumerate(sorted_activities):
        compatible = start >= current_end
        if compatible:
            for j in range(len(headers)):
                table[(i + 1, j)].set_facecolor('#E8F5E8')
            current_end = end
    
    ax2.set_title('Step-by-Step Selection Process', fontsize=12, fontweight='bold')
    
    # Greedy choice visualization
    ax3 = plt.subplot(3, 2, 5)
    
    # Show why sorting by end time is optimal
    end_times = [act[1] for act in activities]
    indices = list(range(len(activities)))
    
    bars = ax3.bar(indices, end_times, color=colors, alpha=0.7)
    
    # Highlight selected activities
    for i in selected:
        bars[i].set_edgecolor('red')
        bars[i].set_linewidth(3)
    
    ax3.set_xlabel('Activity Index')
    ax3.set_ylabel('End Time')
    ax3.set_title('End Times (Greedy Choice Criterion)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.2,
               str(int(height)), ha='center', va='bottom', fontweight='bold')
    
    # Algorithm complexity and summary
    ax4 = plt.subplot(3, 2, 6)
    ax4.axis('off')
    
    summary_text = f"""
ACTIVITY SELECTION PROBLEM - GREEDY ALGORITHM

Total Activities: {len(activities)}
Selected Activities: {len(selected)}
Selection Rate: {len(selected)/len(activities):.1%}

ALGORITHM STEPS:
1. Sort activities by end time: O(n log n)
2. Select first activity
3. For each subsequent activity:
   - If start time >= last selected end time
   - Select the activity
   
TIME COMPLEXITY: O(n log n)
SPACE COMPLEXITY: O(1)

GREEDY CHOICE PROPERTY:
Selecting activity with earliest end time
leaves maximum room for future activities.

OPTIMAL SUBSTRUCTURE:
If we remove selected activity, remaining
problem has same structure.

SELECTED ACTIVITIES: {[activities[i][2] for i in selected]}
TOTAL TIME SPAN: {max(activities[i][1] for i in selected) - min(activities[i][0] for i in selected)} units
"""
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    filename = 'activity_selection.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f" Activity selection visualization saved as {filename}")
    print(f" Selected {len(selected)} out of {len(activities)} activities")
    return filename

if __name__ == "__main__":
    create_activity_selection_visualization()
'''
    
    with open(base_dir / "activity_selection.py", "w") as f:
        f.write(activity_selection_code)
    
    # Huffman Coding Algorithm
    huffman_code = '''#!/usr/bin/env python3
"""
Huffman Coding - Greedy Algorithm for Data Compression
Shows tree construction and encoding process
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import heapq
from collections import Counter, defaultdict
import numpy as np

class Node:
    def __init__(self, char=None, freq=0, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right
    
    def __lt__(self, other):
        return self.freq < other.freq

def create_huffman_visualization():
    """Create comprehensive Huffman coding visualization"""
    
    # Sample text for encoding
    text = "ABRACADABRA"
    
    # Calculate character frequencies
    freq_counter = Counter(text)
    print(f"Text: {text}")
    print(f"Character frequencies: {dict(freq_counter)}")
    
    # Build Huffman tree
    heap = [Node(char, freq) for char, freq in freq_counter.items()]
    heapq.heapify(heap)
    
    tree_steps = []
    step_num = 0
    
    # Store initial state
    tree_steps.append({
        'step': step_num,
        'heap': [(node.char or f"Node{id(node)}", node.freq) for node in heap],
        'description': 'Initial character frequencies'
    })
    
    while len(heap) > 1:
        step_num += 1
        
        # Take two minimum frequency nodes
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        
        # Create new internal node
        merged = Node(freq=left.freq + right.freq, left=left, right=right)
        heapq.heappush(heap, merged)
        
        tree_steps.append({
            'step': step_num,
            'heap': [(node.char or f"Internal({node.freq})", node.freq) for node in heap],
            'merged_nodes': (left.char or f"Node({left.freq})", right.char or f"Node({right.freq})"),
            'new_freq': merged.freq,
            'description': f'Merge {left.char or left.freq} and {right.char or right.freq}'
        })
    
    root = heap[0]
    
    # Generate Huffman codes
    huffman_codes = {}
    
    def generate_codes(node, code=""):
        if node:
            if node.char:  # Leaf node
                huffman_codes[node.char] = code or "0"  # Handle single character case
            else:
                generate_codes(node.left, code + "0")
                generate_codes(node.right, code + "1")
    
    generate_codes(root)
    
    # Encode the text
    encoded_text = ''.join(huffman_codes[char] for char in text)
    
    # Calculate compression ratio
    original_bits = len(text) * 8  # Assuming 8 bits per character
    compressed_bits = len(encoded_text)
    compression_ratio = (1 - compressed_bits / original_bits) * 100
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 16))
    
    # Character frequency chart
    ax1 = plt.subplot(3, 3, 1)
    chars = list(freq_counter.keys())
    freqs = list(freq_counter.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(chars)))
    
    bars = ax1.bar(chars, freqs, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_xlabel('Characters')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Character Frequencies', fontweight='bold')
    
    # Add frequency labels
    for bar, freq in zip(bars, freqs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
               str(freq), ha='center', va='bottom', fontweight='bold')
    
    # Huffman tree construction steps
    ax2 = plt.subplot(3, 3, (2, 3))
    ax2.axis('off')
    
    # Create table showing tree construction
    table_data = []
    for step in tree_steps:
        if step['step'] == 0:
            table_data.append([
                str(step['step']),
                "Initial",
                str(step['heap']),
                step['description']
            ])
        else:
            table_data.append([
                str(step['step']),
                f"{step['merged_nodes'][0]} + {step['merged_nodes'][1]}",
                f"New freq: {step['new_freq']}",
                step['description']
            ])
    
    table = ax2.table(cellText=table_data,
                     colLabels=['Step', 'Action', 'Result', 'Description'],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Color header
    for i in range(4):
        table[(0, i)].set_facecolor('#FF9800')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax2.set_title('Huffman Tree Construction Steps', fontweight='bold', pad=20)
    
    # Huffman codes table
    ax3 = plt.subplot(3, 3, 4)
    ax3.axis('off')
    
    # Sort by code length for better visualization
    sorted_codes = sorted(huffman_codes.items(), key=lambda x: len(x[1]))
    
    code_data = []
    for char, code in sorted_codes:
        freq = freq_counter[char]
        bits_saved = 8 - len(code)  # Assuming 8-bit ASCII
        code_data.append([char, str(freq), code, str(len(code)), str(bits_saved)])
    
    code_table = ax3.table(cellText=code_data,
                          colLabels=['Char', 'Freq', 'Huffman Code', 'Bits', 'Saved'],
                          cellLoc='center', loc='center')
    code_table.auto_set_font_size(False)
    code_table.set_fontsize(10)
    code_table.scale(1, 2)
    
    # Color header
    for i in range(5):
        code_table[(0, i)].set_facecolor('#4CAF50')
        code_table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax3.set_title('Huffman Codes', fontweight='bold')
    
    # Visual tree representation (simplified)
    ax4 = plt.subplot(3, 3, (5, 6))
    
    def draw_tree_node(ax, x, y, node, level=0, pos="root"):
        if not node:
            return
        
        # Draw node
        if node.char:  # Leaf node
            circle = plt.Circle((x, y), 0.3, color='lightgreen', ec='black', linewidth=2)
            ax.add_patch(circle)
            ax.text(x, y, node.char, ha='center', va='center', fontweight='bold')
            ax.text(x, y-0.5, f"f:{node.freq}", ha='center', va='center', fontsize=8)
        else:  # Internal node
            circle = plt.Circle((x, y), 0.3, color='lightblue', ec='black', linewidth=2)
            ax.add_patch(circle)
            ax.text(x, y, str(node.freq), ha='center', va='center', fontweight='bold')
        
        # Draw children
        if node.left or node.right:
            if node.left:
                child_x = x - 1.5 / (level + 1)
                child_y = y - 1
                ax.plot([x, child_x], [y-0.3, child_y+0.3], 'k-', linewidth=2)
                ax.text((x + child_x)/2 - 0.1, (y + child_y)/2, '0', 
                       fontsize=10, fontweight='bold', color='blue')
                draw_tree_node(ax, child_x, child_y, node.left, level+1, "left")
            
            if node.right:
                child_x = x + 1.5 / (level + 1)
                child_y = y - 1
                ax.plot([x, child_x], [y-0.3, child_y+0.3], 'k-', linewidth=2)
                ax.text((x + child_x)/2 + 0.1, (y + child_y)/2, '1', 
                       fontsize=10, fontweight='bold', color='red')
                draw_tree_node(ax, child_x, child_y, node.right, level+1, "right")
    
    draw_tree_node(ax4, 0, 0, root)
    ax4.set_xlim(-3, 3)
    ax4.set_ylim(-4, 1)
    ax4.set_aspect('equal')
    ax4.set_title('Huffman Tree (0=Left, 1=Right)', fontweight='bold')
    ax4.axis('off')
    
    # Encoding example
    ax5 = plt.subplot(3, 3, 7)
    ax5.axis('off')
    
    # Show original vs encoded
    encoding_steps = []
    for i, char in enumerate(text):
        encoding_steps.append(f"{char} -> {huffman_codes[char]}")
    
    encoding_text = f"""
ENCODING EXAMPLE:
Original Text: {text}

Character-by-character encoding:
{chr(10).join(encoding_steps[:6])}
{'...' if len(encoding_steps) > 6 else ''}

Encoded: {encoded_text}

Original size: {original_bits} bits ({len(text)} chars × 8 bits)
Compressed size: {compressed_bits} bits
Compression ratio: {compression_ratio:.1f}%
Space saved: {original_bits - compressed_bits} bits
"""
    
    ax5.text(0.05, 0.95, encoding_text, transform=ax5.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    # Compression comparison
    ax6 = plt.subplot(3, 3, (8, 9))
    
    methods = ['Original\\n(8-bit ASCII)', 'Huffman\\nCoding']
    sizes = [original_bits, compressed_bits]
    colors_comp = ['red', 'green']
    
    bars = ax6.bar(methods, sizes, color=colors_comp, alpha=0.7)
    
    for bar, size in zip(bars, sizes):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 5,
               f'{size} bits', ha='center', va='bottom', fontweight='bold')
    
    ax6.set_ylabel('Size (bits)')
    ax6.set_title(f'Compression Comparison ({compression_ratio:.1f}% reduction)', fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filename = 'huffman_coding.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f" Huffman coding visualization saved as {filename}")
    print(f" Compression: {original_bits} -> {compressed_bits} bits ({compression_ratio:.1f}% reduction)")
    return filename

if __name__ == "__main__":
    create_huffman_visualization()
'''
    
    with open(base_dir / "huffman_coding.py", "w") as f:
        f.write(huffman_code)
    
    print(" Created greedy visualization files")
