# backend/visualization_database/visualizations_union_find.py
#!/usr/bin/env python3
"""
Union-Find Algorithm Visualizations
All union-find (disjoint set) algorithm visualization implementations
"""

import os
from pathlib import Path

def create_union_find_visualizations():
    """Create all union-find algorithm visualization files"""
    
    base_dir = Path("visualizations/union_find")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Union-Find Data Structure
    union_find_code = '''#!/usr/bin/env python3
"""
Union-Find Data Structure - Path Compression and Union by Rank
Shows disjoint set operations with optimizations
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.n = n
        self.operations = []
    
    def find(self, x):
        """Find with path compression"""
        path = []
        original_x = x
        
        # Find root and record path
        while self.parent[x] != x:
            path.append(x)
            x = self.parent[x]
        
        root = x
        
        # Path compression
        for node in path:
            old_parent = self.parent[node]
            self.parent[node] = root
            self.operations.append({
                'type': 'path_compression',
                'node': node,
                'old_parent': old_parent,
                'new_parent': root,
                'path': path.copy(),
                'state': self.parent.copy()
            })
        
        self.operations.append({
            'type': 'find',
            'query': original_x,
            'root': root,
            'path': path + [root],
            'state': self.parent.copy()
        })
        
        return root
    
    def union(self, x, y):
        """Union by rank"""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            self.operations.append({
                'type': 'union_same_set',
                'x': x,
                'y': y,
                'root': root_x,
                'state': self.parent.copy()
            })
            return
        
        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x
        
        self.parent[root_y] = root_x
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1
        
        self.operations.append({
            'type': 'union',
            'x': x,
            'y': y,
            'root_x': root_x,
            'root_y': root_y,
            'new_rank': self.rank[root_x],
            'state': self.parent.copy()
        })

def draw_union_find_state(ax, nodes, parent_array, title):
    """Draw current state of union-find structure"""
    n = len(nodes)
    
    # Calculate positions in a circle
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    radius = 1.5
    positions = [(radius * np.cos(angle), radius * np.sin(angle)) for angle in angles]
    
    # Draw edges (parent relationships)
    for i, node in enumerate(nodes):
        if parent_array[node] != node:  # Not a root
            start_pos = positions[node]
            end_pos = positions[parent_array[node]]
            
            # Draw arrow from child to parent
            dx = end_pos[0] - start_pos[0]
            dy = end_pos[1] - start_pos[1]
            length = np.sqrt(dx**2 + dy**2)
            
            # Normalize and shorten to avoid overlapping with circles
            dx_norm = dx / length * 0.3
            dy_norm = dy / length * 0.3
            
            ax.arrow(start_pos[0] + dx_norm, start_pos[1] + dy_norm,
                    dx - 2*dx_norm, dy - 2*dy_norm,
                    head_width=0.1, head_length=0.1, fc='blue', ec='blue')
    
    # Draw nodes
    for i, node in enumerate(nodes):
        x, y = positions[node]
        
        # Color: roots are green, others are light blue
        color = 'lightgreen' if parent_array[node] == node else 'lightblue'
        
        circle = plt.Circle((x, y), 0.2, color=color, ec='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, str(node), ha='center', va='center', fontsize=12, fontweight='bold')
    
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.axis('off')

def highlight_nodes(ax, nodes, color):
    """Highlight specific nodes"""
    # This is a placeholder - in practice you'd modify the drawing function
    pass

def highlight_path(ax, path, color):
    """Highlight the path during find operation"""
    # This is a placeholder - in practice you'd modify the drawing function
    pass

def draw_union_find_forest(ax, nodes, parent_array, title):
    """Draw the forest structure more clearly"""
    n = len(nodes)
    
    # Find all roots and their trees
    roots = [i for i in range(n) if parent_array[i] == i]
    trees = {root: [] for root in roots}
    
    # Assign each node to its tree
    for node in range(n):
        root = node
        while parent_array[root] != root:
            root = parent_array[root]
        trees[root].append(node)
    
    # Draw each tree separately
    tree_positions = np.linspace(-2, 2, len(roots))
    
    for tree_idx, (root, tree_nodes) in enumerate(trees.items()):
        base_x = tree_positions[tree_idx]
        
        # Simple vertical layout for each tree
        for level, node in enumerate(tree_nodes):
            y = 1 - level * 0.4
            x = base_x + (node - root) * 0.1  # Small horizontal offset
            
            # Draw node
            color = 'lightgreen' if node == root else 'lightblue'
            circle = plt.Circle((x, y), 0.15, color=color, ec='black', linewidth=2)
            ax.add_patch(circle)
            ax.text(x, y, str(node), ha='center', va='center', fontsize=10, fontweight='bold')
            
            # Draw edge to parent (except for root)
            if parent_array[node] != node:
                parent = parent_array[node]
                parent_idx = tree_nodes.index(parent)
                parent_y = 1 - parent_idx * 0.4
                parent_x = base_x + (parent - root) * 0.1
                
                ax.plot([x, parent_x], [y + 0.15, parent_y - 0.15], 'b-', linewidth=2)
    
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2, 1.5)
    ax.set_title(title, fontweight='bold')
    ax.axis('off')

def create_union_find_visualization():
    """Create comprehensive union-find visualization"""
    
    # Create Union-Find instance
    n = 6
    uf = UnionFind(n)
    
    # Perform a series of operations
    operations_sequence = [
        ('union', 0, 1),
        ('union', 2, 3),
        ('union', 4, 5),
        ('union', 0, 2),  # Merge {0,1} with {2,3}
        ('find', 1, None),
        ('union', 0, 4),  # Merge all sets
        ('find', 5, None)
    ]
    
    for op in operations_sequence:
        if op[0] == 'union':
            uf.union(op[1], op[2])
        elif op[0] == 'find':
            uf.find(op[1])
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 16))
    
    # Initial state
    ax1 = plt.subplot(3, 3, 1)
    draw_union_find_state(ax1, list(range(n)), list(range(n)), "Initial State:\\nEach element is its own parent")
    
    # Show key operations
    key_operations = [op for op in uf.operations if op['type'] in ['union', 'find']]
    
    for i, op in enumerate(key_operations[:6]):  # Show first 6 key operations
        ax = plt.subplot(3, 3, i + 2)
        
        if op['type'] == 'union':
            title = f"Union({op['x']}, {op['y']})\\nConnect sets"
        elif op['type'] == 'find':
            title = f"Find({op['query']})\\nRoot = {op['root']}"
        else:
            title = f"Operation {i+1}"
        
        draw_union_find_state(ax, list(range(n)), op['state'], title)
    
    # Final state
    ax_final = plt.subplot(3, 3, (8, 9))
    draw_union_find_forest(ax_final, list(range(n)), uf.parent, "Final Forest Structure")
    
    plt.tight_layout()
    
    # Create second figure for detailed analysis
    fig2 = plt.figure(figsize=(20, 12))
    
    # Operation timeline
    ax_timeline = plt.subplot(2, 2, (1, 2))
    
    timeline_data = []
    for i, op in enumerate(key_operations):
        if op['type'] == 'union':
            timeline_data.append([
                str(i+1),
                f"Union({op['x']}, {op['y']})",
                f"Merge sets containing {op['x']} and {op['y']}",
                "Sets connected"
            ])
        elif op['type'] == 'find':
            timeline_data.append([
                str(i+1),
                f"Find({op['query']})",
                f"Root = {op['root']}",
                f"Path: {' -> '.join(map(str, op['path']))}"
            ])
    
    ax_timeline.axis('tight')
    ax_timeline.axis('off')
    timeline_table = ax_timeline.table(cellText=timeline_data,
                                      colLabels=['Step', 'Operation', 'Action', 'Result'],
                                      cellLoc='center', loc='center')
    timeline_table.auto_set_font_size(False)
    timeline_table.set_fontsize(10)
    timeline_table.scale(1, 1.8)
    
    # Color header
    for i in range(4):
        timeline_table[(0, i)].set_facecolor('#673AB7')
        timeline_table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax_timeline.set_title('Union-Find Operations Timeline', fontweight='bold', pad=20)
    
    # Path compression illustration
    ax_path = plt.subplot(2, 2, 3)
    
    # Create a simple before/after comparison
    ax_path.text(0.25, 0.9, 'Before Path Compression', ha='center', va='center', 
                transform=ax_path.transAxes, fontsize=11, fontweight='bold')
    
    # Chain: 0 -> 1 -> 2 -> 3 (root)
    positions_before = [(0.05, 0.7), (0.15, 0.7), (0.25, 0.7), (0.35, 0.7)]
    for i in range(len(positions_before) - 1):
        x1, y1 = positions_before[i]
        x2, y2 = positions_before[i + 1]
        ax_path.arrow(x1 + 0.02, y1, x2 - x1 - 0.04, 0, 
                     head_width=0.02, head_length=0.01, fc='red', ec='red',
                     transform=ax_path.transAxes)
        
        # Draw node
        circle = patches.Circle((x1, y1), 0.02, color='lightblue', ec='black',
                               transform=ax_path.transAxes)
        ax_path.add_patch(circle)
        ax_path.text(x1, y1, str(i), ha='center', va='center', fontsize=8,
                    transform=ax_path.transAxes)
    
    # Root node
    x, y = positions_before[-1]
    circle = patches.Circle((x, y), 0.02, color='lightgreen', ec='black',
                           transform=ax_path.transAxes)
    ax_path.add_patch(circle)
    ax_path.text(x, y, '3', ha='center', va='center', fontsize=8,
                transform=ax_path.transAxes)
    
    # After path compression
    ax_path.text(0.75, 0.9, 'After Path Compression', ha='center', va='center', 
                transform=ax_path.transAxes, fontsize=11, fontweight='bold')
    
    # All point directly to root
    root_pos = (0.85, 0.7)
    node_positions = [(0.55, 0.7), (0.65, 0.7), (0.75, 0.7)]
    
    for i, (x, y) in enumerate(node_positions):
        # Arrow to root
        ax_path.arrow(x + 0.02, y, root_pos[0] - x - 0.04, 0,
                     head_width=0.02, head_length=0.01, fc='green', ec='green',
                     transform=ax_path.transAxes)
        
        # Node
        circle = patches.Circle((x, y), 0.02, color='lightblue', ec='black',
                               transform=ax_path.transAxes)
        ax_path.add_patch(circle)
        ax_path.text(x, y, str(i), ha='center', va='center', fontsize=8,
                    transform=ax_path.transAxes)
    
    # Root
    circle = patches.Circle(root_pos, 0.02, color='lightgreen', ec='black',
                           transform=ax_path.transAxes)
    ax_path.add_patch(circle)
    ax_path.text(root_pos[0], root_pos[1], '3', ha='center', va='center', fontsize=8,
                transform=ax_path.transAxes)
    
    ax_path.set_title('Path Compression Optimization', fontweight='bold')
    ax_path.axis('off')
    
    # Algorithm complexity comparison
    ax_complexity = plt.subplot(2, 2, 4)
    
    operations_count = [10, 100, 1000, 10000]
    without_optimization = [n * np.log(n) for n in operations_count]
    with_optimization = [n * 1.1 for n in operations_count]  # Nearly constant amortized
    
    ax_complexity.plot(operations_count, without_optimization, 'r-o', 
                      label='Without Optimization: O(log n)', linewidth=2)
    ax_complexity.plot(operations_count, with_optimization, 'g-o', 
                      label='With Path Compression: O(a(n))', linewidth=2)
    
    ax_complexity.set_xlabel('Number of Operations')
    ax_complexity.set_ylabel('Time per Operation (relative)')
    ax_complexity.set_title('Union-Find Performance Comparison', fontweight='bold')
    ax_complexity.legend()
    ax_complexity.grid(True, alpha=0.3)
    ax_complexity.set_xscale('log')
    ax_complexity.set_yscale('log')
    
    plt.tight_layout()
    
    # Algorithm summary
    summary_text = f"""
UNION-FIND DATA STRUCTURE

OPERATIONS PERFORMED:
• Union(0,1): Connect elements 0 and 1
• Union(2,3): Connect elements 2 and 3  
• Union(4,5): Connect elements 4 and 5
• Union(0,2): Merge sets {{0,1}} and {{2,3}}
• Find(1): Find root of element 1
• Union(0,4): Merge all sets into one
• Find(5): Find root of element 5

KEY OPTIMIZATIONS:

1. UNION BY RANK:
   • Always attach smaller tree under root of larger tree
   • Keeps trees shallow
   • Limits tree height to O(log n)

2. PATH COMPRESSION:
   • During find, make all nodes point directly to root
   • Flattens the tree structure
   • Amortized time complexity: O(a(n))
   
   a(n) = Inverse Ackermann function (effectively constant)

APPLICATIONS:
• Cycle detection in graphs
• Kruskal's MST algorithm
• Connected components
• Percolation problems
• Network connectivity

TIME COMPLEXITY:
• Without optimization: O(n) per operation
• With optimization: O(a(n)) amortized per operation

SPACE COMPLEXITY: O(n) for parent and rank arrays
"""
    
    # Add summary to first figure
    fig.text(0.02, 0.02, summary_text, fontsize=10, fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    plt.figure(fig.number)  # Switch back to first figure
    plt.subplots_adjust(bottom=0.35)  # Make room for summary
    
    # Save both figures
    plt.figure(fig.number)
    filename1 = 'union_find_operations.png'
    plt.savefig(filename1, dpi=300, bbox_inches='tight')
    
    plt.figure(fig2.number)
    filename2 = 'union_find_analysis.png'
    plt.savefig(filename2, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    print(f"Union-Find visualizations saved as {filename1} and {filename2}")
    return [filename1, filename2]

if __name__ == "__main__":
    create_union_find_visualization()
'''
    
    with open(base_dir / "union_find_operations.py", "w") as f:
        f.write(union_find_code)
    
    print("Created union-find visualization files")
