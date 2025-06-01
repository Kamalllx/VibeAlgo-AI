# backend/visualization_database/visualizations_graphs.py
#!/usr/bin/env python3
"""
Graph Algorithm Visualizations
All graph algorithm visualization implementations
"""

import os
from pathlib import Path

def create_graph_visualizations():
    """Create all graph algorithm visualization files"""
    
    base_dir = Path("visualizations/graphs")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # BFS Animation
    bfs_code = '''#!/usr/bin/env python3
"""
Breadth-First Search - Level by Level Exploration
Shows BFS traversal using queue
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from collections import deque

def create_bfs_visualization():
    """Create comprehensive BFS visualization"""
    
    # Graph representation
    graph = {
        'A': ['B', 'C'],
        'B': ['A', 'D', 'E'],
        'C': ['A', 'F'],
        'D': ['B'],
        'E': ['B', 'F'],
        'F': ['C', 'E']
    }
    
    # Node positions
    pos = {
        'A': (2, 3), 'B': (1, 2), 'C': (3, 2),
        'D': (0, 1), 'E': (2, 1), 'F': (4, 1)
    }
    
    # BFS algorithm simulation
    start_node = 'A'
    visited = set()
    queue = deque([start_node])
    steps = []
    
    while queue:
        current = queue.popleft()
        if current not in visited:
            visited.add(current)
            
            # Record step
            steps.append({
                'current': current,
                'visited': visited.copy(),
                'queue': list(queue),
                'level': len([n for n in visited if n != current])
            })
            
            # Add neighbors to queue
            for neighbor in graph[current]:
                if neighbor not in visited and neighbor not in queue:
                    queue.append(neighbor)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for step_idx, step in enumerate(steps[:6]):
        ax = axes[step_idx]
        
        # Draw edges
        for node, neighbors in graph.items():
            for neighbor in neighbors:
                x1, y1 = pos[node]
                x2, y2 = pos[neighbor]
                ax.plot([x1, x2], [y1, y2], 'gray', alpha=0.5, linewidth=2)
        
        # Draw nodes
        for node in graph.keys():
            x, y = pos[node]
            
            if node in step['visited']:
                color = 'lightgreen'
            elif node == step['current']:
                color = 'orange'
            elif node in step['queue']:
                color = 'lightblue'
            else:
                color = 'lightgray'
            
            circle = plt.Circle((x, y), 0.2, color=color, ec='black', linewidth=2)
            ax.add_patch(circle)
            ax.text(x, y, node, ha='center', va='center', fontsize=14, fontweight='bold')
        
        # Show queue
        queue_text = f"Queue: {step['queue']}"
        ax.text(0.02, 0.98, queue_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue'))
        
        # Show visited
        visited_text = f"Visited: {sorted(step['visited'])}"
        ax.text(0.02, 0.85, visited_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
        
        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(0.5, 3.5)
        ax.set_aspect('equal')
        ax.set_title(f'Step {step_idx + 1}: Process {step["current"]}', fontweight='bold')
        ax.axis('off')
    
    plt.suptitle('Breadth-First Search: Level-by-Level Exploration', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    filename = 'bfs_animation.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"BFS visualization saved as {filename}")
    return filename

if __name__ == "__main__":
    create_bfs_visualization()
'''
    
    with open(base_dir / "bfs_animation.py", "w") as f:
        f.write(bfs_code)
    
    # DFS Animation
    dfs_code = '''#!/usr/bin/env python3
"""
Depth-First Search - Go Deep First
Shows DFS traversal using stack (recursion)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def create_dfs_visualization():
    """Create comprehensive DFS visualization"""
    
    # Graph representation
    graph = {
        'A': ['B', 'C'],
        'B': ['A', 'D', 'E'],
        'C': ['A', 'F'],
        'D': ['B'],
        'E': ['B', 'F'],
        'F': ['C', 'E']
    }
    
    # Node positions
    pos = {
        'A': (2, 3), 'B': (1, 2), 'C': (3, 2),
        'D': (0, 1), 'E': (2, 1), 'F': (4, 1)
    }
    
    # DFS algorithm simulation
    visited = set()
    stack = []
    steps = []
    
    def dfs_recursive(node, path):
        visited.add(node)
        current_path = path + [node]
        
        steps.append({
            'current': node,
            'visited': visited.copy(),
            'path': current_path.copy(),
            'stack': stack.copy()
        })
        
        for neighbor in sorted(graph[node]):  # Sort for consistent order
            if neighbor not in visited:
                stack.append(neighbor)
                dfs_recursive(neighbor, current_path)
                if stack and stack[-1] == neighbor:
                    stack.pop()
    
    dfs_recursive('A', [])
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for step_idx, step in enumerate(steps[:6]):
        ax = axes[step_idx]
        
        # Draw edges
        for node, neighbors in graph.items():
            for neighbor in neighbors:
                x1, y1 = pos[node]
                x2, y2 = pos[neighbor]
                ax.plot([x1, x2], [y1, y2], 'gray', alpha=0.5, linewidth=2)
        
        # Highlight path edges
        if len(step['path']) > 1:
            for i in range(len(step['path']) - 1):
                node1, node2 = step['path'][i], step['path'][i + 1]
                x1, y1 = pos[node1]
                x2, y2 = pos[node2]
                ax.plot([x1, x2], [y1, y2], 'red', linewidth=4, alpha=0.7)
        
        # Draw nodes
        for node in graph.keys():
            x, y = pos[node]
            
            if node in step['visited']:
                color = 'lightgreen'
            elif node == step['current']:
                color = 'orange'
            else:
                color = 'lightgray'
            
            circle = plt.Circle((x, y), 0.2, color=color, ec='black', linewidth=2)
            ax.add_patch(circle)
            ax.text(x, y, node, ha='center', va='center', fontsize=14, fontweight='bold')
        
        # Show path
        path_text = f"Path: {' -> '.join(step['path'])}"
        ax.text(0.02, 0.98, path_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow'))
        
        # Show visited
        visited_text = f"Visited: {sorted(step['visited'])}"
        ax.text(0.02, 0.85, visited_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
        
        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(0.5, 3.5)
        ax.set_aspect('equal')
        ax.set_title(f'Step {step_idx + 1}: Process {step["current"]}', fontweight='bold')
        ax.axis('off')
    
    plt.suptitle('Depth-First Search: Go Deep First', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    filename = 'dfs_animation.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"DFS visualization saved as {filename}")
    return filename

if __name__ == "__main__":
    create_dfs_visualization()
'''
    
    with open(base_dir / "dfs_animation.py", "w") as f:
        f.write(dfs_code)
    
    print("Created graph visualization files")
