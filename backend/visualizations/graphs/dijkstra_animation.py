# backend/visualization_database/visualizations/graphs/dijkstra_animation.py
#!/usr/bin/env python3
"""
Dijkstra's Algorithm - Step-by-Step Animation
Shows shortest path finding with distance updates
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import time
from matplotlib.animation import FuncAnimation
import os

def create_dijkstra_visualization():
    """Create comprehensive Dijkstra's algorithm visualization"""
    
    # Sample graph data
    nodes = ['A', 'B', 'C', 'D', 'E', 'F']
    edges = [
        ('A', 'B', 4), ('A', 'C', 2), ('B', 'C', 1), ('B', 'D', 5),
        ('C', 'D', 8), ('C', 'E', 10), ('D', 'E', 2), ('D', 'F', 6), ('E', 'F', 3)
    ]
    
    # Node positions for visualization
    pos = {
        'A': (0, 2), 'B': (2, 3), 'C': (2, 1), 
        'D': (4, 2), 'E': (6, 1), 'F': (6, 3)
    }
    
    # Dijkstra algorithm steps
    steps = [
        {
            'current': 'A', 'visited': set(), 'distances': {'A': 0, 'B': float('inf'), 'C': float('inf'), 'D': float('inf'), 'E': float('inf'), 'F': float('inf')},
            'previous': {}, 'step_description': 'Initialize: Start from A, all distances = ∞'
        },
        {
            'current': 'A', 'visited': {'A'}, 'distances': {'A': 0, 'B': 4, 'C': 2, 'D': float('inf'), 'E': float('inf'), 'F': float('inf')},
            'previous': {'B': 'A', 'C': 'A'}, 'step_description': 'Process A: Update neighbors B(4) and C(2)'
        },
        {
            'current': 'C', 'visited': {'A', 'C'}, 'distances': {'A': 0, 'B': 3, 'C': 2, 'D': 10, 'E': 12, 'F': float('inf')},
            'previous': {'B': 'C', 'C': 'A', 'D': 'C', 'E': 'C'}, 'step_description': 'Process C: Better path to B(3), update D(10), E(12)'
        },
        {
            'current': 'B', 'visited': {'A', 'C', 'B'}, 'distances': {'A': 0, 'B': 3, 'C': 2, 'D': 8, 'E': 12, 'F': float('inf')},
            'previous': {'B': 'C', 'C': 'A', 'D': 'B', 'E': 'C'}, 'step_description': 'Process B: Better path to D(8)'
        },
        {
            'current': 'D', 'visited': {'A', 'C', 'B', 'D'}, 'distances': {'A': 0, 'B': 3, 'C': 2, 'D': 8, 'E': 10, 'F': 14},
            'previous': {'B': 'C', 'C': 'A', 'D': 'B', 'E': 'D', 'F': 'D'}, 'step_description': 'Process D: Better path to E(10), update F(14)'
        },
        {
            'current': 'E', 'visited': {'A', 'C', 'B', 'D', 'E'}, 'distances': {'A': 0, 'B': 3, 'C': 2, 'D': 8, 'E': 10, 'F': 13},
            'previous': {'B': 'C', 'C': 'A', 'D': 'B', 'E': 'D', 'F': 'E'}, 'step_description': 'Process E: Better path to F(13)'
        }
    ]
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 12))
    
    # Main graph visualization
    ax_main = plt.subplot2grid((3, 4), (0, 0), colspan=3, rowspan=2)
    
    # Distance table
    ax_table = plt.subplot2grid((3, 4), (0, 3))
    
    # Step description
    ax_desc = plt.subplot2grid((3, 4), (1, 3))
    
    # Algorithm complexity
    ax_complexity = plt.subplot2grid((3, 4), (2, 0), colspan=2)
    
    # Priority queue visualization
    ax_queue = plt.subplot2grid((3, 4), (2, 2), colspan=2)
    
    for step_idx, step in enumerate(steps):
        # Clear all axes
        for ax in [ax_main, ax_table, ax_desc, ax_complexity, ax_queue]:
            ax.clear()
        
        # === Main Graph Visualization ===
        # Draw edges
        for edge in edges:
            start, end, weight = edge
            x1, y1 = pos[start]
            x2, y2 = pos[end]
            
            # Color edge if it's part of shortest path
            edge_color = 'red' if (start in step['previous'].values() and end in step['previous']) or (end in step['previous'].values() and start in step['previous']) else 'gray'
            edge_width = 3 if edge_color == 'red' else 1
            
            ax_main.plot([x1, x2], [y1, y2], color=edge_color, linewidth=edge_width, alpha=0.7)
            
            # Add edge weight
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax_main.text(mid_x, mid_y, str(weight), fontsize=10, ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='black'))
        
        # Draw nodes
        for node in nodes:
            x, y = pos[node]
            
            # Node color based on status
            if node in step['visited']:
                color = 'lightgreen'
                edge_color = 'darkgreen'
            elif node == step['current']:
                color = 'orange'
                edge_color = 'darkorange'
            else:
                color = 'lightblue'
                edge_color = 'blue'
            
            # Draw node circle
            circle = plt.Circle((x, y), 0.3, color=color, ec=edge_color, linewidth=3)
            ax_main.add_patch(circle)
            
            # Node label
            ax_main.text(x, y, node, ha='center', va='center', fontsize=14, fontweight='bold')
            
            # Distance label
            dist = step['distances'][node]
            dist_text = str(dist) if dist != float('inf') else '∞'
            ax_main.text(x, y - 0.6, f'd={dist_text}', ha='center', va='center', fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))
        
        ax_main.set_xlim(-1, 7)
        ax_main.set_ylim(0, 4)
        ax_main.set_aspect('equal')
        ax_main.set_title(f"Dijkstra's Algorithm - Step {step_idx + 1}", fontsize=16, fontweight='bold')
        ax_main.axis('off')
        
        # === Distance Table ===
        table_data = []
        for node in nodes:
            dist = step['distances'][node]
            dist_str = str(dist) if dist != float('inf') else '∞'
            prev = step['previous'].get(node, '-')
            visited = '✓' if node in step['visited'] else ''
            table_data.append([node, dist_str, prev, visited])
        
        ax_table.axis('tight')
        ax_table.axis('off')
        table = ax_table.table(cellText=table_data,
                              colLabels=['Node', 'Distance', 'Previous', 'Visited'],
                              cellLoc='center',
                              loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Color current node row
        for i, node in enumerate(nodes):
            if node == step['current']:
                for j in range(4):
                    table[(i + 1, j)].set_facecolor('orange')
            elif node in step['visited']:
                for j in range(4):
                    table[(i + 1, j)].set_facecolor('lightgreen')
        
        ax_table.set_title('Distance Table', fontweight='bold')
        
        # === Step Description ===
        ax_desc.text(0.5, 0.5, step['step_description'], ha='center', va='center',
                    fontsize=11, wrap=True, transform=ax_desc.transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow'))
        ax_desc.set_title('Current Step', fontweight='bold')
        ax_desc.axis('off')
        
        # === Complexity Analysis ===
        if step_idx == 0:  # Show complexity info on first step
            complexity_text = """
Time Complexity: O((V + E) log V)
Space Complexity: O(V)

V = Vertices, E = Edges
Uses priority queue for efficiency
            """
            ax_complexity.text(0.1, 0.5, complexity_text, ha='left', va='center',
                              fontsize=11, transform=ax_complexity.transAxes,
                              bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcyan'))
        else:
            # Show operations count
            operations = f"Step {step_idx + 1}/{len(steps)}\nNodes processed: {len(step['visited'])}/{len(nodes)}"
            ax_complexity.text(0.1, 0.5, operations, ha='left', va='center',
                              fontsize=12, transform=ax_complexity.transAxes,
                              bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcyan'))
        
        ax_complexity.set_title('Algorithm Info', fontweight='bold')
        ax_complexity.axis('off')
        
        # === Priority Queue Visualization ===
        unvisited = [(node, step['distances'][node]) for node in nodes if node not in step['visited']]
        unvisited.sort(key=lambda x: x[1])
        
        if unvisited:
            queue_text = "Priority Queue (min-heap):\n"
            for i, (node, dist) in enumerate(unvisited[:5]):  # Show top 5
                dist_str = str(dist) if dist != float('inf') else '∞'
                queue_text += f"{i+1}. {node} (d={dist_str})\n"
            
            ax_queue.text(0.1, 0.8, queue_text, ha='left', va='top',
                         fontsize=10, transform=ax_queue.transAxes,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor='lightpink'))
        
        ax_queue.set_title('Priority Queue', fontweight='bold')
        ax_queue.axis('off')
        
        plt.tight_layout()
        
        # Save each step
        filename = f'dijkstra_step_{step_idx + 1}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f" Saved step {step_idx + 1}: {filename}")
        
        # Also save as current step for real-time viewing
        plt.savefig('dijkstra_current.png', dpi=300, bbox_inches='tight')
        
        if step_idx < len(steps) - 1:
            plt.pause(2)  # Pause between steps
    
    plt.show()
    print(" Dijkstra visualization completed!")
    
    return [f'dijkstra_step_{i + 1}.png' for i in range(len(steps))]

if __name__ == "__main__":
    generated_files = create_dijkstra_visualization()
    print(f" Generated files: {generated_files}")
