# backend/visualization_database/visualizations_trees.py
#!/usr/bin/env python3
"""
Tree Algorithm Visualizations
All tree algorithm visualization implementations
"""

import os
from pathlib import Path

def create_tree_visualizations():
    """Create all tree algorithm visualization files"""
    
    base_dir = Path("visualizations/trees")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Tree Traversals
    traversal_code = '''#!/usr/bin/env python3
"""
Binary Tree Traversals - Inorder, Preorder, Postorder
Shows different traversal orders with step-by-step visualization
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def create_traversal_visualization():
    """Create tree traversal visualization"""
    
    # Create sample binary tree
    #       1
    #      / \\
    #     2   3
    #    / \\   \\
    #   4   5   6
    
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    root.left.left = TreeNode(4)
    root.left.right = TreeNode(5)
    root.right.right = TreeNode(6)
    
    # Node positions for visualization
    positions = {
        1: (4, 4),
        2: (2, 3),
        3: (6, 3),
        4: (1, 2),
        5: (3, 2),
        6: (7, 2)
    }
    
    # Different traversals
    def inorder(node, result):
        if node:
            inorder(node.left, result)
            result.append(node.val)
            inorder(node.right, result)
    
    def preorder(node, result):
        if node:
            result.append(node.val)
            preorder(node.left, result)
            preorder(node.right, result)
    
    def postorder(node, result):
        if node:
            postorder(node.left, result)
            postorder(node.right, result)
            result.append(node.val)
    
    # Get traversal orders
    inorder_result = []
    preorder_result = []
    postorder_result = []
    
    inorder(root, inorder_result)
    preorder(root, preorder_result)
    postorder(root, postorder_result)
    
    traversals = {
        'Inorder (Left-Root-Right)': inorder_result,
        'Preorder (Root-Left-Right)': preorder_result,
        'Postorder (Left-Right-Root)': postorder_result
    }
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, (traversal_name, order) in enumerate(traversals.items()):
        ax = axes[idx]
        
        # Draw tree structure
        # Edges
        edges = [(1, 2), (1, 3), (2, 4), (2, 5), (3, 6)]
        for parent, child in edges:
            x1, y1 = positions[parent]
            x2, y2 = positions[child]
            ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2, alpha=0.5)
        
        # Nodes
        for node_val, (x, y) in positions.items():
            # Color nodes based on traversal order
            color_intensity = order.index(node_val) / len(order)
            color = plt.cm.viridis(color_intensity)
            
            circle = plt.Circle((x, y), 0.3, color=color, ec='black', linewidth=2)
            ax.add_patch(circle)
            ax.text(x, y, str(node_val), ha='center', va='center', 
                   fontsize=14, fontweight='bold', color='white')
            
            # Add order number
            ax.text(x + 0.4, y + 0.4, str(order.index(node_val) + 1), 
                   ha='center', va='center', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow'))
        
        # Show traversal order
        order_text = ' ->'.join(map(str, order))
        ax.text(0.5, 0.05, f'Order: {order_text}', transform=ax.transAxes,
                ha='center', va='bottom', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue'))
        
        ax.set_xlim(0, 8)
        ax.set_ylim(1.5, 4.5)
        ax.set_aspect('equal')
        ax.set_title(traversal_name, fontsize=14, fontweight='bold')
        ax.axis('off')
    
    plt.suptitle('Binary Tree Traversals', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    filename = 'traversal_animation.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Tree traversal visualization saved as {filename}")
    return filename

if __name__ == "__main__":
    create_traversal_visualization()
'''
    
    with open(base_dir / "traversal_animation.py", "w") as f:
        f.write(traversal_code)
    
    print(" Created tree visualization files")
