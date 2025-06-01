# backend/visualizations/linked_lists/reverse_linked_list.py
#!/usr/bin/env python3
"""
Reverse Linked List - Step-by-Step Animation
Shows pointer reversal process
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def create_reverse_linked_list_visualization():
    """Create reverse linked list visualization"""
    
    # Sample linked list: 1 -> 2 -> 3 -> 4 -> NULL
    nodes = [1, 2, 3, 4]
    
    # Reversal steps
    steps = [
        {'prev': None, 'curr': 0, 'next': 1, 'description': 'Initial: prev=NULL, curr=1, next=2'},
        {'prev': 0, 'curr': 1, 'next': 2, 'description': 'Step 1: Move pointers, reverse 1->NULL'},
        {'prev': 1, 'curr': 2, 'next': 3, 'description': 'Step 2: Move pointers, reverse 2->1'},
        {'prev': 2, 'curr': 3, 'next': None, 'description': 'Step 3: Move pointers, reverse 3->2'},
        {'prev': 3, 'curr': None, 'next': None, 'description': 'Final: curr=NULL, prev=4 (new head)'}
    ]
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for step_idx, step in enumerate(steps[:6]):
        ax = axes[step_idx]
        
        n_nodes = len(nodes)
        
        # Draw nodes
        for i, value in enumerate(nodes):
            x = i * 2
            y = 0
            
            # Color based on pointer positions
            if i == step.get('prev'):
                color = 'lightblue'
                label = 'prev'
            elif i == step.get('curr'):
                color = 'orange'
                label = 'curr'
            elif i == step.get('next'):
                color = 'lightgreen'
                label = 'next'
            else:
                color = 'lightgray'
                label = ''
            
            # Draw node
            rect = patches.Rectangle((x - 0.3, y - 0.3), 0.6, 0.6, 
                                   linewidth=2, edgecolor='black', facecolor=color)
            ax.add_patch(rect)
            ax.text(x, y, str(value), ha='center', va='center', 
                   fontsize=14, fontweight='bold')
            
            # Add pointer label
            if label:
                ax.text(x, y - 0.7, label, ha='center', va='center', 
                       fontsize=10, fontweight='bold', color='red')
        
        # Draw arrows based on current step
        for i in range(n_nodes - 1):
            x1 = i * 2 + 0.3
            x2 = (i + 1) * 2 - 0.3
            y = 0
            
            # Determine arrow direction
            if step_idx > 0 and i < step.get('curr', 0):
                # Reversed arrows (red)
                ax.arrow(x2, y, x1 - x2, 0, head_width=0.1, head_length=0.1, 
                        fc='red', ec='red', linewidth=2)
            else:
                # Original arrows (blue)
                ax.arrow(x1, y, x2 - x1, 0, head_width=0.1, head_length=0.1, 
                        fc='blue', ec='blue', linewidth=2)
        
        ax.set_xlim(-1, (n_nodes + 1) * 2)
        ax.set_ylim(-1.5, 1)
        ax.set_title(step['description'], fontsize=11, fontweight='bold')
        ax.axis('off')
    
    plt.suptitle('Reverse Linked List: Pointer Manipulation', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    filename = 'reverse_linked_list.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"SUCCESS: Reverse linked list visualization saved as {filename}")
    return filename

if __name__ == "__main__":
    create_reverse_linked_list_visualization()
