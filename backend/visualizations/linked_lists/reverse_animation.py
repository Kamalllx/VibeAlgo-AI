#!/usr/bin/env python3
"""
Reverse Linked List - Step-by-Step Pointer Manipulation
Shows pointer reversal process
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def create_reverse_visualization():
    """Create linked list reversal visualization"""
    
    # Initial linked list: 1 -> 2 -> 3 -> 4 -> 5 -> NULL
    nodes = [1, 2, 3, 4, 5]
    
    # Reversal steps
    steps = [
        {
            'nodes': [1, 2, 3, 4, 5],
            'prev': None,
            'curr': 0,
            'next': 1,
            'description': 'Initial state: prev=NULL, curr=1, next=2'
        },
        {
            'nodes': [1, 2, 3, 4, 5],
            'prev': None,
            'curr': 0,
            'next': 1,
            'description': 'Step 1: Store next, reverse curr->prev'
        },
        {
            'nodes': [1, 2, 3, 4, 5],
            'prev': 0,
            'curr': 1,
            'next': 2,
            'description': 'Step 2: Move pointers forward'
        },
        {
            'nodes': [1, 2, 3, 4, 5],
            'prev': 1,
            'curr': 2,
            'next': 3,
            'description': 'Step 3: Continue reversal process'
        },
        {
            'nodes': [1, 2, 3, 4, 5],
            'prev': 2,
            'curr': 3,
            'next': 4,
            'description': 'Step 4: Reverse connection 3->2'
        },
        {
            'nodes': [1, 2, 3, 4, 5],
            'prev': 3,
            'curr': 4,
            'next': None,
            'description': 'Step 5: Final reversal 5->4'
        }
    ]
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for step_idx, step in enumerate(steps):
        ax = axes[step_idx]
        
        n_nodes = len(step['nodes'])
        
        # Draw nodes
        for i, value in enumerate(step['nodes']):
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
        
        # Draw arrows (original direction)
        for i in range(n_nodes - 1):
            x1 = i * 2 + 0.3
            x2 = (i + 1) * 2 - 0.3
            y = 0
            
            # Determine arrow style based on reversal progress
            if step_idx > 0 and i < step.get('curr', 0):
                # Reversed arrows
                ax.annotate('', xy=(x1, y), xytext=(x2, y),
                           arrowprops=dict(arrowstyle='<-', color='red', lw=2))
            else:
                # Original arrows
                ax.annotate('', xy=(x2, y), xytext=(x1, y),
                           arrowprops=dict(arrowstyle='->', color='blue', lw=2))
        
        # Draw NULL
        ax.text((n_nodes) * 2, 0, 'NULL', ha='center', va='center', 
               fontsize=12, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow'))
        
        ax.set_xlim(-1, (n_nodes + 1) * 2)
        ax.set_ylim(-1.5, 1)
        ax.set_title(step['description'], fontsize=11, fontweight='bold')
        ax.axis('off')
    
    plt.suptitle('Reverse Linked List: Pointer Manipulation', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    filename = 'reverse_animation.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f" Reverse linked list visualization saved as {filename}")
    return filename

if __name__ == "__main__":
    create_reverse_visualization()
