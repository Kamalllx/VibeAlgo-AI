# backend/visualization_database/visualizations_linked_lists.py
#!/usr/bin/env python3
"""
Linked List Algorithm Visualizations
All linked list algorithm visualization implementations
"""

import os
from pathlib import Path

def create_linked_list_visualizations():
    """Create all linked list algorithm visualization files"""
    
    base_dir = Path("visualizations/linked_lists")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Reverse Linked List
    reverse_code = '''#!/usr/bin/env python3
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
'''
    
    with open(base_dir / "reverse_animation.py", "w") as f:
        f.write(reverse_code)
    
    # Cycle Detection (Floyd's Algorithm)
    cycle_detection_code = '''#!/usr/bin/env python3
"""
Cycle Detection - Floyd's Tortoise and Hare Algorithm
Shows slow and fast pointer technique
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def create_cycle_detection_visualization():
    """Create cycle detection visualization"""
    
    # Linked list with cycle: 1 -> 2 -> 3 -> 4 -> 5 -> 3 (cycle back to 3)
    nodes = [1, 2, 3, 4, 5]
    cycle_start = 2  # Index of node 3
    
    # Floyd's algorithm simulation
    steps = []
    slow = 0
    fast = 0
    
    # Phase 1: Detect cycle
    for step in range(8):
        steps.append({
            'slow': slow,
            'fast': fast,
            'phase': 1,
            'description': f'Step {step + 1}: slow at {nodes[slow]}, fast at {nodes[fast]}'
        })
        
        # Move slow pointer one step
        slow = (slow + 1) % len(nodes)
        if slow == cycle_start and slow > cycle_start:
            slow = cycle_start
        
        # Move fast pointer two steps
        for _ in range(2):
            fast = (fast + 1) % len(nodes)
            if fast >= cycle_start and fast != cycle_start:
                if fast == len(nodes):
                    fast = cycle_start
        
        # Check if they meet
        if slow == fast:
            steps.append({
                'slow': slow,
                'fast': fast,
                'phase': 1,
                'description': f'CYCLE DETECTED! Pointers meet at {nodes[slow]}'
            })
            break
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for step_idx, step in enumerate(steps[:6]):
        ax = axes[step_idx]
        
        # Node positions in circular layout for better cycle visualization
        angles = np.linspace(0, 2*np.pi, len(nodes), endpoint=False)
        radius = 2
        positions = [(radius * np.cos(angle), radius * np.sin(angle)) for angle in angles]
        
        # Draw nodes
        for i, value in enumerate(nodes):
            x, y = positions[i]
            
            # Color based on pointer positions
            color = 'lightgray'
            if i == step['slow'] and i == step['fast']:
                color = 'red'  # Both pointers
            elif i == step['slow']:
                color = 'blue'  # Slow pointer
            elif i == step['fast']:
                color = 'green'  # Fast pointer
            
            circle = plt.Circle((x, y), 0.3, color=color, ec='black', linewidth=2)
            ax.add_patch(circle)
            ax.text(x, y, str(value), ha='center', va='center', 
                   fontsize=12, fontweight='bold', color='white')
        
        # Draw connections
        for i in range(len(nodes)):
            x1, y1 = positions[i]
            if i == len(nodes) - 1:
                # Last node connects back to cycle start
                x2, y2 = positions[cycle_start]
            else:
                x2, y2 = positions[i + 1]
            
            # Calculate arrow position
            dx, dy = x2 - x1, y2 - y1
            length = np.sqrt(dx**2 + dy**2)
            dx_norm, dy_norm = dx/length, dy/length
            
            start_x = x1 + 0.3 * dx_norm
            start_y = y1 + 0.3 * dy_norm
            end_x = x2 - 0.3 * dx_norm
            end_y = y2 - 0.3 * dy_norm
            
            ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                       arrowprops=dict(arrowstyle='->', color='black', lw=2))
        
        # Add pointer labels
        slow_pos = positions[step['slow']]
        fast_pos = positions[step['fast']]
        
        ax.text(slow_pos[0], slow_pos[1] - 0.6, 'SLOW', ha='center', va='center', 
               fontsize=10, fontweight='bold', color='blue')
        
        if step['slow'] != step['fast']:
            ax.text(fast_pos[0], fast_pos[1] - 0.6, 'FAST', ha='center', va='center', 
                   fontsize=10, fontweight='bold', color='green')
        else:
            ax.text(slow_pos[0], slow_pos[1] - 0.6, 'MEET', ha='center', va='center', 
                   fontsize=10, fontweight='bold', color='red')
        
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.set_title(step['description'], fontsize=10, fontweight='bold')
        ax.axis('off')
    
    plt.suptitle("Floyd's Cycle Detection: Tortoise and Hare Algorithm", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    filename = 'cycle_detection_animation.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f" Cycle detection visualization saved as {filename}")
    return filename

if __name__ == "__main__":
    create_cycle_detection_visualization()
'''
    
    with open(base_dir / "cycle_detection_animation.py", "w") as f:
        f.write(cycle_detection_code)
    
    print("Created linked list visualization files")
