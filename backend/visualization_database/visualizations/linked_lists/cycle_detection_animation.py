#!/usr/bin/env python3
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
