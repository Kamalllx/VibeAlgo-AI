# backend/visualizations/linked_lists/merge_sorted_lists.py
#!/usr/bin/env python3
"""
Merge Two Sorted Linked Lists - Step-by-Step Animation
Shows merging process with pointer manipulation
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def create_merge_sorted_lists_visualization():
    """Create merge two sorted linked lists visualization"""
    
    # Sample sorted linked lists
    list1 = [1, 2, 4]
    list2 = [1, 3, 4]
    
    # Merge algorithm simulation
    def merge_lists_steps(l1, l2):
        steps = []
        i = j = 0
        result = []
        
        while i < len(l1) and j < len(l2):
            if l1[i] <= l2[j]:
                result.append(l1[i])
                steps.append({
                    'l1': l1, 'l2': l2, 'result': result.copy(),
                    'i': i, 'j': j, 'chosen': l1[i], 'from_list': 1
                })
                i += 1
            else:
                result.append(l2[j])
                steps.append({
                    'l1': l1, 'l2': l2, 'result': result.copy(),
                    'i': i, 'j': j, 'chosen': l2[j], 'from_list': 2
                })
                j += 1
        
        # Add remaining elements
        while i < len(l1):
            result.append(l1[i])
            steps.append({
                'l1': l1, 'l2': l2, 'result': result.copy(),
                'i': i, 'j': j, 'chosen': l1[i], 'from_list': 1
            })
            i += 1
            
        while j < len(l2):
            result.append(l2[j])
            steps.append({
                'l1': l1, 'l2': l2, 'result': result.copy(),
                'i': i, 'j': j, 'chosen': l2[j], 'from_list': 2
            })
            j += 1
            
        return steps
    
    steps = merge_lists_steps(list1, list2)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for step_idx, step in enumerate(steps[:6]):
        ax = axes[step_idx]
        
        # Draw List 1
        for i, val in enumerate(step['l1']):
            color = 'orange' if i == step.get('i', 0) and step['from_list'] == 1 else 'lightblue'
            rect = patches.Rectangle((i * 0.8, 2), 0.6, 0.6, 
                                   facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(i * 0.8 + 0.3, 2.3, str(val), ha='center', va='center', 
                   fontsize=12, fontweight='bold')
        
        # Draw List 2
        for i, val in enumerate(step['l2']):
            color = 'orange' if i == step.get('j', 0) and step['from_list'] == 2 else 'lightgreen'
            rect = patches.Rectangle((i * 0.8, 1), 0.6, 0.6, 
                                   facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(i * 0.8 + 0.3, 1.3, str(val), ha='center', va='center', 
                   fontsize=12, fontweight='bold')
        
        # Draw Result
        for i, val in enumerate(step['result']):
            rect = patches.Rectangle((i * 0.8, 0), 0.6, 0.6, 
                                   facecolor='yellow', edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(i * 0.8 + 0.3, 0.3, str(val), ha='center', va='center', 
                   fontsize=12, fontweight='bold')
        
        # Labels
        ax.text(-0.5, 2.3, 'List1:', ha='right', va='center', fontweight='bold')
        ax.text(-0.5, 1.3, 'List2:', ha='right', va='center', fontweight='bold')
        ax.text(-0.5, 0.3, 'Result:', ha='right', va='center', fontweight='bold')
        
        ax.set_xlim(-1, max(len(list1), len(list2)) * 0.8)
        ax.set_ylim(-0.5, 3)
        ax.set_title(f'Step {step_idx + 1}: Choose {step["chosen"]} from List{step["from_list"]}', 
                    fontweight='bold')
        ax.axis('off')
    
    plt.suptitle('Merge Two Sorted Linked Lists', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    filename = 'merge_sorted_lists.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"SUCCESS: Merge sorted lists visualization saved as {filename}")
    return filename

if __name__ == "__main__":
    create_merge_sorted_lists_visualization()
