#!/usr/bin/env python3
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
