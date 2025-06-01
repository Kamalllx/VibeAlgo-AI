#!/usr/bin/env python3
"""
Euclidean Algorithm - Greatest Common Divisor
Shows step-by-step GCD calculation
"""

import matplotlib.pyplot as plt
import numpy as np

def create_euclidean_visualization():
    """Create Euclidean algorithm visualization"""
    
    # Calculate GCD of two numbers
    a, b = 48, 18
    original_a, original_b = a, b
    
    steps = []
    while b != 0:
        quotient = a // b
        remainder = a % b
        
        steps.append({
            'a': a,
            'b': b,
            'quotient': quotient,
            'remainder': remainder,
            'equation': f'{a} = {b} x {quotient} + {remainder}'
        })
        
        a, b = b, remainder
    
    gcd_result = a
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Step-by-step calculation
    ax1.set_title(f'Euclidean Algorithm: GCD({original_a}, {original_b})', fontsize=14, fontweight='bold')
    
    # Create table showing steps
    table_data = []
    for i, step in enumerate(steps):
        table_data.append([
            f"Step {i+1}",
            str(step['a']),
            str(step['b']),
            str(step['quotient']),
            str(step['remainder']),
            step['equation']
        ])
    
    # Add final step
    table_data.append([
        f"Step {len(steps)+1}",
        str(steps[-1]['remainder']),
        "0",
        "-",
        "0",
        f"GCD = {gcd_result}"
    ])
    
    ax1.axis('tight')
    ax1.axis('off')
    table = ax1.table(cellText=table_data,
                     colLabels=['Step', 'a', 'b', 'q', 'r', 'Equation: a = b × q + r'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color the header
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color the final result
    for i in range(len(table_data[0])):
        table[(len(table_data), i)].set_facecolor('#FFE082')
    
    # Visual representation
    ax2.set_title('Visual GCD Representation', fontsize=14, fontweight='bold')
    
    # Create rectangles showing the division process
    colors = plt.cm.Set3(np.linspace(0, 1, len(steps)))
    
    y_pos = 0
    for i, step in enumerate(steps):
        # Draw rectangle for 'a'
        rect_a = plt.Rectangle((0, y_pos), step['a']/10, 0.5, 
                              facecolor=colors[i], alpha=0.7, edgecolor='black')
        ax2.add_patch(rect_a)
        ax2.text(step['a']/20, y_pos + 0.25, f"a = {step['a']}", 
                ha='center', va='center', fontweight='bold')
        
        # Draw rectangle for 'b'
        rect_b = plt.Rectangle((0, y_pos - 0.7), step['b']/10, 0.5, 
                              facecolor=colors[i], alpha=0.5, edgecolor='black')
        ax2.add_patch(rect_b)
        ax2.text(step['b']/20, y_pos - 0.45, f"b = {step['b']}", 
                ha='center', va='center', fontweight='bold')
        
        # Show quotient and remainder
        ax2.text(max(step['a'], step['b'])/10 + 1, y_pos, 
                f"q = {step['quotient']}, r = {step['remainder']}", 
                ha='left', va='center', fontsize=10)
        
        y_pos -= 1.5
    
    # Highlight final GCD
    ax2.text(0.5, y_pos, f'GCD({original_a}, {original_b}) = {gcd_result}', 
            transform=ax2.transAxes, ha='center', va='center', 
            fontsize=16, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='yellow'))
    
    ax2.set_xlim(-1, max(original_a, original_b)/10 + 3)
    ax2.set_ylim(y_pos - 1, 1)
    ax2.axis('off')
    
    plt.tight_layout()
    
    filename = 'euclidean_gcd.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f" Euclidean algorithm visualization saved as {filename}")
    return filename

if __name__ == "__main__":
    create_euclidean_visualization()
