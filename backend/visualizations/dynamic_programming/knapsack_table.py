#!/usr/bin/env python3
"""
0/1 Knapsack Problem - DP Table Visualization
Shows weight vs value optimization
"""

import matplotlib.pyplot as plt
import numpy as np

def create_knapsack_visualization():
    """Create 0/1 Knapsack visualization"""
    
    # Sample items: (weight, value)
    items = [(2, 3), (3, 4), (4, 5), (5, 6)]
    weights = [item[0] for item in items]
    values = [item[1] for item in items]
    capacity = 8
    n = len(items)
    
    # DP table
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
    
    # Fill DP table
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(
                    dp[i-1][w],  # Don't take item
                    dp[i-1][w - weights[i-1]] + values[i-1]  # Take item
                )
            else:
                dp[i][w] = dp[i-1][w]
    
    # Create visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Items visualization
    ax1.bar(range(len(items)), values, color='lightblue', alpha=0.7, label='Value')
    ax1_twin = ax1.twinx()
    ax1_twin.bar([x + 0.3 for x in range(len(items))], weights, 
                color='orange', alpha=0.7, width=0.3, label='Weight')
    
    ax1.set_xlabel('Item Index')
    ax1.set_ylabel('Value', color='blue')
    ax1_twin.set_ylabel('Weight', color='orange')
    ax1.set_title('Items: Weight vs Value')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    
    # DP table heatmap
    im = ax2.imshow(dp, cmap='YlOrRd', aspect='auto')
    ax2.set_xlabel('Capacity')
    ax2.set_ylabel('Items (0 to n)')
    ax2.set_title(f'DP Table: Max Value for Each (Items, Capacity)')
    
    # Add text annotations
    for i in range(n + 1):
        for j in range(capacity + 1):
            ax2.text(j, i, str(dp[i][j]), ha='center', va='center', 
                    color='white' if dp[i][j] > 5 else 'black', fontweight='bold')
    
    # Colorbar
    plt.colorbar(im, ax=ax2)
    
    # Solution tracing
    ax3.set_title('Optimal Solution Trace')
    
    # Trace back solution
    w = capacity
    selected_items = []
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            selected_items.append(i-1)
            w -= weights[i-1]
    
    selected_items.reverse()
    
    # Show selected items
    colors = ['red' if i in selected_items else 'lightgray' for i in range(len(items))]
    bars = ax3.bar(range(len(items)), values, color=colors, alpha=0.7, edgecolor='black')
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        weight = weights[i]
        status = "SELECTED" if i in selected_items else "NOT SELECTED"
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'W:{weight}\nV:{height}\n{status}', 
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    total_weight = sum(weights[i] for i in selected_items)
    total_value = sum(values[i] for i in selected_items)
    
    ax3.text(0.5, 0.95, f'Total Weight: {total_weight}/{capacity}\nTotal Value: {total_value}', 
            transform=ax3.transAxes, ha='center', va='top', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow'))
    
    ax3.set_xlabel('Item Index')
    ax3.set_ylabel('Value')
    
    plt.tight_layout()
    
    filename = 'knapsack_table.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f" Knapsack visualization saved as {filename}")
    return filename

if __name__ == "__main__":
    create_knapsack_visualization()
