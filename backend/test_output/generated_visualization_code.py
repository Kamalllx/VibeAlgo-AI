# Generated Visualization Code
# Created: 2025-05-31 22:41:25.998983


import matplotlib.pyplot as plt
import numpy as np
import os

print("🎨 Starting visualization generation...")

# Create complexity comparison
def create_complexity_chart():
    print("📊 Creating complexity comparison chart...")
    
    # Data
    n_values = np.logspace(1, 3, 50)  # 10 to 1000
    
    complexities = {
        'O(1)': np.ones_like(n_values),
        'O(log n)': np.log2(n_values),
        'O(n)': n_values,
        'O(n log n)': n_values * np.log2(n_values),
        'O(n²)': n_values ** 2 / 1000  # Scale down for visibility
    }
    
    print(f"📈 Plotting {len(complexities)} complexity curves...")
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Linear scale
    colors = ['green', 'blue', 'orange', 'red', 'purple']
    for i, (name, values) in enumerate(complexities.items()):
        if name != 'O(n²)':  # Skip quadratic for linear plot
            ax1.plot(n_values, values, label=name, color=colors[i], linewidth=2)
    
    ax1.set_xlabel('Input Size (n)')
    ax1.set_ylabel('Operations')
    ax1.set_title('Time Complexity Comparison (Linear Scale)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Log scale
    for i, (name, values) in enumerate(complexities.items()):
        ax2.loglog(n_values, values, label=name, color=colors[i], linewidth=2, marker='o')
    
    ax2.set_xlabel('Input Size (n)')
    ax2.set_ylabel('Operations (log scale)')
    ax2.set_title('Time Complexity Comparison (Log Scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    filename = 'complexity_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"💾 Saved plot: {filename}")
    
    plt.close()
    return filename

# Execute the function
if __name__ == "__main__":
    result_file = create_complexity_chart()
    print(f"✅ Visualization completed: {result_file}")
