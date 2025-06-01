# backend/visualization_database/visualizations_bit.py
#!/usr/bin/env python3
"""
Bit Manipulation Algorithm Visualizations
All bit manipulation algorithm visualization implementations
"""

import os
from pathlib import Path

def create_bit_manipulation_visualizations():
    """Create all bit manipulation algorithm visualization files"""
    
    base_dir = Path("visualizations/bit_manipulation")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Count Set Bits (Brian Kernighan's Algorithm)
    count_set_bits_code = '''#!/usr/bin/env python3
"""
Count Set Bits - Brian Kernighan's Algorithm
Shows efficient bit counting with n & (n-1) technique
"""

import matplotlib.pyplot as plt
import numpy as np

def create_count_set_bits_visualization():
    """Create comprehensive count set bits visualization"""
    
    # Test numbers
    test_numbers = [12, 25, 31, 64, 127]
    
    def count_set_bits_naive(n):
        """Naive approach - check each bit"""
        count = 0
        steps = []
        original_n = n
        bit_pos = 0
        
        while n > 0:
            bit = n & 1
            if bit:
                count += 1
            
            steps.append({
                'n': n,
                'bit_pos': bit_pos,
                'current_bit': bit,
                'count': count,
                'binary': bin(original_n)[2:].zfill(8),
                'checking_bit': bit_pos
            })
            
            n >>= 1
            bit_pos += 1
        
        return count, steps
    
    def count_set_bits_kernighan(n):
        """Brian Kernighan's algorithm - n & (n-1)"""
        count = 0
        steps = []
        original_n = n
        
        while n > 0:
            steps.append({
                'n': n,
                'n_minus_1': n - 1,
                'n_and_n_minus_1': n & (n - 1),
                'count': count,
                'binary_n': bin(n)[2:].zfill(8),
                'binary_n_minus_1': bin(n - 1)[2:].zfill(8),
                'operation': f'{n} & {n-1} = {n & (n-1)}'
            })
            
            n = n & (n - 1)
            count += 1
        
        return count, steps
    
    # Create comprehensive visualization for one example
    num = 25  # Binary: 11001
    naive_count, naive_steps = count_set_bits_naive(num)
    kernighan_count, kernighan_steps = count_set_bits_kernighan(num)
    
    fig = plt.figure(figsize=(20, 16))
    
    # Binary representation
    ax1 = plt.subplot(4, 2, (1, 2))
    binary_rep = bin(num)[2:].zfill(8)
    
    # Draw binary representation
    for i, bit in enumerate(binary_rep):
        color = 'red' if bit == '1' else 'lightgray'
        rect = plt.Rectangle((i, 0), 0.8, 0.8, facecolor=color, edgecolor='black', linewidth=2)
        ax1.add_patch(rect)
        ax1.text(i + 0.4, 0.4, bit, ha='center', va='center', fontsize=14, fontweight='bold')
        ax1.text(i + 0.4, -0.3, str(7-i), ha='center', va='center', fontsize=10)
    
    ax1.set_xlim(-0.1, 8.1)
    ax1.set_ylim(-0.5, 1)
    ax1.set_title(f'Binary Representation of {num}: {binary_rep}', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Naive approach visualization
    ax2 = plt.subplot(4, 2, 3)
    ax2.set_title('Naive Approach: Check Each Bit', fontweight='bold')
    
    # Show steps in table format
    naive_data = []
    for step in naive_steps[:8]:  # Show first 8 steps
        naive_data.append([
            str(step['bit_pos']),
            step['binary'][7-step['bit_pos']],
            str(step['current_bit']),
            str(step['count'])
        ])
    
    ax2.axis('tight')
    ax2.axis('off')
    naive_table = ax2.table(cellText=naive_data,
                           colLabels=['Bit Position', 'Bit Value', 'Is Set?', 'Count'],
                           cellLoc='center', loc='center')
    naive_table.auto_set_font_size(False)
    naive_table.set_fontsize(10)
    naive_table.scale(1, 1.5)
    
    # Color header
    for i in range(4):
        naive_table[(0, i)].set_facecolor('#2196F3')
        naive_table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Kernighan's approach visualization
    ax3 = plt.subplot(4, 2, 4)
    ax3.set_title("Brian Kernighan's Algorithm: n & (n-1)", fontweight='bold')
    
    kernighan_data = []
    for step in kernighan_steps:
        kernighan_data.append([
            step['binary_n'],
            step['binary_n_minus_1'],
            bin(step['n_and_n_minus_1'])[2:].zfill(8),
            str(step['count'])
        ])
    
    ax3.axis('tight')
    ax3.axis('off')
    kernighan_table = ax3.table(cellText=kernighan_data,
                               colLabels=['n', 'n-1', 'n & (n-1)', 'Iterations'],
                               cellLoc='center', loc='center')
    kernighan_table.auto_set_font_size(False)
    kernighan_table.set_fontsize(10)
    kernighan_table.scale(1, 1.5)
    
    # Color header
    for i in range(4):
        kernighan_table[(0, i)].set_facecolor('#4CAF50')
        kernighan_table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Step-by-step bit manipulation for Kernighan's algorithm
    ax4 = plt.subplot(4, 2, (5, 6))
    
    step_num = 0
    current_n = num
    y_pos = len(kernighan_steps)
    
    # Show each step of n & (n-1)
    for step in kernighan_steps:
        # Draw binary representation for each step
        y = y_pos - step_num
        
        # n
        for i, bit in enumerate(step['binary_n']):
            color = 'red' if bit == '1' else 'lightgray'
            rect = plt.Rectangle((i, y), 0.8, 0.3, facecolor=color, edgecolor='black')
            ax4.add_patch(rect)
            ax4.text(i + 0.4, y + 0.15, bit, ha='center', va='center', fontsize=10, fontweight='bold')
        
        ax4.text(-1, y + 0.15, f'n = {step["n"]}:', ha='right', va='center', fontweight='bold')
        
        # n-1
        y -= 0.4
        for i, bit in enumerate(step['binary_n_minus_1']):
            color = 'orange' if bit == '1' else 'lightgray'
            rect = plt.Rectangle((i, y), 0.8, 0.3, facecolor=color, edgecolor='black')
            ax4.add_patch(rect)
            ax4.text(i + 0.4, y + 0.15, bit, ha='center', va='center', fontsize=10, fontweight='bold')
        
        ax4.text(-1, y + 0.15, f'n-1 = {step["n_minus_1"]}:', ha='right', va='center', fontweight='bold')
        
        # Result
        y -= 0.4
        result_binary = bin(step['n_and_n_minus_1'])[2:].zfill(8)
        for i, bit in enumerate(result_binary):
            color = 'green' if bit == '1' else 'lightgray'
            rect = plt.Rectangle((i, y), 0.8, 0.3, facecolor=color, edgecolor='black')
            ax4.add_patch(rect)
            ax4.text(i + 0.4, y + 0.15, bit, ha='center', va='center', fontsize=10, fontweight='bold')
        
        ax4.text(-1, y + 0.15, f'n & (n-1) = {step["n_and_n_minus_1"]}:', ha='right', va='center', fontweight='bold')
        
        # AND operation lines
        for i in range(8):
            ax4.plot([i + 0.4, i + 0.4], [y + 0.3, y + 1.1], 'k--', alpha=0.3)
        
        step_num += 1
        y_pos -= 1.5
    
    ax4.set_xlim(-3, 8.5)
    ax4.set_ylim(y_pos - 0.5, len(kernighan_steps) + 0.5)
    ax4.set_title('Step-by-Step Bit Manipulation', fontweight='bold')
    ax4.axis('off')
    
    # Comparison of algorithms
    ax5 = plt.subplot(4, 2, 7)
    
    # Test on multiple numbers
    test_results = []
    for test_num in test_numbers:
        naive_c, naive_s = count_set_bits_naive(test_num)
        kernighan_c, kernighan_s = count_set_bits_kernighan(test_num)
        test_results.append({
            'number': test_num,
            'binary': bin(test_num)[2:],
            'set_bits': naive_c,
            'naive_ops': len(naive_s),
            'kernighan_ops': len(kernighan_s)
        })
    
    # Plot comparison
    numbers = [r['number'] for r in test_results]
    naive_ops = [r['naive_ops'] for r in test_results]
    kernighan_ops = [r['kernighan_ops'] for r in test_results]
    
    x = np.arange(len(numbers))
    width = 0.35
    
    bars1 = ax5.bar(x - width/2, naive_ops, width, label='Naive Approach', color='lightblue')
    bars2 = ax5.bar(x + width/2, kernighan_ops, width, label="Kernighan's Algorithm", color='lightgreen')
    
    ax5.set_xlabel('Test Numbers')
    ax5.set_ylabel('Number of Operations')
    ax5.set_title('Algorithm Comparison: Operations Count')
    ax5.set_xticks(x)
    ax5.set_xticklabels([f'{n}\\n({bin(n)[2:]})' for n in numbers])
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   str(int(height)), ha='center', va='bottom', fontweight='bold')
    
    # Algorithm summary
    ax6 = plt.subplot(4, 2, 8)
    ax6.axis('off')
    
    summary_text = f"""
BIT MANIPULATION: COUNT SET BITS

NAIVE APPROACH:
• Check each bit position (LSB to MSB)
• Time Complexity: O(log n)
• Space Complexity: O(1)
• Operations for {num}: {len(naive_steps)}

BRIAN KERNIGHAN'S ALGORITHM:
• Use n & (n-1) to clear rightmost set bit
• Time Complexity: O(number of set bits)
• Space Complexity: O(1)
• Operations for {num}: {len(kernighan_steps)}

KEY INSIGHT:
n & (n-1) always clears the rightmost set bit
Example: 12 & 11 = 1100 & 1011 = 1000

OPTIMIZATION:
Kernighan's algorithm is faster when the number
of set bits is small compared to total bits.

For {num} (binary: {bin(num)[2:]}):
• Set bits: {naive_count}
• Naive operations: {len(naive_steps)}
• Kernighan operations: {len(kernighan_steps)}
• Efficiency gain: {((len(naive_steps) - len(kernighan_steps)) / len(naive_steps) * 100):.1f}%
"""
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightcyan'))
    
    plt.tight_layout()
    
    filename = 'count_set_bits.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f" Count set bits visualization saved as {filename}")
    return filename

if __name__ == "__main__":
    create_count_set_bits_visualization()
'''
    
    with open(base_dir / "count_set_bits.py", "w") as f:
        f.write(count_set_bits_code)
    
    print(" Created bit manipulation visualization files")
