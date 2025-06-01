# backend/visualization_database/visualizations_recursion.py
#!/usr/bin/env python3
"""
Recursion and Backtracking Algorithm Visualizations
All recursion and backtracking algorithm visualization implementations
"""

import os
from pathlib import Path

def create_recursion_backtracking_visualizations():
    """Create all recursion and backtracking algorithm visualization files"""
    
    base_dir = Path("visualizations/recursion_backtracking")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # N-Queens Problem
    n_queens_code = '''#!/usr/bin/env python3
"""
N-Queens Problem - Backtracking Visualization
Shows queen placement and backtracking
"""

import matplotlib.pyplot as plt
import numpy as np

def create_n_queens_visualization():
    """Create N-Queens backtracking visualization"""
    
    n = 4  # 4x4 chessboard
    board = [[0 for _ in range(n)] for _ in range(n)]
    solutions = []
    steps = []
    
    def is_safe(board, row, col):
        # Check column
        for i in range(row):
            if board[i][col] == 1:
                return False
        
        # Check upper diagonal
        i, j = row - 1, col - 1
        while i >= 0 and j >= 0:
            if board[i][j] == 1:
                return False
            i, j = i - 1, j - 1
        
        # Check upper anti-diagonal
        i, j = row - 1, col + 1
        while i >= 0 and j < n:
            if board[i][j] == 1:
                return False
            i, j = i - 1, j + 1
        
        return True
    
    def solve_n_queens(board, row):
        if row >= n:
            # Found a solution
            solution = [row[:] for row in board]
            solutions.append(solution)
            steps.append({
                'board': solution,
                'row': row,
                'action': 'Solution found!'
            })
            return True
        
        for col in range(n):
            if is_safe(board, row, col):
                board[row][col] = 1
                steps.append({
                    'board': [row[:] for row in board],
                    'row': row,
                    'col': col,
                    'action': f'Place queen at ({row}, {col})'
                })
                
                if solve_n_queens(board, row + 1):
                    return True
                
                # Backtrack
                board[row][col] = 0
                steps.append({
                    'board': [row[:] for row in board],
                    'row': row,
                    'col': col,
                    'action': f'Backtrack: Remove queen from ({row}, {col})'
                })
        
        return False
    
    solve_n_queens(board, 0)
    
    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    axes = axes.flatten()
    
    for step_idx, step in enumerate(steps[:8]):
        ax = axes[step_idx]
        
        # Create chessboard
        board_visual = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if (i + j) % 2 == 0:
                    board_visual[i][j] = 0.8  # Light squares
                else:
                    board_visual[i][j] = 0.4  # Dark squares
        
        ax.imshow(board_visual, cmap='gray', alpha=0.7)
        
        # Place queens
        for i in range(n):
            for j in range(n):
                if step['board'][i][j] == 1:
                    ax.text(j, i, 'Q', ha='center', va='center', 
                           fontsize=20, color='red', fontweight='bold')
        
        # Highlight current position if applicable
        if 'col' in step:
            row, col = step['row'], step['col']
            if 'Remove' not in step['action']:
                circle = plt.Circle((col, row), 0.3, fill=False, 
                                  edgecolor='blue', linewidth=3)
                ax.add_patch(circle)
        
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(-0.5, n - 0.5)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_title(f'Step {step_idx + 1}: {step["action"]}', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{n}-Queens Problem: Backtracking Solution', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    filename = 'n_queens.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"N-Queens visualization saved as {filename}")
    print(f" Found {len(solutions)} solution(s)")
    return filename

if __name__ == "__main__":
    create_n_queens_visualization()
'''
    
    with open(base_dir / "n_queens.py", "w") as f:
        f.write(n_queens_code)
    
    print("Created recursion/backtracking visualization files")
