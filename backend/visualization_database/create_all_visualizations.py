# backend/visualization_database/create_all_visualizations.py (IMPORT VERSION)
#!/usr/bin/env python3
"""
Generate ALL visualization files using modular approach
"""

from pathlib import Path

# Import all category modules
try:
    from .visualizations_searching import create_searching_visualizations
    from .visualizations_sorting import create_sorting_visualizations
    from .visualizations_graphs import create_graph_visualizations
    from .visualizations_dp import create_dp_visualizations
    from .visualizations_strings import create_string_visualizations
    from .visualizations_trees import create_tree_visualizations
    from .visualizations_linked_lists import create_linked_list_visualizations
    from .visualizations_stacks_queues import create_stack_queue_visualizations
    from .visualizations_math import create_math_visualizations
    from .visualizations_recursion import create_recursion_backtracking_visualizations
    from .visualizations_greedy import create_greedy_visualizations
    from .visualizations_bit import create_bit_manipulation_visualizations
    from .visualizations_sliding import create_sliding_window_visualizations
    from .visualizations_heap import create_heap_visualizations
    from .visualizations_hashing import create_hashing_visualizations
    from .visualizations_union_find import create_union_find_visualizations
except ImportError as e:
    print(f"Import error: {e}")
    print("Create the individual visualization files first")

def create_all_visualization_files():
    """Create all visualization files using modular approach"""
    
    base_dir = Path("visualizations")
    
    # Ensure all directories exist
    categories = [
        "searching", "sorting", "graphs", "dynamic_programming", "trees", 
        "strings", "linked_lists", "stacks_queues", "math_number_theory",
        "recursion_backtracking", "greedy", "bit_manipulation", "sliding_window",
        "heap", "hashing", "union_find"
    ]
    
    for category in categories:
        (base_dir / category).mkdir(parents=True, exist_ok=True)
    
    # Create all visualization files
    print(" Creating all visualization files...")
    
    try:
        create_searching_visualizations()
        create_sorting_visualizations()
        create_graph_visualizations()
        create_dp_visualizations()
        create_string_visualizations()
        create_tree_visualizations()
        create_linked_list_visualizations()
        create_stack_queue_visualizations()
        create_math_visualizations()
        create_recursion_backtracking_visualizations()
        create_greedy_visualizations()
        create_bit_manipulation_visualizations()
        create_sliding_window_visualizations()
        create_heap_visualizations()
        create_hashing_visualizations()
        create_union_find_visualizations()
        
        print(" All visualization files created successfully!")
        
    except Exception as e:
        print(f" Error creating visualization files: {e}")

if __name__ == "__main__":
    create_all_visualization_files()
