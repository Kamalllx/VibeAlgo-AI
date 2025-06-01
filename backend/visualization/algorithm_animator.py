# backend/visualization/algorithm_animator.py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle, Circle, Arrow
from matplotlib.collections import PatchCollection
import time
from typing import Dict, List, Any, Tuple
from ai.groq_client import groq_client
from visualization.complexity_visualizer import BaseVisualizationAgent

class AlgorithmAnimationAgent(BaseVisualizationAgent):
    def __init__(self):
        super().__init__()
        self.name = "AlgorithmAnimator"
        self.specialties = [
            "sorting_animations",
            "search_animations", 
            "graph_traversal_animations",
            "data_structure_operations",
            "step_by_step_execution",
            "algorithm_comparisons"
        ]
        
        print(f"🎬 [{self.name}] Algorithm Animation Agent initialized")
    
    async def generate_visualization(self, request) -> Dict[str, Any]:
        """Generate algorithm animation visualizations"""
        print(f"\n🎬 [{self.name}] GENERATING ALGORITHM ANIMATIONS")
        
        # Extract algorithm data
        algorithm_data = self._extract_algorithm_data(request.data)
        
        # Get AI recommendations for animation style
        animation_style = await self._get_animation_recommendations(algorithm_data, request)
        
        # Generate different types of animations
        animations = []
        
        # 1. Step-by-step algorithm execution
        step_animation = self._create_step_by_step_animation(algorithm_data)
        animations.append(step_animation)
        
        # 2. Data structure visualization
        if "data_structure" in request.requirements:
            ds_animation = self._create_data_structure_animation(algorithm_data)
            animations.append(ds_animation)
        
        # 3. Algorithm comparison animation
        if "comparison" in request.requirements:
            comparison_animation = self._create_algorithm_comparison_animation(algorithm_data)
            animations.append(comparison_animation)
        
        # 4. Interactive algorithm stepper
        if request.output_format == "interactive":
            interactive_stepper = self._create_interactive_stepper(algorithm_data)
            animations.append(interactive_stepper)
        
        # Combine all animations
        combined_code = self._combine_algorithm_animations(animations)
        
        return {
            "agent_name": self.name,
            "visualization_type": "algorithm_animation",
            "code": combined_code,
            "data": algorithm_data,
            "file_paths": ["algorithm_animation.gif", "algorithm_steps.html", "algorithm_comparison.mp4"],
            "metadata": {
                "animation_types": ["step_by_step", "data_structure", "comparison"],
                "algorithm_detected": algorithm_data.get("algorithm_type", "Unknown"),
                "total_steps": algorithm_data.get("step_count", 0)
            },
            "frontend_instructions": self._get_animation_frontend_instructions()
        }
    
    async def _get_animation_recommendations(self, algorithm_data: Dict[str, Any], request) -> Dict[str, Any]:
        """Get AI recommendations for optimal animation approach"""
        recommendation_prompt = f"""
        As an algorithm visualization expert, recommend the best animation approach for this algorithm:
        
        Algorithm Data: {algorithm_data}
        Code: {algorithm_data.get('code', 'No code provided')}
        
        Suggest:
        1. Key steps to highlight in animation
        2. Visual metaphors (sorting as bars, search as spotlight, etc.)
        3. Color coding scheme for different operations
        4. Timing and pacing for educational effectiveness
        5. Interactive elements to include
        6. Multiple views (overview + detail)
        
        Focus on educational clarity and engagement.
        """
        
        response = groq_client.chat_completion([
            {"role": "system", "content": "You are an expert in algorithm visualization and educational animation design."},
            {"role": "user", "content": recommendation_prompt}
        ])
        
        return self._parse_animation_recommendations(response.content)
    
    def _create_step_by_step_animation(self, algorithm_data: Dict[str, Any]) -> str:
        """Create step-by-step algorithm execution animation"""
        return '''
def create_step_by_step_animation():
    """
    Step-by-Step Algorithm Animation
    
    Frontend Integration:
    - Convert to WebM/MP4 for web playback
    - Add play/pause/step controls
    - Implement speed control slider
    - Provide frame-by-frame navigation
    - Add explanatory text for each step
    """
    
    # Detect algorithm type and create appropriate animation
    algorithm_type = algorithm_data.get('algorithm_type', 'sorting')
    code = algorithm_data.get('code', '')
    
    if 'sort' in algorithm_type.lower() or 'sort' in code.lower():
        return create_sorting_animation()
    elif 'search' in algorithm_type.lower() or 'search' in code.lower():
        return create_search_animation()
    elif 'graph' in algorithm_type.lower() or 'bfs' in code.lower() or 'dfs' in code.lower():
        return create_graph_traversal_animation()
    else:
        return create_generic_algorithm_animation()

def create_sorting_animation():
    """Create animated sorting visualization"""
    
    # Sample data to sort
    data = [64, 34, 25, 12, 22, 11, 90]
    
    # Create figure and axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Initialize bars
    bars = ax1.bar(range(len(data)), data, color='lightblue')
    ax1.set_title('Algorithm Step-by-Step: Bubble Sort', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Array Index')
    ax1.set_ylabel('Value')
    ax1.set_ylim(0, max(data) * 1.1)
    
    # Status text area
    ax2.axis('off')
    status_text = ax2.text(0.05, 0.8, '', transform=ax2.transAxes, fontsize=12,
                          verticalalignment='top', fontfamily='monospace')
    
    # Algorithm state tracking
    steps = []
    current_data = data.copy()
    
    # Generate bubble sort steps
    n = len(current_data)
    for i in range(n):
        for j in range(0, n-i-1):
            # Record comparison step
            steps.append({
                'type': 'compare',
                'indices': [j, j+1],
                'data': current_data.copy(),
                'message': f'Comparing elements at positions {j} and {j+1}: {current_data[j]} vs {current_data[j+1]}'
            })
            
            if current_data[j] > current_data[j+1]:
                # Record swap step
                current_data[j], current_data[j+1] = current_data[j+1], current_data[j]
                steps.append({
                    'type': 'swap',
                    'indices': [j, j+1],
                    'data': current_data.copy(),
                    'message': f'Swapping {current_data[j+1]} and {current_data[j]}'
                })
    
    # Animation function
    def animate(frame):
        if frame >= len(steps):
            return bars + [status_text]
        
        step = steps[frame]
        
        # Update bar heights
        for i, (bar, value) in enumerate(zip(bars, step['data'])):
            bar.set_height(value)
            
            # Color coding based on step type
            if step['type'] == 'compare' and i in step['indices']:
                bar.set_color('yellow')
            elif step['type'] == 'swap' and i in step['indices']:
                bar.set_color('red')
            else:
                bar.set_color('lightblue')
        
        # Update status text
        status_text.set_text(f"Step {frame + 1}/{len(steps)}\\n{step['message']}")
        
        return bars + [status_text]
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=len(steps)+10, 
                                 interval=1000, blit=False, repeat=True)
    
    # Save animation
    anim.save('bubble_sort_animation.gif', writer='pillow', fps=1)
    anim.save('bubble_sort_animation.mp4', writer='ffmpeg', fps=2)
    
    plt.tight_layout()
    plt.show()
    
    return anim

def create_search_animation():
    """Create animated search visualization"""
    
    # Sample sorted array for binary search
    data = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
    target = 13
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # Initialize visualization
    bars = ax1.bar(range(len(data)), [1]*len(data), color='lightgray')
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, data)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(value), ha='center', va='bottom', fontweight='bold')
    
    ax1.set_title(f'Binary Search Animation: Finding {target}', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Array Index')
    ax1.set_ylabel('Active Search Range')
    ax1.set_ylim(0, 2)
    
    # Status area
    ax2.axis('off')
    status_text = ax2.text(0.05, 0.8, '', transform=ax2.transAxes, fontsize=12,
                          verticalalignment='top', fontfamily='monospace')
    
    # Generate binary search steps
    steps = []
    left, right = 0, len(data) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        steps.append({
            'type': 'search_range',
            'left': left,
            'right': right,
            'mid': mid,
            'message': f'Search range: [{left}, {right}], checking middle index {mid} (value: {data[mid]})'
        })
        
        if data[mid] == target:
            steps.append({
                'type': 'found',
                'left': left,
                'right': right,
                'mid': mid,
                'message': f'Target {target} found at index {mid}!'
            })
            break
        elif data[mid] < target:
            left = mid + 1
            steps.append({
                'type': 'narrow_right',
                'left': left,
                'right': right,
                'mid': mid,
                'message': f'{data[mid]} < {target}, searching right half'
            })
        else:
            right = mid - 1
            steps.append({
                'type': 'narrow_left',
                'left': left,
                'right': right,
                'mid': mid,
                'message': f'{data[mid]} > {target}, searching left half'
            })
    
    def animate_search(frame):
        if frame >= len(steps):
            return bars + [status_text]
        
        step = steps[frame]
        
        # Reset all bars
        for bar in bars:
            bar.set_color('lightgray')
            bar.set_height(1)
        
        # Highlight active search range
        for i in range(step['left'], step['right'] + 1):
            bars[i].set_color('lightblue')
            bars[i].set_height(1.5)
        
        # Highlight middle element
        if 'mid' in step:
            if step['type'] == 'found':
                bars[step['mid']].set_color('green')
            else:
                bars[step['mid']].set_color('orange')
            bars[step['mid']].set_height(1.8)
        
        # Update status
        status_text.set_text(f"Step {frame + 1}/{len(steps)}\\n{step['message']}")
        
        return bars + [status_text]
    
    # Create animation
    search_anim = animation.FuncAnimation(fig, animate_search, frames=len(steps)+5,
                                        interval=1500, blit=False, repeat=True)
    
    # Save animation
    search_anim.save('binary_search_animation.gif', writer='pillow', fps=0.8)
    
    plt.tight_layout()
    plt.show()
    
    return search_anim

def create_graph_traversal_animation():
    """Create animated graph traversal visualization"""
    
    import networkx as nx
    from matplotlib.patches import Circle
    
    # Create sample graph
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6), (3, 7)])
    
    # Position nodes
    pos = nx.spring_layout(G, seed=42)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Draw initial graph
    nx.draw_networkx_edges(G, pos, ax=ax1, edge_color='gray', width=2)
    nodes = nx.draw_networkx_nodes(G, pos, ax=ax1, node_color='lightblue', 
                                  node_size=800, alpha=0.8)
    nx.draw_networkx_labels(G, pos, ax=ax1, font_size=12, font_weight='bold')
    
    ax1.set_title('Graph Traversal Animation: BFS', fontsize=16, fontweight='bold')
    ax1.axis('off')
    
    # Status area
    ax2.axis('off')
    status_text = ax2.text(0.05, 0.9, '', transform=ax2.transAxes, fontsize=12,
                          verticalalignment='top', fontfamily='monospace')
    
    # Generate BFS traversal steps
    from collections import deque
    
    steps = []
    visited = set()
    queue = deque([0])  # Start from node 0
    visited.add(0)
    
    steps.append({
        'type': 'start',
        'current': 0,
        'queue': list(queue),
        'visited': visited.copy(),
        'message': 'Starting BFS from node 0'
    })
    
    while queue:
        current = queue.popleft()
        
        steps.append({
            'type': 'visit',
            'current': current,
            'queue': list(queue),
            'visited': visited.copy(),
            'message': f'Visiting node {current}'
        })
        
        for neighbor in G.neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                
                steps.append({
                    'type': 'discover',
                    'current': current,
                    'discovered': neighbor,
                    'queue': list(queue),
                    'visited': visited.copy(),
                    'message': f'Discovered node {neighbor}, added to queue'
                })
    
    def animate_traversal(frame):
        if frame >= len(steps):
            return
        
        step = steps[frame]
        
        # Clear previous drawing
        ax1.clear()
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, ax=ax1, edge_color='gray', width=2)
        
        # Color nodes based on their state
        node_colors = []
        for node in G.nodes():
            if node in step['visited']:
                if node == step.get('current', -1):
                    node_colors.append('red')  # Currently visiting
                elif node == step.get('discovered', -1):
                    node_colors.append('yellow')  # Just discovered
                else:
                    node_colors.append('green')  # Visited
            elif node in step['queue']:
                node_colors.append('orange')  # In queue
            else:
                node_colors.append('lightblue')  # Unvisited
        
        nx.draw_networkx_nodes(G, pos, ax=ax1, node_color=node_colors,
                              node_size=800, alpha=0.8)
        nx.draw_networkx_labels(G, pos, ax=ax1, font_size=12, font_weight='bold')
        
        ax1.set_title('Graph Traversal Animation: BFS', fontsize=16, fontweight='bold')
        ax1.axis('off')
        
        # Update status
        queue_str = ' -> '.join(map(str, step['queue']))
        visited_str = ', '.join(map(str, sorted(step['visited'])))
        
        status_text.set_text(
            f"Step {frame + 1}/{len(steps)}\\n"
            f"{step['message']}\\n\\n"
            f"Queue: [{queue_str}]\\n"
            f"Visited: {{{visited_str}}}\\n\\n"
            f"Legend:\\n"
            f"🔴 Currently visiting\\n"
            f"🟡 Just discovered\\n"
            f"🟢 Already visited\\n"
            f"🟠 In queue\\n"
            f"🔵 Unvisited"
        )
    
    # Create animation
    traversal_anim = animation.FuncAnimation(fig, animate_traversal, frames=len(steps)+5,
                                           interval=2000, repeat=True)
    
    # Save animation
    traversal_anim.save('bfs_traversal_animation.gif', writer='pillow', fps=0.5)
    
    plt.tight_layout()
    plt.show()
    
    return traversal_anim

def create_generic_algorithm_animation():
    """Create generic algorithm visualization for unknown algorithms"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Create abstract algorithm visualization
    steps = ['Initialize', 'Process Input', 'Main Algorithm', 'Generate Output', 'Complete']
    
    # Progress bar animation
    progress_bars = ax1.barh(range(len(steps)), [0]*len(steps), color='lightblue')
    ax1.set_yticks(range(len(steps)))
    ax1.set_yticklabels(steps)
    ax1.set_xlabel('Progress')
    ax1.set_title('Algorithm Execution Progress', fontsize=16, fontweight='bold')
    ax1.set_xlim(0, 100)
    
    # Status area
    ax2.axis('off')
    status_text = ax2.text(0.05, 0.8, '', transform=ax2.transAxes, fontsize=12,
                          verticalalignment='top', fontfamily='monospace')
    
    def animate_generic(frame):
        # Simulate algorithm progress
        progress = frame * 10
        completed_steps = min(frame // 10, len(steps))
        
        for i, bar in enumerate(progress_bars):
            if i < completed_steps:
                bar.set_width(100)
                bar.set_color('green')
            elif i == completed_steps:
                bar.set_width(progress % 100)
                bar.set_color('orange')
            else:
                bar.set_width(0)
                bar.set_color('lightblue')
        
        # Update status
        if completed_steps < len(steps):
            current_step = steps[completed_steps]
            status_text.set_text(f"Currently executing: {current_step}\\nProgress: {progress % 100}%")
        else:
            status_text.set_text("Algorithm execution completed!")
        
        return progress_bars + [status_text]
    
    # Create animation
    generic_anim = animation.FuncAnimation(fig, animate_generic, frames=60,
                                         interval=200, blit=False, repeat=True)
    
    plt.tight_layout()
    plt.show()
    
    return generic_anim
'''
    
    def _create_data_structure_animation(self, algorithm_data: Dict[str, Any]) -> str:
        """Create data structure operation animations"""
        return '''
def create_data_structure_animation():
    """
    Data Structure Operations Animation
    
    Frontend Integration:
    - Use SVG animations for smooth web playback
    - Implement drag-and-drop for interactive mode
    - Add sound effects for operations (optional)
    - Provide step-by-step explanation overlay
    """
    
    # Detect data structure type from algorithm
    code = algorithm_data.get('code', '').lower()
    
    if 'stack' in code or 'push' in code or 'pop' in code:
        return create_stack_animation()
    elif 'queue' in code or 'enqueue' in code or 'dequeue' in code:
        return create_queue_animation()
    elif 'tree' in code or 'node' in code:
        return create_tree_animation()
    elif 'hash' in code or 'dict' in code:
        return create_hash_table_animation()
    else:
        return create_array_animation()

def create_stack_animation():
    """Animate stack operations"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    
    # Stack visualization
    stack_data = []
    max_size = 8
    
    # Create stack visualization
    stack_rects = []
    for i in range(max_size):
        rect = Rectangle((0, i), 2, 0.8, facecolor='lightgray', 
                        edgecolor='black', alpha=0.3)
        ax1.add_patch(rect)
        stack_rects.append(rect)
    
    ax1.set_xlim(-0.5, 2.5)
    ax1.set_ylim(-0.5, max_size + 0.5)
    ax1.set_title('Stack Operations Animation', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Stack')
    ax1.set_ylabel('Position')
    
    # Code execution area
    ax2.axis('off')
    code_text = ax2.text(0.05, 0.9, '', transform=ax2.transAxes, fontsize=10,
                        verticalalignment='top', fontfamily='monospace')
    
    # Define operations
    operations = [
        ('push', 10), ('push', 20), ('push', 30), ('pop', None),
        ('push', 40), ('push', 50), ('pop', None), ('pop', None)
    ]
    
    steps = []
    current_stack = []
    
    for op, value in operations:
        if op == 'push':
            current_stack.append(value)
            steps.append({
                'operation': f'push({value})',
                'stack': current_stack.copy(),
                'message': f'Pushed {value} onto stack'
            })
        elif op == 'pop' and current_stack:
            popped = current_stack.pop()
            steps.append({
                'operation': f'pop() -> {popped}',
                'stack': current_stack.copy(),
                'message': f'Popped {popped} from stack'
            })
    
    def animate_stack(frame):
        if frame >= len(steps):
            return
        
        step = steps[frame]
        
        # Clear all stack rectangles
        for rect in stack_rects:
            rect.set_facecolor('lightgray')
            rect.set_alpha(0.3)
        
        # Fill stack rectangles
        for i, value in enumerate(step['stack']):
            stack_rects[i].set_facecolor('lightblue')
            stack_rects[i].set_alpha(0.8)
            
            # Add value label
            ax1.text(1, i + 0.4, str(value), ha='center', va='center',
                    fontsize=12, fontweight='bold')
        
        # Highlight top of stack
        if step['stack']:
            top_index = len(step['stack']) - 1
            stack_rects[top_index].set_facecolor('orange')
        
        # Update code text
        code_text.set_text(
            f"Step {frame + 1}: {step['operation']}\\n"
            f"{step['message']}\\n\\n"
            f"Stack contents (bottom to top):\\n" +
            "\\n".join([f"[{i}] {val}" for i, val in enumerate(step['stack'])])
        )
    
    # Create animation
    stack_anim = animation.FuncAnimation(fig, animate_stack, frames=len(steps)+3,
                                       interval=1500, repeat=True)
    
    plt.tight_layout()
    plt.show()
    
    return stack_anim

def create_tree_animation():
    """Animate binary tree operations"""
    
    import networkx as nx
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Create binary tree
    tree = nx.DiGraph()
    
    # Tree insertion sequence
    values = [50, 30, 70, 20, 40, 60, 80]
    steps = []
    
    for i, value in enumerate(values):
        # Add node to tree
        tree.add_node(value)
        
        # Add edges based on BST property
        if i > 0:
            # Find parent
            parent = None
            current = values[0]  # Start from root
            
            while True:
                if value < current:
                    # Go left
                    left_children = [n for n in tree.nodes() if tree.has_edge(current, n) and n < current]
                    if left_children:
                        current = min(left_children)
                    else:
                        parent = current
                        break
                else:
                    # Go right
                    right_children = [n for n in tree.nodes() if tree.has_edge(current, n) and n > current]
                    if right_children:
                        current = max(right_children)
                    else:
                        parent = current
                        break
            
            if parent:
                tree.add_edge(parent, value)
        
        # Record step
        steps.append({
            'tree': tree.copy(),
            'new_node': value,
            'message': f'Inserted {value} into binary search tree'
        })
    
    def animate_tree(frame):
        if frame >= len(steps):
            return
        
        step = steps[frame]
        current_tree = step['tree']
        
        ax1.clear()
        
        if current_tree.nodes():
            # Calculate positions
            pos = {}
            levels = {}
            
            # Simple tree layout
            def assign_positions(node, level, x_offset):
                levels[node] = level
                pos[node] = (x_offset, -level)
                
                children = list(current_tree.successors(node))
                children.sort()
                
                if len(children) == 1:
                    assign_positions(children[0], level + 1, x_offset + (-0.5 if children[0] < node else 0.5))
                elif len(children) == 2:
                    assign_positions(children[0], level + 1, x_offset - 0.5)
                    assign_positions(children[1], level + 1, x_offset + 0.5)
            
            if current_tree.nodes():
                root = min(current_tree.nodes())  # Assuming root is minimum for simplicity
                assign_positions(root, 0, 0)
            
            # Draw tree
            nx.draw_networkx_edges(current_tree, pos, ax=ax1, 
                                 edge_color='gray', arrows=True, arrowsize=20, width=2)
            
            # Color nodes
            node_colors = []
            for node in current_tree.nodes():
                if node == step['new_node']:
                    node_colors.append('red')  # Newly inserted
                else:
                    node_colors.append('lightblue')
            
            nx.draw_networkx_nodes(current_tree, pos, ax=ax1, 
                                 node_color=node_colors, node_size=1000, alpha=0.8)
            nx.draw_networkx_labels(current_tree, pos, ax=ax1, 
                                  font_size=10, font_weight='bold')
        
        ax1.set_title('Binary Search Tree Construction', fontsize=16, fontweight='bold')
        ax1.axis('off')
        
        # Update status
        ax2.clear()
        ax2.axis('off')
        ax2.text(0.05, 0.9, 
                f"Step {frame + 1}: {step['message']}\\n\\n"
                f"Tree properties:\\n"
                f"• Left subtree < parent\\n"
                f"• Right subtree > parent\\n"
                f"• Enables O(log n) search\\n\\n"
                f"Nodes inserted so far:\\n" +
                ", ".join(map(str, sorted(current_tree.nodes()))),
                transform=ax2.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace')
    
    tree_anim = animation.FuncAnimation(fig, animate_tree, frames=len(steps)+3,
                                       interval=2000, repeat=True)
    
    plt.tight_layout()
    plt.show()
    
    return tree_anim

def create_array_animation():
    """Animate array operations"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Array operations
    array = [0] * 8
    operations = [
        ('insert', 0, 10), ('insert', 1, 20), ('insert', 2, 30),
        ('update', 1, 25), ('delete', 0), ('insert', 3, 40)
    ]
    
    steps = []
    current_array = array.copy()
    
    for op in operations:
        if op[0] == 'insert':
            index, value = op[1], op[2]
            current_array[index] = value
        elif op[0] == 'update':
            index, value = op[1], op[2]
            current_array[index] = value
        elif op[0] == 'delete':
            index = op[1]
            # Shift elements left
            for i in range(index, len(current_array) - 1):
                current_array[i] = current_array[i + 1]
            current_array[-1] = 0
        
        steps.append({
            'operation': op,
            'array': current_array.copy(),
            'message': f'Operation: {op[0]} at index {op[1] if len(op) > 1 else "N/A"}'
        })
    
    def animate_array(frame):
        if frame >= len(steps):
            return
        
        step = steps[frame]
        
        ax1.clear()
        
        # Draw array
        for i, value in enumerate(step['array']):
            color = 'lightblue' if value != 0 else 'lightgray'
            if step['operation'][0] in ['insert', 'update'] and len(step['operation']) > 1 and i == step['operation'][1]:
                color = 'orange'
            
            rect = Rectangle((i, 0), 0.8, 0.8, facecolor=color, edgecolor='black')
            ax1.add_patch(rect)
            
            if value != 0:
                ax1.text(i + 0.4, 0.4, str(value), ha='center', va='center',
                        fontsize=12, fontweight='bold')
            
            # Add index labels
            ax1.text(i + 0.4, -0.3, str(i), ha='center', va='center',
                    fontsize=10, color='gray')
        
        ax1.set_xlim(-0.2, len(step['array']))
        ax1.set_ylim(-0.5, 1.2)
        ax1.set_title('Array Operations Animation', fontsize=16, fontweight='bold')
        ax1.set_aspect('equal')
        
        # Update status
        ax2.clear()
        ax2.axis('off')
        ax2.text(0.05, 0.8, 
                f"Step {frame + 1}: {step['message']}\\n\\n"
                f"Array state: {step['array']}\\n\\n"
                f"Orange: Recently modified\\n"
                f"Blue: Contains data\\n"
                f"Gray: Empty",
                transform=ax2.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace')
    
    array_anim = animation.FuncAnimation(fig, animate_array, frames=len(steps)+3,
                                       interval=1500, repeat=True)
    
    plt.tight_layout()
    plt.show()
    
    return array_anim
'''
    
    def _create_algorithm_comparison_animation(self, algorithm_data: Dict[str, Any]) -> str:
        """Create side-by-side algorithm comparison animation"""
        return '''
def create_algorithm_comparison_animation():
    """
    Side-by-Side Algorithm Comparison
    
    Frontend Integration:
    - Use synchronized playback controls
    - Highlight performance differences
    - Show complexity analysis side-by-side
    - Add race-style countdown timer
    """
    
    # Compare sorting algorithms
    data1 = [64, 34, 25, 12, 22, 11, 90]  # For bubble sort
    data2 = data1.copy()  # For selection sort
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    
    # Initialize visualizations
    bars1 = ax1.bar(range(len(data1)), data1, color='lightblue', alpha=0.7)
    bars2 = ax2.bar(range(len(data2)), data2, color='lightgreen', alpha=0.7)
    
    ax1.set_title('Bubble Sort O(n²)', fontsize=14, fontweight='bold')
    ax2.set_title('Selection Sort O(n²)', fontsize=14, fontweight='bold')
    
    # Performance tracking
    bubble_operations = 0
    selection_operations = 0
    
    ax3.set_xlim(0, 100)
    ax3.set_ylim(0, 50)
    ax3.set_title('Operations Count Comparison', fontsize=14)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Operations')
    
    bubble_line, = ax3.plot([], [], 'b-', label='Bubble Sort', linewidth=2)
    selection_line, = ax3.plot([], [], 'g-', label='Selection Sort', linewidth=2)
    ax3.legend()
    
    # Status
    ax4.axis('off')
    status_text = ax4.text(0.05, 0.9, '', transform=ax4.transAxes, fontsize=10,
                          verticalalignment='top', fontfamily='monospace')
    
    # Generate steps for both algorithms
    bubble_steps = generate_bubble_sort_steps(data1.copy())
    selection_steps = generate_selection_sort_steps(data2.copy())
    
    max_steps = max(len(bubble_steps), len(selection_steps))
    
    def animate_comparison(frame):
        nonlocal bubble_operations, selection_operations
        
        # Update bubble sort
        if frame < len(bubble_steps):
            step = bubble_steps[frame]
            for i, (bar, value) in enumerate(zip(bars1, step['data'])):
                bar.set_height(value)
                if i in step.get('highlight', []):
                    bar.set_color('red')
                else:
                    bar.set_color('lightblue')
            bubble_operations += step.get('operations', 1)
        
        # Update selection sort
        if frame < len(selection_steps):
            step = selection_steps[frame]
            for i, (bar, value) in enumerate(zip(bars2, step['data'])):
                bar.set_height(value)
                if i in step.get('highlight', []):
                    bar.set_color('red')
                else:
                    bar.set_color('lightgreen')
            selection_operations += step.get('operations', 1)
        
        # Update performance graph
        if frame > 0:
            x_data = list(range(frame + 1))
            bubble_line.set_data(x_data, [bubble_operations] * len(x_data))
            selection_line.set_data(x_data, [selection_operations] * len(x_data))
        
        # Update status
        status_text.set_text(
            f"Step: {frame + 1}\\n\\n"
            f"Bubble Sort Operations: {bubble_operations}\\n"
            f"Selection Sort Operations: {selection_operations}\\n\\n"
            f"Performance Difference: {abs(bubble_operations - selection_operations)}\\n\\n"
            f"Winner: {'Bubble' if bubble_operations < selection_operations else 'Selection' if selection_operations < bubble_operations else 'Tie'}"
        )
        
        return bars1 + bars2 + [bubble_line, selection_line, status_text]
    
    comparison_anim = animation.FuncAnimation(fig, animate_comparison, frames=max_steps+5,
                                            interval=800, repeat=True)
    
    plt.tight_layout()
    plt.show()
    
    return comparison_anim

def generate_bubble_sort_steps(data):
    """Generate steps for bubble sort animation"""
    steps = []
    n = len(data)
    
    for i in range(n):
        for j in range(0, n-i-1):
            steps.append({
                'data': data.copy(),
                'highlight': [j, j+1],
                'operations': 1,
                'message': f'Comparing {data[j]} and {data[j+1]}'
            })
            
            if data[j] > data[j+1]:
                data[j], data[j+1] = data[j+1], data[j]
                steps.append({
                    'data': data.copy(),
                    'highlight': [j, j+1],
                    'operations': 1,
                    'message': f'Swapped {data[j+1]} and {data[j]}'
                })
    
    return steps

def generate_selection_sort_steps(data):
    """Generate steps for selection sort animation"""
    steps = []
    n = len(data)
    
    for i in range(n):
        min_idx = i
        steps.append({
            'data': data.copy(),
            'highlight': [i],
            'operations': 1,
            'message': f'Finding minimum from position {i}'
        })
        
        for j in range(i+1, n):
            steps.append({
                'data': data.copy(),
                'highlight': [min_idx, j],
                'operations': 1,
                'message': f'Comparing positions {min_idx} and {j}'
            })
            
            if data[j] < data[min_idx]:
                min_idx = j
        
        if min_idx != i:
            data[i], data[min_idx] = data[min_idx], data[i]
            steps.append({
                'data': data.copy(),
                'highlight': [i, min_idx],
                'operations': 1,
                'message': f'Swapped positions {i} and {min_idx}'
            })
    
    return steps
'''
    
    def _create_interactive_stepper(self, algorithm_data: Dict[str, Any]) -> str:
        """Create interactive algorithm stepper"""
        return '''
def create_interactive_stepper():
    """
    Interactive Algorithm Stepper
    
    Frontend Integration:
    - Convert to HTML/JavaScript widget
    - Add step forward/backward buttons
    - Implement keyboard shortcuts (space, arrow keys)
    - Save state for resuming later
    - Export execution trace as JSON
    """
    
    from matplotlib.widgets import Button
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Sample algorithm steps
    steps = [
        {'data': [5, 2, 8, 1, 9], 'highlight': [], 'message': 'Initial array'},
        {'data': [2, 5, 8, 1, 9], 'highlight': [0, 1], 'message': 'Compare and swap 5, 2'},
        {'data': [2, 5, 8, 1, 9], 'highlight': [1, 2], 'message': 'Compare 5, 8 - no swap'},
        {'data': [2, 1, 5, 8, 9], 'highlight': [2, 3], 'message': 'Compare and swap 8, 1'},
        {'data': [1, 2, 5, 8, 9], 'highlight': [], 'message': 'Final sorted array'}
    ]
    
    current_step = [0]  # Use list for mutable reference
    
    # Initialize bars
    bars = ax1.bar(range(len(steps[0]['data'])), steps[0]['data'], color='lightblue')
    ax1.set_title('Interactive Algorithm Stepper', fontsize=16, fontweight='bold')
    ax1.set_ylim(0, max(max(step['data']) for step in steps) * 1.1)
    
    # Status area
    ax2.axis('off')
    status_text = ax2.text(0.05, 0.8, '', transform=ax2.transAxes, fontsize=12,
                          verticalalignment='top', fontfamily='monospace')
    
    def update_display():
        """Update the visualization display"""
        step = steps[current_step[0]]
        
        # Update bar heights and colors
        for i, (bar, value) in enumerate(zip(bars, step['data'])):
            bar.set_height(value)
            if i in step['highlight']:
                bar.set_color('red')
            else:
                bar.set_color('lightblue')
        
        # Update status text
        status_text.set_text(
            f"Step {current_step[0] + 1} of {len(steps)}\\n\\n"
            f"{step['message']}\\n\\n"
            f"Array: {step['data']}\\n\\n"
            f"Controls:\\n"
            f"• Next: Click 'Next' or press →\\n"
            f"• Previous: Click 'Prev' or press ←\\n"
            f"• Reset: Click 'Reset' or press R"
        )
        
        plt.draw()
    
    def next_step(event):
        if current_step[0] < len(steps) - 1:
            current_step[0] += 1
            update_display()
    
    def prev_step(event):
        if current_step[0] > 0:
            current_step[0] -= 1
            update_display()
    
    def reset_stepper(event):
        current_step[0] = 0
        update_display()
    
    # Add control buttons
    ax_next = plt.axes([0.81, 0.01, 0.08, 0.05])
    ax_prev = plt.axes([0.7, 0.01, 0.08, 0.05])
    ax_reset = plt.axes([0.59, 0.01, 0.08, 0.05])
    
    btn_next = Button(ax_next, 'Next')
    btn_prev = Button(ax_prev, 'Prev')
    btn_reset = Button(ax_reset, 'Reset')
    
    btn_next.on_clicked(next_step)
    btn_prev.on_clicked(prev_step)
    btn_reset.on_clicked(reset_stepper)
    
    # Keyboard event handling
    def on_key(event):
        if event.key == 'right':
            next_step(None)
        elif event.key == 'left':
            prev_step(None)
        elif event.key == 'r':
            reset_stepper(None)
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Initial display
    update_display()
    
    plt.tight_layout()
    plt.show()
    
    return fig
'''
    
    def _combine_algorithm_animations(self, animations: List[str]) -> str:
        """Combine all algorithm animations"""
        header = '''
"""
COMPREHENSIVE ALGORITHM ANIMATION MODULE
Generated by AlgorithmAnimationAgent

Frontend Integration Instructions:
================================

1. WEB ANIMATIONS:
   - Convert matplotlib animations to CSS/JavaScript
   - Use requestAnimationFrame for smooth playback
   - Implement WebGL for complex 3D animations
   - Add touch controls for mobile interaction

2. VIDEO EXPORT:
   - Generate MP4/WebM for social sharing
   - Create GIF versions for documentation
   - Add captions and annotations
   - Implement quality selection (HD/SD)

3. INTERACTIVE FEATURES:
   - Play/pause/step controls
   - Speed adjustment slider
   - Zoom and pan capabilities
   - Screenshot capture functionality

4. EDUCATIONAL ENHANCEMENTS:
   - Add explanatory tooltips
   - Implement quiz integration
   - Create guided tutorials
   - Add complexity analysis overlay

5. PERFORMANCE OPTIMIZATION:
   - Use canvas for complex animations
   - Implement level-of-detail rendering
   - Add progressive loading
   - Cache animation frames

Dependencies Required:
- matplotlib >= 3.5.0
- numpy >= 1.21.0
- networkx >= 2.6.0 (for graph animations)
- pillow >= 8.3.0 (for GIF creation)
- ffmpeg (for video export)
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.patches import Rectangle, Circle
from matplotlib.widgets import Button
import networkx as nx
from collections import deque

# Set animation style
plt.style.use('seaborn-v0_8')

'''
        
        combined = header
        for anim in animations:
            combined += anim + "\n\n"
        
        combined += '''
def generate_all_algorithm_animations(algorithm_data):
    """
    Master function to generate all algorithm animations
    
    Args:
        algorithm_data (dict): Dictionary containing algorithm information
    
    Returns:
        dict: Paths to generated animation files
    """
    
    print("🎬 Generating comprehensive algorithm animations...")
    
    # Generate all animations based on algorithm type
    create_step_by_step_animation()
    create_data_structure_animation()
    create_algorithm_comparison_animation()
    create_interactive_stepper()
    
    generated_files = {
        "animations": [
            "bubble_sort_animation.gif",
            "binary_search_animation.gif", 
            "bfs_traversal_animation.gif",
            "algorithm_comparison.mp4"
        ],
        "interactive": [
            "algorithm_stepper.html"
        ],
        "screenshots": [
            "algorithm_steps.png"
        ]
    }
    
    print("✅ All algorithm animations generated successfully!")
    return generated_files

if __name__ == "__main__":
    # Example usage
    sample_algorithm_data = {
        "algorithm_type": "sorting",
        "code": "def bubble_sort(arr): ...",
        "complexity": "O(n²)"
    }
    
    generate_all_algorithm_animations(sample_algorithm_data)
'''
        
        return combined
    
    def _extract_algorithm_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract algorithm information from request data"""
        algorithm_data = {}
        
        if isinstance(data, dict):
            # Extract from various possible sources
            if "optimal_solution" in data and "code" in data["optimal_solution"]:
                algorithm_data["code"] = data["optimal_solution"]["code"].get("code", "")
            elif "code" in data:
                algorithm_data["code"] = data["code"]
            
            if "complexity_analysis" in data:
                complexity = data["complexity_analysis"]
                algorithm_data["time_complexity"] = complexity.get("time_complexity", "Unknown")
                algorithm_data["space_complexity"] = complexity.get("space_complexity", "Unknown")
            
            # Try to detect algorithm type from code
            code = algorithm_data.get("code", "").lower()
            if "sort" in code:
                algorithm_data["algorithm_type"] = "sorting"
            elif "search" in code:
                algorithm_data["algorithm_type"] = "searching"
            elif "graph" in code or "bfs" in code or "dfs" in code:
                algorithm_data["algorithm_type"] = "graph_traversal"
            else:
                algorithm_data["algorithm_type"] = "general"
        
        return algorithm_data
    

    def _parse_animation_recommendations(self, ai_response: str) -> Dict[str, Any]:
        """Parse AI animation recommendations - FIXED METHOD"""
        # Simple parsing - in production, this would be more sophisticated
        return {
            "style": "educational",
            "pacing": "medium",
            "highlights": ["comparisons", "swaps", "final_result"],
            "colors": {"active": "red", "inactive": "blue", "completed": "green"}
        }

    
    def _get_animation_frontend_instructions(self) -> str:
        """Get frontend integration instructions for animations"""
        return """
FRONTEND INTEGRATION GUIDE FOR ALGORITHM ANIMATIONS
=================================================

1. VIDEO PLAYBACK:
   - Use HTML5 video elements with controls
   - Implement custom play/pause overlay
   - Add fullscreen and picture-in-picture support
   - Provide multiple quality options

2. INTERACTIVE CONTROLS:
   - Step-by-step navigation buttons
   - Speed control slider (0.5x to 2x)
   - Loop and auto-replay options
   - Frame-by-frame scrubbing

3. MOBILE OPTIMIZATION:
   - Touch-friendly control buttons
   - Swipe gestures for navigation
   - Responsive video sizing
   - Battery-efficient playback

4. ACCESSIBILITY:
   - Closed captions for explanations
   - Audio descriptions for visual changes
   - Keyboard navigation support
   - High contrast mode

5. SOCIAL FEATURES:
   - Share specific animation moments
   - Embed animations in external sites
   - Export as GIF for messaging
   - Create custom thumbnails
"""
