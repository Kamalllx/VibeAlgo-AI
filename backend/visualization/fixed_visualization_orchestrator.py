# backend/visualization/fixed_visualization_orchestrator.py (CORRECTED VERSION)
import os
import subprocess
import sys
from datetime import datetime
import traceback
import json
from typing import Dict, List, Any

class FixedVisualizationOrchestrator:
    def __init__(self):
        self.name = "FixedVisualizationOrchestrator"
        self.output_dir = "generated_visualizations"
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(self.output_dir, f"session_{timestamp}")
        
        # Store original working directory
        self.original_cwd = os.getcwd()
        
        # FIXED: Create directories properly and persistently
        self._ensure_directories_exist()
        
        print(f"🎨 [{self.name}] Fixed Visualization Orchestrator initialized")
        print(f"📁 Output directory: {os.path.abspath(self.session_dir)}")
        print(f"🏠 Original directory: {self.original_cwd}")
    
    def _ensure_directories_exist(self):
        """Ensure all required directories exist"""
        try:
            # Create main output directory
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"✅ Created main directory: {os.path.abspath(self.output_dir)}")
            
            # Create session directory
            os.makedirs(self.session_dir, exist_ok=True)
            print(f"✅ Created session directory: {os.path.abspath(self.session_dir)}")
            
            # Create debug directory
            self.debug_dir = os.path.join(self.session_dir, "debug_code")
            os.makedirs(self.debug_dir, exist_ok=True)
            print(f"✅ Created debug directory: {os.path.abspath(self.debug_dir)}")
            
            # Verify directories exist
            assert os.path.exists(self.session_dir), f"Session dir doesn't exist: {self.session_dir}"
            assert os.path.exists(self.debug_dir), f"Debug dir doesn't exist: {self.debug_dir}"
            
        except Exception as e:
            print(f"❌ Failed to create directories: {e}")
            # Fallback to current directory
            self.session_dir = os.path.join(self.original_cwd, "fallback_visualizations")
            self.debug_dir = os.path.join(self.session_dir, "debug_code")
            os.makedirs(self.session_dir, exist_ok=True)
            os.makedirs(self.debug_dir, exist_ok=True)
            print(f"🔄 Using fallback directory: {self.session_dir}")
    
    async def create_simple_visualization(self, scenario_type: str = "complexity") -> List[str]:
        """Create a simple but working visualization"""
        print(f"\n{'='*60}")
        print(f"🎨 CREATING FIXED VISUALIZATION: {scenario_type}")
        print(f"{'='*60}")
        print(f"📁 Session dir: {os.path.abspath(self.session_dir)}")
        print(f"🐛 Debug dir: {os.path.abspath(self.debug_dir)}")
        
        # FIXED: Ensure directories still exist before each operation
        if not os.path.exists(self.debug_dir):
            print(f"⚠️ Debug directory missing, recreating...")
            os.makedirs(self.debug_dir, exist_ok=True)
        
        # Generate working Python code
        working_code = self._generate_working_visualization_code(scenario_type)
        
        # Save the working code with absolute path
        code_file = os.path.join(self.debug_dir, f"{scenario_type}_working.py")
        abs_code_file = os.path.abspath(code_file)
        
        print(f"💾 Saving working code to: {abs_code_file}")
        
        try:
            with open(abs_code_file, 'w', encoding='utf-8') as f:
                f.write(working_code)
            print(f"✅ Code saved successfully to: {abs_code_file}")
            
            # Verify file was created
            if not os.path.exists(abs_code_file):
                raise FileNotFoundError(f"Code file was not created: {abs_code_file}")
                
        except Exception as e:
            print(f"❌ Failed to save code file: {e}")
            return []
        
        # Execute the working code
        generated_files = self._execute_working_code(working_code, scenario_type)
        
        # FIXED: Properly detect generated files
        actual_files = self._detect_generated_files()
        
        print(f"📁 Total files in session directory: {len(actual_files)}")
        print(f"📋 Files: {actual_files}")
        
        # Open results
        self._open_output_directory()
        
        return actual_files
    
    def _detect_generated_files(self) -> List[str]:
        """Detect all files in the session directory"""
        try:
            files = []
            session_abs = os.path.abspath(self.session_dir)
            
            if os.path.exists(session_abs):
                for item in os.listdir(session_abs):
                    item_path = os.path.join(session_abs, item)
                    if os.path.isfile(item_path) and not item.startswith('.'):
                        files.append(item)
                        print(f"🔍 Found file: {item}")
            
            return files
            
        except Exception as e:
            print(f"❌ Error detecting files: {e}")
            return []
    
    def _execute_working_code(self, code: str, scenario_type: str) -> List[str]:
        """Execute the working code and return generated files - FIXED"""
        print(f"⚡ Executing working code for {scenario_type}...")
        
        # Save current directory
        original_dir = os.getcwd()
        session_abs = os.path.abspath(self.session_dir)
        
        print(f"🏠 Original directory: {original_dir}")
        print(f"📁 Changing to: {session_abs}")
        
        try:
            # Change to session directory for file generation
            os.chdir(session_abs)
            print(f"✅ Changed directory to: {os.getcwd()}")
            
            # Get files before execution
            files_before = set(os.listdir('.')) if os.path.exists('.') else set()
            print(f"📋 Files before: {files_before}")
            
            # Create execution environment
            exec_globals = {
                '__name__': '__main__',
                '__file__': f'{scenario_type}_working.py'
            }
            
            # Execute the code
            print(f"🚀 Executing code...")
            exec(code, exec_globals)
            
            # Check for new files
            files_after = set(os.listdir('.')) if os.path.exists('.') else set()
            new_files = list(files_after - files_before)
            
            print(f"📋 Files after: {files_after}")
            print(f"✅ Code execution successful!")
            print(f"📁 New files generated: {new_files}")
            
            return new_files
            
        except Exception as e:
            print(f"❌ Code execution failed: {e}")
            traceback.print_exc()
            return []
        finally:
            # Always return to original directory
            os.chdir(original_dir)
            print(f"🏠 Returned to original directory: {os.getcwd()}")
    
    def _generate_working_visualization_code(self, scenario_type: str) -> str:
        """Generate guaranteed working Python visualization code"""
        
        if scenario_type == "complexity":
            return self._generate_complexity_code()
        elif scenario_type == "performance":
            return self._generate_performance_code()
        elif scenario_type == "animation":
            return self._generate_animation_code()
        else:
            return self._generate_general_code()
    
    def _generate_complexity_code(self) -> str:
        """Generate working complexity visualization code"""
        return '''
"""
WORKING COMPLEXITY VISUALIZATION
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

print("🎨 Starting complexity visualization generation...")
print(f"📁 Working in: {os.getcwd()}")

def create_complexity_comparison():
    """Create Big O complexity comparison chart"""
    print("📊 Creating complexity comparison...")
    
    # Input sizes
    n_values = np.logspace(1, 3, 50)  # 10 to 1000
    
    # Different complexities
    complexities = {
        'O(1)': np.ones_like(n_values),
        'O(log n)': np.log2(n_values),
        'O(n)': n_values,
        'O(n log n)': n_values * np.log2(n_values),
        'O(n²)': n_values ** 2 / 1000  # Scaled for visibility
    }
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Colors
    colors = ['green', 'blue', 'orange', 'red', 'purple']
    
    # Linear scale plot
    for i, (name, values) in enumerate(complexities.items()):
        if name != 'O(n²)':  # Skip quadratic for linear scale
            ax1.plot(n_values, values, label=name, color=colors[i], linewidth=3)
    
    ax1.set_xlabel('Input Size (n)', fontsize=12)
    ax1.set_ylabel('Operations', fontsize=12)
    ax1.set_title('Algorithm Complexity Comparison (Linear Scale)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Log scale plot
    for i, (name, values) in enumerate(complexities.items()):
        ax2.loglog(n_values, values, label=name, color=colors[i], linewidth=3, marker='o', markersize=4)
    
    ax2.set_xlabel('Input Size (n)', fontsize=12)
    ax2.set_ylabel('Operations (log scale)', fontsize=12)
    ax2.set_title('Algorithm Complexity Comparison (Log Scale)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    filename = 'complexity_comparison_fixed.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"💾 Saved: {filename}")
    
    plt.close()
    return filename

# Execute
if __name__ == "__main__":
    result = create_complexity_comparison()
    print(f"✅ Created: {result}")
'''
    
    def _generate_performance_code(self) -> str:
        """Generate working performance visualization code"""
        return '''
"""
WORKING PERFORMANCE VISUALIZATION
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

print("📈 Starting performance visualization generation...")
print(f"📁 Working in: {os.getcwd()}")

def create_performance_dashboard():
    """Create performance analysis dashboard"""
    print("📈 Creating performance dashboard...")
    
    # Create 2x2 subplot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Execution Time Comparison
    algorithms = ['Bubble\\nSort', 'Quick\\nSort', 'Merge\\nSort', 'Heap\\nSort']
    times = [100, 15, 18, 20]
    colors = ['red', 'green', 'blue', 'orange']
    
    bars1 = ax1.bar(algorithms, times, color=colors, alpha=0.7)
    ax1.set_title('Execution Time Comparison', fontweight='bold')
    ax1.set_ylabel('Time (ms)')
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height}ms', ha='center', va='bottom')
    
    # 2. Memory Usage
    sizes = [10, 50, 100, 500, 1000]
    memory_usage = [size * 0.8 for size in sizes]
    
    ax2.plot(sizes, memory_usage, 'bo-', linewidth=2, markersize=8)
    ax2.set_title('Memory Usage vs Input Size', fontweight='bold')
    ax2.set_xlabel('Input Size')
    ax2.set_ylabel('Memory (MB)')
    ax2.grid(True, alpha=0.3)
    
    # 3. CPU Usage
    time_points = np.arange(0, 30, 1)
    cpu_usage = 50 + 20 * np.sin(time_points * 0.3) + np.random.normal(0, 5, len(time_points))
    cpu_usage = np.clip(cpu_usage, 0, 100)
    
    ax3.plot(time_points, cpu_usage, 'r-', linewidth=2)
    ax3.fill_between(time_points, cpu_usage, alpha=0.3, color='red')
    ax3.set_title('CPU Usage Over Time', fontweight='bold')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('CPU Usage (%)')
    ax3.set_ylim(0, 100)
    ax3.grid(True, alpha=0.3)
    
    # 4. Efficiency Comparison
    efficiency_data = [85, 95, 92, 88]
    efficiency_colors = ['lightcoral', 'lightgreen', 'lightblue', 'gold']
    
    wedges, texts, autotexts = ax4.pie(efficiency_data, labels=algorithms, 
                                      colors=efficiency_colors, autopct='%1.1f%%')
    ax4.set_title('Algorithm Efficiency Distribution', fontweight='bold')
    
    plt.tight_layout()
    
    filename = 'performance_dashboard_fixed.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"💾 Saved: {filename}")
    
    plt.close()
    return filename

# Execute
if __name__ == "__main__":
    result = create_performance_dashboard()
    print(f"✅ Created: {result}")
'''
    
    def _generate_animation_code(self) -> str:
        """Generate working animation code (static frames)"""
        return '''
"""
WORKING ANIMATION VISUALIZATION (Static Frames)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

print("🎬 Starting animation visualization generation...")
print(f"📁 Working in: {os.getcwd()}")

def create_sorting_animation_frames():
    """Create sorting algorithm animation frames"""
    print("🎬 Creating animation frames...")
    
    # Sample data to sort
    data = [64, 34, 25, 12, 22, 11, 90]
    
    # Create frames showing sorting steps
    frames = []
    current_data = data.copy()
    n = len(current_data)
    
    # Bubble sort with step recording
    for i in range(n):
        for j in range(0, n-i-1):
            # Record current state
            frames.append({
                'data': current_data.copy(),
                'comparing': [j, j+1],
                'step': len(frames)
            })
            
            # Perform swap if needed
            if current_data[j] > current_data[j+1]:
                current_data[j], current_data[j+1] = current_data[j+1], current_data[j]
                frames.append({
                    'data': current_data.copy(),
                    'swapped': [j, j+1],
                    'step': len(frames)
                })
    
    # Create visualization of key frames
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Show 6 key frames
    key_frames = [0, len(frames)//5, 2*len(frames)//5, 3*len(frames)//5, 4*len(frames)//5, len(frames)-1]
    
    for idx, frame_num in enumerate(key_frames):
        if idx < len(axes):
            frame = frames[frame_num]
            ax = axes[idx]
            
            # Create bars
            bars = ax.bar(range(len(frame['data'])), frame['data'], 
                         color=['red' if i in frame.get('comparing', []) or i in frame.get('swapped', []) 
                               else 'lightblue' for i in range(len(frame['data']))])
            
            # Add value labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       str(frame['data'][i]), ha='center', va='bottom', fontweight='bold')
            
            ax.set_title(f'Step {frame["step"]}', fontweight='bold')
            ax.set_ylim(0, max(data) * 1.2)
            ax.set_xlabel('Array Index')
            ax.set_ylabel('Value')
    
    plt.suptitle('Bubble Sort Algorithm Animation Frames', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    filename = 'sorting_animation_frames_fixed.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"💾 Saved: {filename}")
    
    plt.close()
    return filename

# Execute
if __name__ == "__main__":
    result = create_sorting_animation_frames()
    print(f"✅ Created: {result}")
'''
    
    def _generate_general_code(self) -> str:
        """Generate general working visualization code"""
        return '''
"""
WORKING GENERAL VISUALIZATION
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

print("🎨 Starting general visualization generation...")
print(f"📁 Working in: {os.getcwd()}")

def create_general_dashboard():
    """Create general algorithm visualization dashboard"""
    print("🎨 Creating general dashboard...")
    
    # Create a comprehensive dashboard
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Algorithm Complexity Growth
    x = np.linspace(1, 100, 100)
    ax1.plot(x, x, label='O(n)', linewidth=2, color='green')
    ax1.plot(x, x * np.log2(x), label='O(n log n)', linewidth=2, color='blue')
    ax1.plot(x, x**2 / 50, label='O(n²)', linewidth=2, color='red')
    ax1.set_title('Algorithm Complexity Growth', fontweight='bold')
    ax1.set_xlabel('Input Size')
    ax1.set_ylabel('Operations')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Data Structure Comparison
    structures = ['Array', 'LinkedList', 'HashTable', 'BST']
    access_times = [1, 5, 1, 3]
    colors = ['red', 'orange', 'green', 'blue']
    
    ax2.bar(structures, access_times, color=colors, alpha=0.7)
    ax2.set_title('Data Structure Access Time', fontweight='bold')
    ax2.set_ylabel('Time Complexity')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # 3. Algorithm Efficiency
    algorithms = ['Linear Search', 'Binary Search', 'Hash Lookup']
    efficiency = [30, 85, 95]
    
    bars = ax3.barh(algorithms, efficiency, color=['red', 'orange', 'green'])
    ax3.set_title('Algorithm Efficiency', fontweight='bold')
    ax3.set_xlabel('Efficiency Score')
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax3.text(width + 1, bar.get_y() + bar.get_height()/2,
                f'{width}%', ha='left', va='center')
    
    # 4. Algorithm Categories
    categories = ['Sorting', 'Searching', 'Graph', 'DP']
    sizes = [30, 25, 25, 20]
    colors = ['gold', 'lightcoral', 'lightgreen', 'lightblue']
    
    ax4.pie(sizes, labels=categories, colors=colors, autopct='%1.1f%%')
    ax4.set_title('Algorithm Categories', fontweight='bold')
    
    plt.suptitle('Algorithm Intelligence Dashboard', fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    filename = 'general_dashboard_fixed.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"💾 Saved: {filename}")
    
    plt.close()
    return filename

# Execute
if __name__ == "__main__":
    result = create_general_dashboard()
    print(f"✅ Created: {result}")
'''
    
    def _open_output_directory(self):
        """Open the output directory"""
        try:
            import platform
            import subprocess
            
            abs_path = os.path.abspath(self.session_dir)
            system = platform.system()
            
            if system == "Windows":
                subprocess.run(f'explorer "{abs_path}"', shell=True)
            elif system == "Darwin":
                subprocess.run(["open", abs_path])
            elif system == "Linux":
                subprocess.run(["xdg-open", abs_path])
            
            print(f"📁 Opened directory: {abs_path}")
            
        except Exception as e:
            print(f"⚠️ Could not open directory: {e}")
            print(f"📁 Manual path: {os.path.abspath(self.session_dir)}")

# Create global instance
fixed_orchestrator = FixedVisualizationOrchestrator()
