# backend/visualization_database/visualizations_strings.py
#!/usr/bin/env python3
"""
String Algorithm Visualizations
All string algorithm visualization implementations
"""

import os
from pathlib import Path

def create_string_visualizations():
    """Create all string algorithm visualization files"""
    
    base_dir = Path("visualizations/strings")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Rabin-Karp Algorithm
    rabin_karp_code = '''#!/usr/bin/env python3
"""
Rabin-Karp Algorithm - Rolling Hash Pattern Matching
Shows hash-based string matching with collision handling
"""

import matplotlib.pyplot as plt
import numpy as np

def create_rabin_karp_visualization():
    """Create Rabin-Karp algorithm visualization"""
    
    text = "ABAAABABABCABABA"
    pattern = "ABABA"
    base = 256
    prime = 101
    
    def hash_function(s, length):
        """Calculate hash value for string"""
        result = 0
        for i in range(length):
            result = (result * base + ord(s[i])) % prime
        return result
    
    def rolling_hash(old_hash, old_char, new_char, base_power):
        """Calculate new hash using rolling hash technique"""
        new_hash = (old_hash - ord(old_char) * base_power) % prime
        new_hash = (new_hash * base + ord(new_char)) % prime
        return new_hash
    
    # Rabin-Karp algorithm with visualization steps
    pattern_hash = hash_function(pattern, len(pattern))
    text_length = len(text)
    pattern_length = len(pattern)
    base_power = pow(base, pattern_length - 1, prime)
    
    steps = []
    matches = []
    
    # Calculate hash for first window
    current_hash = hash_function(text, pattern_length)
    
    for i in range(text_length - pattern_length + 1):
        # Record step
        window = text[i:i + pattern_length]
        hash_match = current_hash == pattern_hash
        
        if hash_match:
            # Verify actual string match
            actual_match = window == pattern
            if actual_match:
                matches.append(i)
        else:
            actual_match = False
        
        steps.append({
            'position': i,
            'window': window,
            'window_hash': current_hash,
            'pattern_hash': pattern_hash,
            'hash_match': hash_match,
            'actual_match': actual_match,
            'collision': hash_match and not actual_match
        })
        
        # Calculate rolling hash for next position
        if i < text_length - pattern_length:
            current_hash = rolling_hash(
                current_hash, 
                text[i], 
                text[i + pattern_length], 
                base_power
            )
    
    # Create visualization
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    for step_idx, step in enumerate(steps[:12]):
        ax = axes[step_idx]
        
        # Draw text
        for i, char in enumerate(text):
            color = 'lightgray'
            if step['position'] <= i < step['position'] + pattern_length:
                if step['hash_match']:
                    color = 'green' if step['actual_match'] else 'orange'  # Orange for collision
                else:
                    color = 'lightblue'
            
            ax.text(i, 1, char, ha='center', va='center', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=color))
        
        # Draw pattern
        for i, char in enumerate(pattern):
            ax.text(i, 0, char, ha='center', va='center', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow'))
        
        # Hash information
        hash_info = f"Window: {step['window']}\\n"
        hash_info += f"Window Hash: {step['window_hash']}\\n"
        hash_info += f"Pattern Hash: {step['pattern_hash']}\\n"
        
        if step['collision']:
            hash_info += "COLLISION!"
        elif step['actual_match']:
            hash_info += "MATCH FOUND!"
        elif step['hash_match']:
            hash_info += "Hash Match - Verifying..."
        else:
            hash_info += "No Hash Match"
        
        ax.text(0.02, 0.7, hash_info, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", 
                         facecolor='lightgreen' if step['actual_match'] else 'lightyellow'))
        
        ax.set_xlim(-0.5, len(text) - 0.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_title(f'Step {step_idx + 1}: Position {step["position"]}', fontweight='bold')
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Pattern', 'Text'])
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(steps), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Rabin-Karp Algorithm: Find "{pattern}" in "{text}"', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    filename = 'rabin_karp.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"SUCCESS: Rabin-Karp visualization saved as {filename}")
    print(f"INFO: Matches found at positions: {matches}")
    return filename

if __name__ == "__main__":
    create_rabin_karp_visualization()
'''
    
    with open(base_dir / "rabin_karp.py", "w") as f:
        f.write(rabin_karp_code)
    
    print("SUCCESS: Created string visualization files")
