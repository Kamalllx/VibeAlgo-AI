# backend/test_visualization_database.py
#!/usr/bin/env python3
"""
Test and demonstrate the visualization database system
"""

import asyncio
import sys
import os
sys.path.append('.')

from visualization_database.visualization_manager import visualization_manager

async def test_visualization_system():
    """Test the complete visualization system"""
    
    print("🎨 Testing Algorithm Visualization Database System")
    print("=" * 60)
    
    # Test 1: Algorithm Detection
    print("\n🔍 Test 1: Algorithm Detection")
    test_inputs = [
        "How does Dijkstra's algorithm work?",
        "Show me binary search",
        "I need help with bubble sort",
        "Explain the knapsack problem",
        "Tree traversal algorithms"
    ]
    
    for input_text in test_inputs:
        print(f"\nInput: '{input_text}'")
        matches = visualization_manager.detect_algorithm(input_text)
        for category, algorithm, confidence in matches[:3]:
            print(f"  → {category}/{algorithm} (confidence: {confidence:.2f})")
    
    # Test 2: Auto Visualization
    print("\n\n🎯 Test 2: Auto Visualization")
    test_cases = [
        ("Dijkstra's shortest path algorithm", ""),
        ("Binary search in sorted array", "def binary_search(arr, target):"),
        ("Bubble sort implementation", "for i in range(n): for j in range(n-1):")
    ]
    
    for user_input, code in test_cases:
        print(f"\nAuto-visualizing: '{user_input}'")
        result = visualization_manager.auto_visualize(user_input, code)
        print(f"Success: {result}")
    
    # Test 3: List All Algorithms
    print("\n\n📚 Test 3: Available Algorithms")
    all_algorithms = visualization_manager.list_all_algorithms()
    for category, algorithms in all_algorithms.items():
        print(f"\n{category}:")
        for algo in algorithms:
            info = visualization_manager.get_algorithm_info(category, algo)
            print(f"  • {info['name']} - {info['complexity']['time']}")
    
    # Test 4: Search Functionality
    print("\n\n🔍 Test 4: Algorithm Search")
    search_queries = ["sort", "graph", "search", "tree"]
    
    for query in search_queries:
        print(f"\nSearching for: '{query}'")
        results = visualization_manager.search_algorithms(query)
        for category, algo_key, name in results[:3]:
            print(f"  → {name} ({category})")
    
    print("\n" + "=" * 60)
    print("Visualization database system test completed!")

if __name__ == "__main__":
    asyncio.run(test_visualization_system())
