# backend/visualization_database/mongodb_manager.py
#!/usr/bin/env python3
"""
MongoDB-powered Visualization Database Manager
Stores algorithms, visualizations, and execution results in MongoDB
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pymongo import MongoClient
from bson import ObjectId
import gridfs  # This is correct - comes with pymongo

class MongoDBVisualizationManager:
    def __init__(self, connection_string: str = None):
        self.connection_string ="mongodb+srv://kamalkarteek1:rvZSeyVHhgOd2fbE@gbh.iliw2.mongodb.net/"
        self.client = None
        self.db = None
        self.fs = None  # GridFS for storing visualization files
        
        self.connect()
        self.setup_collections()
        
        print(f"SUCCESS: MongoDB Visualization Manager initialized")
        print(f"INFO: Connected to: {self.connection_string}")
    
    def connect(self):
        """Connect to MongoDB"""
        try:
            self.client = MongoClient(self.connection_string)
            self.db = self.client.algorithm_intelligence_suite
            
            # Initialize GridFS - CORRECT way
            self.fs = gridfs.GridFS(self.db)
            
            # Test connection
            self.client.admin.command('ping')
            print("SUCCESS: MongoDB connection successful")
            
        except Exception as e:
            print(f"ERROR: MongoDB connection failed: {e}")
            raise
    
    def setup_collections(self):
        """Setup MongoDB collections and indexes"""
        try:
            # Create collections
            self.algorithms = self.db.algorithms
            self.visualizations = self.db.visualizations
            self.execution_results = self.db.execution_results
            self.user_queries = self.db.user_queries
            
            # Create indexes for better performance
            self.algorithms.create_index([("category", 1), ("algorithm_key", 1)])
            self.algorithms.create_index([("name", "text"), ("description", "text")])
            self.visualizations.create_index([("algorithm_id", 1), ("visualization_type", 1)])
            self.execution_results.create_index([("algorithm_id", 1), ("timestamp", -1)])
            self.user_queries.create_index([("query", "text")])
            
            print("SUCCESS: MongoDB collections and indexes created")
            
        except Exception as e:
            print(f"❌ Error setting up collections: {e}")
    
    def populate_algorithms_database(self):
        """Populate MongoDB with comprehensive algorithm database"""
        
        # Comprehensive algorithm database (expanded from original)
        algorithms_data = {
            "searching_algorithms": {
                "linear_search": {
                    "name": "Linear Search",
                    "description": "Sequential search through array elements",
                    "complexity": {"time": "O(n)", "space": "O(1)"},
                    "difficulty": "Basic",
                    "visualization_types": ["animation", "bar_chart", "step_by_step"],
                    "keywords": ["linear", "sequential", "search", "find", "index"],
                    "use_cases": ["unsorted arrays", "small datasets", "simple search"]
                },
                "binary_search": {
                    "name": "Binary Search", 
                    "description": "Divide and conquer search in sorted arrays",
                    "complexity": {"time": "O(log n)", "space": "O(1)"},
                    "difficulty": "Intermediate",
                    "visualization_types": ["animation", "tree", "complexity_graph"],
                    "keywords": ["binary", "sorted", "divide", "conquer", "logarithmic"],
                    "use_cases": ["sorted arrays", "large datasets", "efficient search"]
                },
                "ternary_search": {
                    "name": "Ternary Search",
                    "description": "Find maximum/minimum in unimodal function using divide by 3",
                    "complexity": {"time": "O(log n)", "space": "O(1)"},
                    "difficulty": "Advanced",
                    "visualization_types": ["function_plot", "animation", "mathematical"],
                    "keywords": ["ternary", "unimodal", "optimization", "divide"],
                    "use_cases": ["optimization problems", "unimodal functions"]
                }
            },
            "sorting_algorithms": {
                "bubble_sort": {
                    "name": "Bubble Sort",
                    "description": "Sort by repeatedly swapping adjacent elements",
                    "complexity": {"time": "O(n²)", "space": "O(1)"},
                    "difficulty": "Basic",
                    "visualization_types": ["animation", "comparison", "heatmap"],
                    "keywords": ["bubble", "swap", "adjacent", "simple"],
                    "use_cases": ["educational purposes", "small arrays"]
                },
                "selection_sort": {
                    "name": "Selection Sort",
                    "description": "Find minimum element and place it at beginning",
                    "complexity": {"time": "O(n²)", "space": "O(1)"},
                    "difficulty": "Basic",
                    "visualization_types": ["animation", "selection_view", "comparison"],
                    "keywords": ["selection", "minimum", "in-place"],
                    "use_cases": ["educational purposes", "memory-constrained systems"]
                },
                "insertion_sort": {
                    "name": "Insertion Sort",
                    "description": "Build sorted array one element at a time",
                    "complexity": {"time": "O(n²)", "space": "O(1)"},
                    "difficulty": "Basic",
                    "visualization_types": ["animation", "insertion_view", "adaptive"],
                    "keywords": ["insertion", "adaptive", "stable"],
                    "use_cases": ["small arrays", "nearly sorted data"]
                },
                "merge_sort": {
                    "name": "Merge Sort",
                    "description": "Divide and conquer with merging sorted halves",
                    "complexity": {"time": "O(n log n)", "space": "O(n)"},
                    "difficulty": "Intermediate",
                    "visualization_types": ["animation", "tree", "divide_conquer"],
                    "keywords": ["merge", "divide", "conquer", "stable"],
                    "use_cases": ["large datasets", "stable sorting", "external sorting"]
                },
                "quick_sort": {
                    "name": "Quick Sort",
                    "description": "Divide and conquer with pivot partitioning",
                    "complexity": {"time": "O(n log n)", "space": "O(log n)"},
                    "difficulty": "Intermediate",
                    "visualization_types": ["animation", "partition_view", "tree"],
                    "keywords": ["quick", "pivot", "partition", "in-place"],
                    "use_cases": ["general purpose", "large datasets", "in-place sorting"]
                },
                "heap_sort": {
                    "name": "Heap Sort",
                    "description": "Sort using binary heap data structure",
                    "complexity": {"time": "O(n log n)", "space": "O(1)"},
                    "difficulty": "Advanced",
                    "visualization_types": ["heap_view", "animation", "tree"],
                    "keywords": ["heap", "binary", "tree", "in-place"],
                    "use_cases": ["guaranteed O(n log n)", "memory-constrained"]
                }
            },
            "graphs": {
                "dijkstra": {
                    "name": "Dijkstra's Algorithm",
                    "description": "Find shortest path in weighted graph with non-negative weights",
                    "complexity": {"time": "O((V+E)log V)", "space": "O(V)"},
                    "difficulty": "Advanced",
                    "visualization_types": ["graph_animation", "table", "path_finding"],
                    "keywords": ["dijkstra", "shortest", "path", "weighted", "priority"],
                    "use_cases": ["GPS navigation", "network routing", "shortest path"]
                },
                "bfs": {
                    "name": "Breadth-First Search",
                    "description": "Explore graph level by level using queue",
                    "complexity": {"time": "O(V+E)", "space": "O(V)"},
                    "difficulty": "Intermediate",
                    "visualization_types": ["graph_animation", "level_order", "queue_view"],
                    "keywords": ["breadth", "first", "queue", "level", "shortest"],
                    "use_cases": ["shortest path unweighted", "level order traversal"]
                },
                "dfs": {
                    "name": "Depth-First Search",
                    "description": "Explore graph by going deep first using stack",
                    "complexity": {"time": "O(V+E)", "space": "O(V)"},
                    "difficulty": "Intermediate",
                    "visualization_types": ["graph_animation", "stack_view", "recursive"],
                    "keywords": ["depth", "first", "stack", "recursive", "explore"],
                    "use_cases": ["topological sort", "cycle detection", "path finding"]
                },
                "bellman_ford": {
                    "name": "Bellman-Ford Algorithm",
                    "description": "Find shortest path with negative weight edges",
                    "complexity": {"time": "O(VE)", "space": "O(V)"},
                    "difficulty": "Advanced",
                    "visualization_types": ["graph_animation", "table", "relaxation"],
                    "keywords": ["bellman", "ford", "negative", "weights", "relaxation"],
                    "use_cases": ["negative weights", "currency arbitrage", "network"]
                },
                "floyd_warshall": {
                    "name": "Floyd-Warshall Algorithm",
                    "description": "All-pairs shortest path using dynamic programming",
                    "complexity": {"time": "O(V³)", "space": "O(V²)"},
                    "difficulty": "Advanced",
                    "visualization_types": ["matrix_animation", "dp_table", "heatmap"],
                    "keywords": ["floyd", "warshall", "all-pairs", "shortest", "matrix"],
                    "use_cases": ["all-pairs shortest path", "transitive closure"]
                },
                "kruskals": {
                    "name": "Kruskal's Algorithm",
                    "description": "Find Minimum Spanning Tree using Union-Find",
                    "complexity": {"time": "O(E log E)", "space": "O(V)"},
                    "difficulty": "Advanced",
                    "visualization_types": ["graph_animation", "union_find", "tree_growth"],
                    "keywords": ["kruskal", "mst", "spanning", "tree", "union"],
                    "use_cases": ["network design", "clustering", "MST"]
                },
                "prims": {
                    "name": "Prim's Algorithm",
                    "description": "Find Minimum Spanning Tree using priority queue",
                    "complexity": {"time": "O(E log V)", "space": "O(V)"},
                    "difficulty": "Advanced",
                    "visualization_types": ["graph_animation", "priority_queue", "tree_growth"],
                    "keywords": ["prim", "mst", "spanning", "tree", "priority"],
                    "use_cases": ["network design", "clustering", "MST"]
                }
            },
            "dynamic_programming": {
                "fibonacci": {
                    "name": "Fibonacci with Memoization",
                    "description": "Calculate Fibonacci numbers with caching to avoid recomputation",
                    "complexity": {"time": "O(n)", "space": "O(n)"},
                    "difficulty": "Basic",
                    "visualization_types": ["recursion_tree", "dp_table", "optimization"],
                    "keywords": ["fibonacci", "memoization", "cache", "recursive"],
                    "use_cases": ["introduction to DP", "optimization problems"]
                },
                "knapsack_01": {
                    "name": "0/1 Knapsack",
                    "description": "Maximize value within weight constraint using DP",
                    "complexity": {"time": "O(nW)", "space": "O(nW)"},
                    "difficulty": "Advanced",
                    "visualization_types": ["dp_table", "heatmap", "selection_trace"],
                    "keywords": ["knapsack", "weight", "value", "optimize", "dp"],
                    "use_cases": ["resource allocation", "portfolio optimization"]
                },
                "longest_increasing_subsequence": {
                    "name": "Longest Increasing Subsequence",
                    "description": "Find longest increasing subsequence in array",
                    "complexity": {"time": "O(n log n)", "space": "O(n)"},
                    "difficulty": "Intermediate",
                    "visualization_types": ["array_view", "dp_table", "subsequence"],
                    "keywords": ["longest", "increasing", "subsequence", "lis"],
                    "use_cases": ["sequence analysis", "optimization"]
                },
                "edit_distance": {
                    "name": "Edit Distance",
                    "description": "Minimum operations to transform one string to another",
                    "complexity": {"time": "O(mn)", "space": "O(mn)"},
                    "difficulty": "Intermediate",
                    "visualization_types": ["dp_table", "string_alignment", "operations"],
                    "keywords": ["edit", "distance", "levenshtein", "transform"],
                    "use_cases": ["spell checking", "DNA sequencing", "diff tools"]
                }
            },
            "trees": {
                "tree_traversals": {
                    "name": "Tree Traversals",
                    "description": "Inorder, Preorder, Postorder traversals of binary tree",
                    "complexity": {"time": "O(n)", "space": "O(h)"},
                    "difficulty": "Intermediate",
                    "visualization_types": ["tree_animation", "traversal_order", "recursive"],
                    "keywords": ["inorder", "preorder", "postorder", "traversal"],
                    "use_cases": ["tree processing", "expression evaluation"]
                },
                "bst_operations": {
                    "name": "BST Operations",
                    "description": "Insert, delete, search in Binary Search Tree",
                    "complexity": {"time": "O(log n)", "space": "O(1)"},
                    "difficulty": "Intermediate",
                    "visualization_types": ["tree_animation", "bst_property", "operations"],
                    "keywords": ["bst", "binary", "search", "tree", "insert", "delete"],
                    "use_cases": ["ordered data", "searching", "range queries"]
                },
                "avl_tree": {
                    "name": "AVL Tree Operations",
                    "description": "Self-balancing binary search tree with rotations",
                    "complexity": {"time": "O(log n)", "space": "O(1)"},
                    "difficulty": "Advanced",
                    "visualization_types": ["tree_animation", "rotation", "balance"],
                    "keywords": ["avl", "self-balancing", "rotation", "height"],
                    "use_cases": ["guaranteed O(log n)", "frequent insertions/deletions"]
                }
            },
            "strings": {
                "kmp_algorithm": {
                    "name": "KMP Pattern Matching",
                    "description": "Efficient string pattern matching using failure function",
                    "complexity": {"time": "O(n+m)", "space": "O(m)"},
                    "difficulty": "Advanced",
                    "visualization_types": ["string_matching", "failure_function", "animation"],
                    "keywords": ["kmp", "pattern", "matching", "failure", "prefix"],
                    "use_cases": ["text search", "DNA matching", "plagiarism detection"]
                },
                "rabin_karp": {
                    "name": "Rabin-Karp Algorithm",
                    "description": "Pattern matching using rolling hash",
                    "complexity": {"time": "O(nm)", "space": "O(1)"},
                    "difficulty": "Intermediate",
                    "visualization_types": ["hash_rolling", "pattern_matching", "collision"],
                    "keywords": ["rabin", "karp", "rolling", "hash", "pattern"],
                    "use_cases": ["multiple pattern search", "plagiarism detection"]
                },
                "longest_palindromic_substring": {
                    "name": "Longest Palindromic Substring",
                    "description": "Find longest palindrome in string using expand around centers",
                    "complexity": {"time": "O(n²)", "space": "O(1)"},
                    "difficulty": "Intermediate",
                    "visualization_types": ["string_view", "center_expansion", "palindrome"],
                    "keywords": ["palindrome", "longest", "substring", "center"],
                    "use_cases": ["string analysis", "bioinformatics"]
                }
            },
            "linked_lists": {
                "reverse_linked_list": {
                    "name": "Reverse Linked List",
                    "description": "Reverse the direction of linked list pointers",
                    "complexity": {"time": "O(n)", "space": "O(1)"},
                    "difficulty": "Basic",
                    "visualization_types": ["pointer_animation", "step_by_step", "reversal"],
                    "keywords": ["reverse", "linked", "list", "pointers"],
                    "use_cases": ["list manipulation", "data structure operations"]
                },
                "detect_cycle": {
                    "name": "Detect Cycle in Linked List",
                    "description": "Find if linked list has cycle using Floyd's tortoise and hare",
                    "complexity": {"time": "O(n)", "space": "O(1)"},
                    "difficulty": "Intermediate",
                    "visualization_types": ["two_pointer", "cycle_animation", "floyd"],
                    "keywords": ["cycle", "detection", "floyd", "tortoise", "hare"],
                    "use_cases": ["cycle detection", "infinite loop prevention"]
                },
                "merge_sorted_lists": {
                    "name": "Merge Two Sorted Linked Lists",
                    "description": "Merge two sorted linked lists into one sorted list",
                    "complexity": {"time": "O(n+m)", "space": "O(1)"},
                    "difficulty": "Basic",
                    "visualization_types": ["merge_animation", "pointer_view", "comparison"],
                    "keywords": ["merge", "sorted", "lists", "combine"],
                    "use_cases": ["merge operations", "sorted data combination"]
                }
            },
            "stacks_queues": {
                "balanced_parentheses": {
                    "name": "Balanced Parentheses",
                    "description": "Check if parentheses are balanced using stack",
                    "complexity": {"time": "O(n)", "space": "O(n)"},
                    "difficulty": "Basic",
                    "visualization_types": ["stack_animation", "character_by_character", "matching"],
                    "keywords": ["balanced", "parentheses", "stack", "matching"],
                    "use_cases": ["expression validation", "compiler design"]
                },
                "next_greater_element": {
                    "name": "Next Greater Element",
                    "description": "Find next greater element for each array element using stack",
                    "complexity": {"time": "O(n)", "space": "O(n)"},
                    "difficulty": "Intermediate",
                    "visualization_types": ["stack_animation", "array_view", "greater_element"],
                    "keywords": ["next", "greater", "element", "stack", "monotonic"],
                    "use_cases": ["stock span problem", "histogram analysis"]
                },
                "sliding_window_maximum": {
                    "name": "Sliding Window Maximum",
                    "description": "Find maximum in sliding window using deque",
                    "complexity": {"time": "O(n)", "space": "O(k)"},
                    "difficulty": "Advanced",
                    "visualization_types": ["window_animation", "deque_view", "maximum"],
                    "keywords": ["sliding", "window", "maximum", "deque"],
                    "use_cases": ["time series analysis", "real-time processing"]
                }
            },
            "math_number_theory": {
                "sieve_of_eratosthenes": {
                    "name": "Sieve of Eratosthenes",
                    "description": "Find all prime numbers up to given limit",
                    "complexity": {"time": "O(n log log n)", "space": "O(n)"},
                    "difficulty": "Intermediate",
                    "visualization_types": ["grid_animation", "prime_marking", "sieve"],
                    "keywords": ["sieve", "eratosthenes", "prime", "numbers"],
                    "use_cases": ["prime generation", "cryptography", "number theory"]
                },
                "euclidean_gcd": {
                    "name": "Euclidean Algorithm",
                    "description": "Find Greatest Common Divisor using Euclidean algorithm",
                    "complexity": {"time": "O(log min(a,b))", "space": "O(1)"},
                    "difficulty": "Basic",
                    "visualization_types": ["step_by_step", "division", "remainder"],
                    "keywords": ["euclidean", "gcd", "greatest", "common", "divisor"],
                    "use_cases": ["fraction simplification", "number theory"]
                },
                "fast_exponentiation": {
                    "name": "Fast Exponentiation",
                    "description": "Calculate a^n efficiently using binary exponentiation",
                    "complexity": {"time": "O(log n)", "space": "O(1)"},
                    "difficulty": "Intermediate",
                    "visualization_types": ["binary_view", "exponentiation", "optimization"],
                    "keywords": ["fast", "exponentiation", "binary", "power"],
                    "use_cases": ["cryptography", "large number computation"]
                }
            }
        }
        
        # Clear existing data
        self.algorithms.delete_many({})
        
        # Insert algorithms into MongoDB
        for category, algorithms in algorithms_data.items():
            for algo_key, algo_data in algorithms.items():
                document = {
                    "category": category,
                    "algorithm_key": algo_key,
                    "name": algo_data["name"],
                    "description": algo_data["description"],
                    "complexity": algo_data["complexity"],
                    "difficulty": algo_data["difficulty"],
                    "visualization_types": algo_data["visualization_types"],
                    "keywords": algo_data["keywords"],
                    "use_cases": algo_data["use_cases"],
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                }
                
                result = self.algorithms.insert_one(document)
                print(f"Inserted {algo_data['name']} with ID: {result.inserted_id}")
        
        print(f"🗄️ Database populated with {self.algorithms.count_documents({})} algorithms")
    
    def detect_algorithm(self, user_input: str, generated_code: str = "", rag_context: List[Dict] = None) -> List[Tuple[str, str, str, float]]:
        """
        Detect algorithms using MongoDB text search and context matching
        Returns list of (category, algorithm_key, name, confidence) tuples
        """
        matches = []
        user_input_lower = user_input.lower()
        code_lower = generated_code.lower() if generated_code else ""
        
        # Text search in MongoDB
        text_results = self.algorithms.find(
            {"$text": {"$search": user_input}},
            {"score": {"$meta": "textScore"}}
        ).sort([("score", {"$meta": "textScore"})])
        
        # Add text search results
        for result in text_results:
            confidence = result.get("score", 0) * 0.3  # Scale text score
            matches.append((
                result["category"],
                result["algorithm_key"], 
                result["name"],
                confidence
            ))
        
        # Keyword matching for all algorithms
        all_algorithms = self.algorithms.find({})
        
        for algo in all_algorithms:
            confidence = 0.0
            
            # Check if already added from text search
            existing = any(m[1] == algo["algorithm_key"] for m in matches)
            if existing:
                continue
            
            # Algorithm name matching
            if algo["name"].lower() in user_input_lower:
                confidence += 0.8
            
            # Keywords matching
            for keyword in algo.get("keywords", []):
                if keyword in user_input_lower:
                    confidence += 0.2
            
            # Code pattern matching
            if code_lower:
                for keyword in algo.get("keywords", []):
                    if keyword in code_lower:
                        confidence += 0.3
            
            # RAG context matching
            if rag_context:
                for doc in rag_context:
                    doc_content = doc.get("content", "").lower()
                    for keyword in algo.get("keywords", []):
                        if keyword in doc_content:
                            confidence += 0.1
            
            if confidence > 0.3:
                matches.append((
                    algo["category"],
                    algo["algorithm_key"],
                    algo["name"], 
                    confidence
                ))
        
        # Sort by confidence and return top matches
        matches.sort(key=lambda x: x[3], reverse=True)
        return matches[:5]
    
    def store_visualization_file(self, file_path: str, algorithm_id: str, viz_type: str) -> str:
        """Store visualization file in GridFS"""
        try:
            with open(file_path, 'rb') as f:
                file_id = self.fs.put(
                    f,
                    filename=os.path.basename(file_path),
                    algorithm_id=algorithm_id,
                    visualization_type=viz_type,
                    content_type="image/png",
                    upload_date=datetime.utcnow()
                )
            
            return str(file_id)
            
        except Exception as e:
            print(f"❌ Error storing file: {e}")
            return None
    
    def get_visualization_file(self, file_id: str) -> bytes:
        """Retrieve visualization file from GridFS"""
        try:
            file_obj = self.fs.get(ObjectId(file_id))
            return file_obj.read()
        except Exception as e:
            print(f"❌ Error retrieving file: {e}")
            return None
    
    def log_execution_result(self, algorithm_id: str, user_query: str, success: bool, files_generated: List[str], execution_time: float):
        """Log visualization execution results"""
        result = {
            "algorithm_id": ObjectId(algorithm_id),
            "user_query": user_query,
            "success": success,
            "files_generated": files_generated,
            "execution_time": execution_time,
            "timestamp": datetime.utcnow()
        }
        
        return self.execution_results.insert_one(result)
    
    def get_popular_algorithms(self, limit: int = 10) -> List[Dict]:
        """Get most frequently executed algorithms"""
        pipeline = [
            {"$group": {
                "_id": "$algorithm_id",
                "execution_count": {"$sum": 1},
                "success_rate": {"$avg": {"$cond": ["$success", 1, 0]}}
            }},
            {"$sort": {"execution_count": -1}},
            {"$limit": limit},
            {"$lookup": {
                "from": "algorithms",
                "localField": "_id", 
                "foreignField": "_id",
                "as": "algorithm"
            }}
        ]
        
        return list(self.execution_results.aggregate(pipeline))

# Global MongoDB manager instance
mongo_viz_manager = None

def initialize_mongodb_manager(connection_string: str):
    """Initialize MongoDB manager with connection string"""
    global mongo_viz_manager
    mongo_viz_manager = MongoDBVisualizationManager(connection_string)
    return mongo_viz_manager
