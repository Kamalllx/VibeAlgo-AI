# backend/ai/enhanced_rag_system.py
import os
import json
import numpy as np
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

try:
    from pymongo import MongoClient
    import faiss
    from sentence_transformers import SentenceTransformer
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ RAG dependencies not installed: {e}")
    print("💡 Run: pip install pymongo sentence-transformers faiss-cpu")
    MongoClient = None
    faiss = None
    SentenceTransformer = None
    DEPENDENCIES_AVAILABLE = False

@dataclass
class KnowledgeItem:
    id: str
    title: str
    content: str
    category: str
    subcategory: str
    complexity: str
    tags: List[str]
    source: str
    quality_score: float
    usage_count: int
    success_rate: float
    embedding: Optional[List[float]] = None
    created_at: str = ""
    updated_at: str = ""

@dataclass
class RAGInteraction:
    query: str
    retrieved_docs: List[str]
    generated_response: str
    user_feedback: Optional[int] = None  # 1-5 rating
    response_quality: Optional[float] = None
    timestamp: str = ""

class EnhancedRAGSystem:
    def __init__(self, mongodb_uri: str = None):
        self.mongodb_uri = "mongodb+srv://kamalkarteek1:rvZSeyVHhgOd2fbE@gbh.iliw2.mongodb.net/"
        self.embedding_dimension = 384
        
        print(f"\n🚀 Initializing Enhanced RAG System...")
        print(f"{'='*60}")
        print(f"📊 Target Embedding Dimension: {self.embedding_dimension}")
        print(f"🔗 MongoDB URI: {'SET' if self.mongodb_uri else 'MISSING'}")
        print(f"📦 Dependencies Available: {DEPENDENCIES_AVAILABLE}")
        
        # Initialize embedding model
        self._init_embedding_model()
        
        # Initialize MongoDB
        self._init_mongodb()
        
        # Initialize FAISS
        self._init_faiss()
        
        # Load trusted knowledge base
        self._populate_knowledge_base()
        
        print(f"✅ Enhanced RAG System initialized successfully!")
        print(f"{'='*60}")
    
    def _init_embedding_model(self):
        """Initialize sentence transformer model"""
        if DEPENDENCIES_AVAILABLE and SentenceTransformer:
            try:
                print(f"🧠 Loading SentenceTransformer model...")
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                print(f"✅ Embedding model loaded: all-MiniLM-L6-v2")
            except Exception as e:
                print(f"❌ Failed to load embedding model: {e}")
                self.embedding_model = None
        else:
            print(f"⚠️ SentenceTransformer not available, using fallback")
            self.embedding_model = None
    
    def _init_mongodb(self):
        """Initialize MongoDB connection and collections"""
        if not self.mongodb_uri:
            print(f"⚠️ No MongoDB URI provided, using local storage")
            self.mongo_client = None
            return
        
        try:
            if not DEPENDENCIES_AVAILABLE or not MongoClient:
                print(f"⚠️ MongoDB client not available")
                self.mongo_client = None
                return
                
            print(f"🔗 Connecting to MongoDB...")
            self.mongo_client = MongoClient(self.mongodb_uri)
            
            # Test connection
            self.mongo_client.admin.command('ping')
            
            self.db = self.mongo_client['algorithm_intelligence']
            
            # Collections
            self.knowledge_collection = self.db['knowledge_base']
            self.interactions_collection = self.db['rag_interactions'] 
            self.embeddings_collection = self.db['embeddings']
            
            # Create indexes for better performance
            self.knowledge_collection.create_index([("id", 1)], unique=True)
            self.knowledge_collection.create_index([("category", 1), ("subcategory", 1)])
            self.knowledge_collection.create_index([("tags", 1)])
            self.knowledge_collection.create_index([("quality_score", -1)])
            self.interactions_collection.create_index([("timestamp", -1)])
            
            print(f"✅ MongoDB connected successfully")
            print(f"📚 Collections: knowledge_base, rag_interactions, embeddings")
            
        except Exception as e:
            print(f"❌ MongoDB connection failed: {str(e)}")
            print(f"⚠️ Falling back to local storage mode")
            self.mongo_client = None
    
    def _init_faiss(self):
        """Initialize FAISS index for fast similarity search"""
        try:
            if not DEPENDENCIES_AVAILABLE or not faiss:
                print(f"⚠️ FAISS not available, using fallback search")
                self.faiss_index = None
                return
                
            print(f"🔍 Initializing FAISS index...")
            # Create FAISS index
            self.faiss_index = faiss.IndexFlatL2(self.embedding_dimension)
            self.id_to_doc_mapping = {}
            self.doc_counter = 0
            
            print(f"✅ FAISS index initialized (L2 distance)")
            
        except Exception as e:
            print(f"❌ FAISS initialization failed: {str(e)}")
            self.faiss_index = None
    
    def _populate_knowledge_base(self):
        """Populate knowledge base with comprehensive algorithmic knowledge"""
        print(f"\n📚 Populating Knowledge Base...")
        print(f"{'─'*50}")
        
        # Check if already populated
        if self.mongo_client:
            try:
                existing_count = self.knowledge_collection.count_documents({})
                if existing_count > 0:
                    print(f"📊 Found {existing_count} existing knowledge items")
                    print(f"🔄 Refreshing FAISS index...")
                    self._rebuild_faiss_index()
                    return
            except:
                pass
        
        # Comprehensive algorithmic knowledge from trusted sources
        trusted_knowledge = self._get_trusted_knowledge_base()
        
        # Add knowledge items to database
        added_count = 0
        for item_data in trusted_knowledge:
            knowledge_item = KnowledgeItem(
                id=self._generate_id(item_data["title"]),
                title=item_data["title"],
                content=item_data["content"].strip(),
                category=item_data["category"],
                subcategory=item_data["subcategory"],
                complexity=item_data["complexity"],
                tags=item_data["tags"],
                source=item_data["source"],
                quality_score=1.0,  # Start with high quality for trusted sources
                usage_count=0,
                success_rate=1.0,
                created_at=datetime.now().isoformat()
            )
            
            if self._add_knowledge_item(knowledge_item):
                added_count += 1
        
        print(f"✅ Added {added_count}/{len(trusted_knowledge)} knowledge items")
        print(f"📊 Knowledge base populated successfully!")
    
    def _get_trusted_knowledge_base(self) -> List[Dict[str, Any]]:
        """Get comprehensive trusted algorithmic knowledge"""
        return [
            # Big O Notation
            {
                "title": "Big O Notation Fundamentals",
                "content": """
Big O notation describes the upper bound of algorithm performance as input size grows.

Common complexities:
- O(1): Constant time - performance doesn't change with input size
- O(log n): Logarithmic time - halving search space each step
- O(n): Linear time - performance grows linearly with input size  
- O(n log n): Linearithmic time - efficient sorting algorithms
- O(n²): Quadratic time - nested loops, bubble sort
- O(2^n): Exponential time - recursive Fibonacci

Analysis types:
- Best Case: minimum time needed
- Average Case: expected time for random input
- Worst Case: maximum time needed (most important)

Rules for calculating:
1. Drop constants: O(2n) becomes O(n)
2. Drop non-dominant terms: O(n² + n) becomes O(n²)
3. Consider worst-case scenario
4. Focus on input size approaching infinity
                """,
                "category": "fundamentals",
                "subcategory": "complexity_analysis",
                "complexity": "basic",
                "tags": ["big-o", "time-complexity", "performance", "analysis"],
                "source": "CLRS Introduction to Algorithms"
            },
            
            # Sorting Algorithms
            {
                "title": "Bubble Sort Algorithm Analysis",
                "content": """
Bubble Sort repeatedly compares adjacent elements and swaps them if in wrong order.

Time Complexity:
- Best Case: O(n) - array already sorted with early termination
- Average Case: O(n²) - random order
- Worst Case: O(n²) - reverse sorted array

Space Complexity: O(1) - in-place sorting

Algorithm steps:
1. Compare adjacent elements from start
2. Swap if they are in wrong order
3. Continue to end of array (one element "bubbles" to correct position)
4. Repeat for remaining unsorted portion
5. Stop when no swaps needed in a pass

Optimizations:
- Early termination when no swaps occur
- Reduce comparison range after each pass
- Cocktail shaker sort (bidirectional)

Characteristics:
- Stable: maintains relative order of equal elements
- In-place: only O(1) extra memory
- Simple: easy to understand and implement

Use cases: Educational purposes, very small datasets (< 10 elements)
Better alternatives: Quick Sort O(n log n), Merge Sort O(n log n)
                """,
                "category": "sorting",
                "subcategory": "comparison_sorts",
                "complexity": "basic",
                "tags": ["bubble-sort", "O(n²)", "in-place", "stable"],
                "source": "Algorithms 4th Edition by Sedgewick"
            },
            
            {
                "title": "Quick Sort Algorithm Analysis",
                "content": """
Quick Sort uses divide-and-conquer with pivot partitioning.

Time Complexity:
- Best Case: O(n log n) - pivot divides array evenly
- Average Case: O(n log n) - random pivot selection
- Worst Case: O(n²) - already sorted with poor pivot choice

Space Complexity: O(log n) - recursion stack depth

Algorithm steps:
1. Choose pivot element from array
2. Partition: rearrange so elements < pivot go left, > pivot go right
3. Recursively sort left and right subarrays
4. Combine results (no work needed due to in-place partitioning)

Pivot Selection Strategies:
- First/Last element: simple but worst case on sorted data
- Random element: good average performance, avoids worst case
- Median-of-three: takes median of first, middle, last elements
- True median: guarantees O(n log n) but expensive to compute

Partitioning Methods:
- Lomuto partition: simpler, more swaps
- Hoare partition: fewer swaps, more efficient

Optimizations:
- Three-way partitioning for many duplicate elements
- Hybrid with insertion sort for small subarrays (< 10 elements)
- Iterative implementation to avoid stack overflow
- Tail recursion optimization

Characteristics:
- Unstable: may change relative order of equal elements
- In-place: O(1) extra space if implemented iteratively
- Cache-friendly: good locality of reference

Use cases: General purpose sorting, when average performance matters
                """,
                "category": "sorting",
                "subcategory": "comparison_sorts",
                "complexity": "intermediate",
                "tags": ["quick-sort", "divide-conquer", "O(n log n)", "pivot", "partition"],
                "source": "CLRS Introduction to Algorithms"
            },
            
            {
                "title": "Merge Sort Algorithm Analysis",
                "content": """
Merge Sort is a stable divide-and-conquer algorithm with guaranteed performance.

Time Complexity: O(n log n) in all cases (best, average, worst)
Space Complexity: O(n) - requires auxiliary array for merging

Algorithm steps:
1. Divide array into two halves recursively until single elements
2. Merge sorted halves back together in correct order
3. Continue merging until complete sorted array

Merging Process:
1. Compare elements at front of two sorted arrays
2. Take smaller element and advance its pointer
3. Repeat until one array exhausted
4. Copy remaining elements from other array

Key Properties:
- Stable: maintains relative order of equal elements
- Predictable: always O(n log n) regardless of input
- External: can sort data larger than available memory
- Parallelizable: subarrays can be sorted independently

Variations:
- Bottom-up merge sort: iterative approach, no recursion
- Natural merge sort: takes advantage of existing sorted runs
- In-place merge sort: complex but O(1) space
- Multi-way merge sort: merge k sorted arrays simultaneously

Space Optimizations:
- Ping-pong merging: alternate between two auxiliary arrays
- In-place merging: complex but reduces space to O(1)

Use Cases:
- When stability is required
- External sorting (data doesn't fit in memory)
- When guaranteed O(n log n) performance needed
- Parallel processing environments
- LinkedList sorting (no random access penalty)
                """,
                "category": "sorting",
                "subcategory": "comparison_sorts",
                "complexity": "intermediate",
                "tags": ["merge-sort", "stable", "divide-conquer", "O(n log n)", "guaranteed"],
                "source": "Algorithms 4th Edition by Sedgewick"
            },
            
            # Search Algorithms
            {
                "title": "Binary Search Algorithm Analysis",
                "content": """
Binary Search efficiently finds target in sorted array by halving search space.

Time Complexity: O(log n)
Space Complexity: O(1) iterative, O(log n) recursive

Prerequisites:
- Array must be sorted
- Random access to elements (arrays, not linked lists)

Algorithm Steps:
1. Set left = 0, right = array.length - 1
2. While left <= right:
   a. Calculate mid = left + (right - left) / 2
   b. If array[mid] == target: return mid
   c. If array[mid] < target: left = mid + 1
   d. If array[mid] > target: right = mid - 1
3. Return -1 (not found)

Implementation Tips:
- Use left + (right - left) / 2 to avoid integer overflow
- Be careful with boundary conditions
- Consider whether to find first/last occurrence

Variations:
- Lower bound: find first position where element could be inserted
- Upper bound: find last position where element could be inserted
- Search for range: find first and last occurrence
- Peak finding: find local maximum in array

Applications:
- Dictionary lookups
- Database indexing
- Finding insertion point for sorted insertion
- Square root calculation with precision
- Search in rotated sorted array

Error-prone Areas:
- Off-by-one errors in boundary conditions
- Infinite loops with incorrect mid calculation
- Integer overflow with (left + right) / 2
- Handling duplicate elements incorrectly

Related Algorithms:
- Exponential search: find range first, then binary search
- Interpolation search: estimate position based on value
- Ternary search: divide into three parts instead of two
                """,
                "category": "searching",
                "subcategory": "sorted_search",
                "complexity": "basic",
                "tags": ["binary-search", "O(log n)", "sorted", "divide-conquer"],
                "source": "Programming Pearls by Jon Bentley"
            },
            
            # Data Structures
            {
                "title": "Array Data Structure Analysis",
                "content": """
Arrays store elements in contiguous memory locations with constant-time access.

Time Complexities:
- Access by index: O(1)
- Search (unsorted): O(n)
- Search (sorted): O(log n) with binary search
- Insertion at end: O(1) amortized (dynamic arrays)
- Insertion at position: O(n) - requires shifting
- Deletion from end: O(1)
- Deletion from position: O(n) - requires shifting

Space Complexity: O(n)

Memory Characteristics:
- Contiguous allocation: elements stored sequentially
- Cache-friendly: spatial locality benefits
- Fixed size (static) or dynamic resizing
- Direct addressing: address = base + index * element_size

Types:
- Static arrays: fixed size at compile time
- Dynamic arrays: resizable (ArrayList, vector, Python list)
- Multi-dimensional: arrays of arrays

Dynamic Array Resizing:
- Growth factor: typically 1.5x or 2x when full
- Amortized analysis: average O(1) insertion
- Memory overhead: unused capacity for future growth

Advantages:
- Fast random access by index
- Memory efficient (no pointer overhead)
- Cache performance due to spatial locality
- Simple iteration patterns

Disadvantages:
- Fixed size (static arrays)
- Expensive insertion/deletion in middle
- Memory waste in dynamic arrays
- No efficient insertion at arbitrary positions

Common Operations:
- Finding min/max: O(n)
- Sorting: O(n log n) optimal comparison-based
- Reversing: O(n)
- Rotation: O(n)
- Prefix sums: O(n) preprocessing, O(1) range queries

Use Cases:
- Mathematical computations and matrices
- Implementing other data structures
- When fast random access needed
- Cache-sensitive applications
                """,
                "category": "data_structures",
                "subcategory": "linear_structures",
                "complexity": "basic",
                "tags": ["array", "O(1) access", "contiguous", "cache-friendly"],
                "source": "Data Structures and Algorithms in Python"
            },
            
            {
                "title": "Hash Table Data Structure Analysis",
                "content": """
Hash Tables provide average O(1) operations using hash functions for key-to-index mapping.

Average Time Complexities:
- Search: O(1)
- Insertion: O(1)
- Deletion: O(1)

Worst Case: O(n) when all keys hash to same bucket

Space Complexity: O(n)

Core Components:
1. Hash Function: maps keys to array indices
2. Collision Resolution: handles multiple keys mapping to same index
3. Dynamic Resizing: maintains performance as size grows

Hash Function Properties:
- Deterministic: same key always produces same hash
- Uniform distribution: spreads keys evenly across buckets
- Fast computation: O(1) hash calculation
- Avalanche effect: small input changes cause large hash changes

Collision Resolution Strategies:

1. Chaining (Separate Chaining):
   - Store colliding elements in linked lists/arrays at each bucket
   - Simple to implement and delete
   - Performance degrades gracefully
   - Extra memory for pointers

2. Open Addressing:
   - Find alternative slots when collision occurs
   - Linear probing: check next slot sequentially
   - Quadratic probing: check slots at quadratic intervals
   - Double hashing: use second hash function for step size

Load Factor Management:
- Load factor α = n/m (elements/buckets)
- Keep α < 0.75 for good performance
- Resize when load factor exceeds threshold
- Rehash all elements to new table size

Resizing Strategy:
- Double table size when load factor > 0.75
- Halve table size when load factor < 0.25
- Rehash all existing elements
- Amortized O(1) operations despite occasional O(n) resize

Applications:
- Database indexing and caching
- Compiler symbol tables
- Set and dictionary implementations
- Frequency counting and analytics
- Distributed systems (consistent hashing)

Performance Considerations:
- Choose good hash function for data type
- Monitor and maintain appropriate load factor
- Consider cache performance in hash function design
- Handle hash collisions gracefully
                """,
                "category": "data_structures",
                "subcategory": "hash_based",
                "complexity": "intermediate",
                "tags": ["hash-table", "O(1) average", "collision-resolution", "load-factor"],
                "source": "Introduction to Algorithms (CLRS)"
            },
            
            # Graph Algorithms
            {
                "title": "Depth-First Search (DFS) Analysis",
                "content": """
DFS explores graph by going as deep as possible before backtracking.

Time Complexity: O(V + E) where V = vertices, E = edges
Space Complexity: O(V) for recursion stack or explicit stack

Algorithm (Recursive):
def dfs(graph, start, visited=set()):
visited.add(start)
process(start)
for neighbor in graph[start]:
if neighbor not in visited:
dfs(graph, neighbor, visited)

Algorithm (Iterative):
def dfs_iterative(graph, start):
visited = set()
stack = [start]
while stack:
vertex = stack.pop()
if vertex not in visited:
visited.add(vertex)
process(vertex)
for neighbor in graph[vertex]:
if neighbor not in visited:
stack.append(neighbor)

Applications:
- Topological sorting of DAGs
- Detecting cycles in directed graphs
- Finding strongly connected components
- Maze solving and pathfinding
- Tree/forest detection in undirected graphs
- Solving puzzles with backtracking

DFS Traversal Orders:
- Preorder: process vertex before its children
- Postorder: process vertex after its children
- Both useful for different applications

Variants:
- DFS on trees: no cycle detection needed
- DFS with timestamps: discovery and finish times
- Iterative deepening: DFS with depth limits
- Bidirectional DFS: search from both ends

Implementation Considerations:
- Track visited vertices to avoid infinite loops
- Handle disconnected components
- Choose starting vertex strategically
- Stack overflow prevention in deep graphs

Time Complexity Analysis:
- Each vertex visited exactly once: O(V)
- Each edge examined exactly twice: O(E)
- Total: O(V + E)

Use Cases:
- When you need to explore all reachable vertices
- Detecting cycles or checking connectivity
- Topological ordering
- Finding cut vertices/bridges
                """,
                "category": "algorithms",
                "subcategory": "graph_algorithms",
                "complexity": "intermediate",
                "tags": ["dfs", "graph-traversal", "O(V+E)", "backtracking"],
                "source": "Algorithms 4th Edition by Sedgewick"
            },
            
            {
                "title": "Breadth-First Search (BFS) Analysis",
                "content": """
BFS explores graph level by level using a queue data structure.

Time Complexity: O(V + E) where V = vertices, E = edges
Space Complexity: O(V) for queue storage

Algorithm:
from collections import deque

def bfs(graph, start):
visited = set()
queue = deque([start])
visited.add(start)
while queue:
    vertex = queue.popleft()
    process(vertex)
    
    for neighbor in graph[vertex]:
        if neighbor not in visited:
            visited.add(neighbor)
            queue.append(neighbor)

Key Properties:
- Finds shortest path in unweighted graphs
- Explores vertices in order of distance from start
- Guarantees minimum number of edges in path
- Level-by-level exploration pattern

Applications:
- Shortest path in unweighted graphs
- Level-order tree traversal
- Finding connected components
- Bipartite graph detection
- Web crawling with depth limits
- Social network analysis (degrees of separation)

Shortest Path with BFS:
def bfs_shortest_path(graph, start, target):
queue = deque([(start, [start])])
visited = {start}
while queue:
    vertex, path = queue.popleft()
    if vertex == target:
        return path
        
    for neighbor in graph[vertex]:
        if neighbor not in visited:
            visited.add(neighbor)
            queue.append((neighbor, path + [neighbor]))

return None  # No path found

Distance Tracking:
- Maintain distance array: dist[v] = shortest distance to v
- Parent tracking: parent[v] = previous vertex in shortest path
- Path reconstruction by following parent pointers

BFS vs DFS Comparison:
- BFS: shortest path, level-by-level, queue-based, O(V) space
- DFS: deeper exploration, stack-based, O(V) space, path finding

Variants:
- Multi-source BFS: start from multiple vertices simultaneously
- Bidirectional BFS: search from both start and target
- 0-1 BFS: for graphs with edge weights 0 or 1

Time Complexity Analysis:
- Each vertex enqueued and dequeued once: O(V)
- Each edge examined exactly once: O(E)
- Total: O(V + E)

Use Cases:
- Finding shortest unweighted paths
- Level-order processing
- Minimum spanning tree algorithms
- Network broadcast protocols
                """,
                "category": "algorithms",
                "subcategory": "graph_algorithms",
                "complexity": "intermediate",
                "tags": ["bfs", "shortest-path", "level-order", "queue", "O(V+E)"],
                "source": "Introduction to Algorithms (CLRS)"
            }
        ]
    
    def _generate_id(self, text: str) -> str:
        """Generate unique ID for knowledge item"""
        return hashlib.md5(text.encode()).hexdigest()[:12]
    
    def _add_knowledge_item(self, item: KnowledgeItem) -> bool:
        """Add knowledge item to database and FAISS index"""
        try:
            # Generate embedding if model available
            if self.embedding_model:
                embedding = self.embedding_model.encode(
                    f"{item.title} {item.content}"
                ).tolist()
                item.embedding = embedding
            
            # Add to MongoDB
            if self.mongo_client:
                try:
                    self.knowledge_collection.insert_one(asdict(item))
                except Exception as e:
                    if "duplicate key" not in str(e).lower():
                        print(f"❌ MongoDB insert error: {e}")
                        return False
            
            # Add to FAISS index
            if self.faiss_index and item.embedding:
                embedding_array = np.array([item.embedding]).astype('float32')
                self.faiss_index.add(embedding_array)
                self.id_to_doc_mapping[self.doc_counter] = item.id
                self.doc_counter += 1
            
            return True
            
        except Exception as e:
            print(f"❌ Error adding knowledge item: {str(e)}")
            return False
    
    def _rebuild_faiss_index(self):
        """Rebuild FAISS index from MongoDB data"""
        if not self.mongo_client or not self.faiss_index or not self.embedding_model:
            return
        
        try:
            print(f"🔄 Rebuilding FAISS index...")
            
            # Clear existing index
            self.faiss_index.reset()
            self.id_to_doc_mapping.clear()
            self.doc_counter = 0
            
            # Load all documents and rebuild embeddings if needed
            for doc in self.knowledge_collection.find({}):
                if 'embedding' not in doc or not doc['embedding']:
                    # Generate missing embedding
                    embedding = self.embedding_model.encode(
                        f"{doc['title']} {doc['content']}"
                    ).tolist()
                    
                    # Update document with embedding
                    self.knowledge_collection.update_one(
                        {"id": doc["id"]},
                        {"$set": {"embedding": embedding}}
                    )
                else:
                    embedding = doc['embedding']
                
                # Add to FAISS index
                embedding_array = np.array([embedding]).astype('float32')
                self.faiss_index.add(embedding_array)
                self.id_to_doc_mapping[self.doc_counter] = doc['id']
                self.doc_counter += 1
            
            print(f"✅ FAISS index rebuilt with {self.doc_counter} documents")
            
        except Exception as e:
            print(f"❌ Failed to rebuild FAISS index: {e}")
    
    def retrieve_relevant_context(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve most relevant knowledge items for query"""
        print(f"\n🔍 RAG KNOWLEDGE RETRIEVAL")
        print(f"{'─'*50}")
        print(f"📝 Query: '{query}'")
        print(f"🎯 Retrieving top {top_k} relevant documents...")
        
        try:
            # Use FAISS if available
            if self.faiss_index and self.embedding_model and self.faiss_index.ntotal > 0:
                return self._faiss_search(query, top_k)
            else:
                return self._fallback_search(query, top_k)
                
        except Exception as e:
            print(f"❌ RAG retrieval error: {str(e)}")
            return self._fallback_search(query, top_k)
    
    def _faiss_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Search using FAISS vector similarity"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).astype('float32')
            
            # Search FAISS index
            distances, indices = self.faiss_index.search(query_embedding, min(top_k, self.faiss_index.ntotal))
            
            relevant_docs = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx == -1:  # No more results
                    break
                
                doc_id = self.id_to_doc_mapping.get(idx)
                if doc_id and self.mongo_client:
                    doc = self.knowledge_collection.find_one({"id": doc_id})
                    if doc:
                        relevance_score = max(0, 1.0 - (distance / 2.0))  # Convert distance to relevance
                        relevant_docs.append({
                            'id': doc['id'],
                            'title': doc['title'],
                            'content': doc['content'][:500] + "..." if len(doc['content']) > 500 else doc['content'],
                            'category': doc['category'],
                            'subcategory': doc['subcategory'],
                            'tags': doc['tags'],
                            'source': doc['source'],
                            'relevance_score': relevance_score,
                            'quality_score': doc['quality_score'],
                            'full_content': doc['content']
                        })
            
            # Sort by relevance score
            relevant_docs.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            print(f"📖 Retrieved {len(relevant_docs)} relevant documents via FAISS:")
            for i, doc in enumerate(relevant_docs, 1):
                print(f"   {i}. {doc['title']} (relevance: {doc['relevance_score']:.3f})")
                print(f"      Category: {doc['category']}/{doc['subcategory']}")
                print(f"      Source: {doc['source']}")
            
            return relevant_docs
            
        except Exception as e:
            print(f"❌ FAISS search failed: {e}")
            return self._fallback_search(query, top_k)
    
    def _fallback_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Fallback search using text matching"""
        print(f"🔄 Using fallback text-based search...")
        
        query_lower = query.lower()
        relevant_docs = []
        
        # Define fallback knowledge base
        fallback_docs = [
            {
                'id': 'fallback_001',
                'title': 'Big O Time Complexity Analysis',
                'content': 'Big O notation describes algorithm performance. O(1) constant, O(log n) logarithmic, O(n) linear, O(n²) quadratic...',
                'category': 'fundamentals',
                'subcategory': 'complexity_analysis',
                'tags': ['big-o', 'time-complexity'],
                'source': 'CLRS Algorithms',
                'quality_score': 1.0,
                'full_content': 'Big O notation describes the upper bound of algorithm performance as input size grows.'
            },
            {
                'id': 'fallback_002', 
                'title': 'Bubble Sort Algorithm',
                'content': 'Bubble sort has O(n²) time complexity with nested loops comparing adjacent elements...',
                'category': 'sorting',
                'subcategory': 'comparison_sorts',
                'tags': ['bubble-sort', 'O(n²)'],
                'source': 'Algorithms Textbook',
                'quality_score': 1.0,
                'full_content': 'Bubble Sort repeatedly steps through the list, compares adjacent elements and swaps them if wrong order.'
            },
            {
                'id': 'fallback_003',
                'title': 'Binary Search Algorithm', 
                'content': 'Binary search achieves O(log n) time complexity by halving the search space...',
                'category': 'searching',
                'subcategory': 'sorted_search',
                'tags': ['binary-search', 'O(log n)'],
                'source': 'Programming Pearls',
                'quality_score': 1.0,
                'full_content': 'Binary Search efficiently finds target in sorted array by repeatedly dividing search space in half.'
            }
        ]
        
        # Search in MongoDB if available
        if self.mongo_client:
            try:
                search_results = self.knowledge_collection.find({
                    "$or": [
                        {"title": {"$regex": query_lower, "$options": "i"}},
                        {"content": {"$regex": query_lower, "$options": "i"}},
                        {"tags": {"$in": [query_lower]}}
                    ]
                }).limit(top_k)
                
                for doc in search_results:
                    relevance_score = self._calculate_text_relevance(query_lower, doc)
                    relevant_docs.append({
                        'id': doc['id'],
                        'title': doc['title'],
                        'content': doc['content'][:500] + "..." if len(doc['content']) > 500 else doc['content'],
                        'category': doc['category'],
                        'subcategory': doc['subcategory'],
                        'tags': doc['tags'],
                        'source': doc['source'],
                        'relevance_score': relevance_score,
                        'quality_score': doc['quality_score'],
                        'full_content': doc['content']
                    })
                
            except Exception as e:
                print(f"❌ MongoDB fallback search failed: {str(e)}")
        
        # Use fallback docs if no MongoDB results
        if not relevant_docs:
            for doc in fallback_docs:
                relevance_score = self._calculate_text_relevance(query_lower, doc)
                if relevance_score > 0:
                    doc['relevance_score'] = relevance_score
                    relevant_docs.append(doc)
        
        # Sort by relevance
        relevant_docs.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        print(f"📖 Retrieved {len(relevant_docs)} documents via fallback search:")
        for i, doc in enumerate(relevant_docs[:top_k], 1):
            print(f"   {i}. {doc['title']} (relevance: {doc.get('relevance_score', 0):.3f})")
        
        return relevant_docs[:top_k]
    
    def _calculate_text_relevance(self, query: str, doc: Dict) -> float:
        """Calculate relevance score based on text matching"""
        score = 0.0
        
        # Title match (highest weight)
        if query in doc['title'].lower():
            score += 0.5
        
        # Content match
        if query in doc['content'].lower():
            score += 0.3
        
        # Tag match
        for tag in doc.get('tags', []):
            if query in tag.lower():
                score += 0.2
        
        # Category match
        if query in doc.get('category', '').lower():
            score += 0.1
        
        return min(score, 1.0)
    
    def learn_from_interaction(self, query: str, retrieved_docs: List[str], 
                             response: str, quality_score: float):
        """Learn from user interactions using reinforcement learning"""
        print(f"\n🧠 REINFORCEMENT LEARNING UPDATE")
        print(f"{'─'*50}")
        print(f"📝 Query: {query[:50]}...")
        print(f"📊 Response Quality Score: {quality_score:.2f}/5.0")
        
        try:
            # Store interaction
            interaction = RAGInteraction(
                query=query,
                retrieved_docs=retrieved_docs,
                generated_response=response,
                response_quality=quality_score,
                timestamp=datetime.now().isoformat()
            )
            
            if self.mongo_client:
                self.interactions_collection.insert_one(asdict(interaction))
            
            # Update quality scores of retrieved documents
            for doc_id in retrieved_docs:
                self._update_document_quality(doc_id, quality_score)
            
            # Extract new knowledge if high quality response
            if quality_score >= 4.0:
                self._extract_new_knowledge(query, response)
            
            print(f"✅ Learning update completed")
            
        except Exception as e:
            print(f"❌ Learning update failed: {e}")
    
    def _update_document_quality(self, doc_id: str, quality_score: float):
        """Update document quality using exponential moving average"""
        if not self.mongo_client:
            return
        
        try:
            doc = self.knowledge_collection.find_one({"id": doc_id})
            if doc:
                # Update metrics using exponential moving average
                new_usage_count = doc['usage_count'] + 1
                alpha = 0.1  # Learning rate
                new_success_rate = (1 - alpha) * doc['success_rate'] + alpha * (quality_score / 5.0)
                new_quality_score = (1 - alpha) * doc['quality_score'] + alpha * quality_score
                
                self.knowledge_collection.update_one(
                    {"id": doc_id},
                    {
                        "$set": {
                            "usage_count": new_usage_count,
                            "success_rate": new_success_rate,
                            "quality_score": new_quality_score,
                            "updated_at": datetime.now().isoformat()
                        }
                    }
                )
                
                print(f"📈 Updated {doc_id}: quality={new_quality_score:.3f}, usage={new_usage_count}")
                
        except Exception as e:
            print(f"❌ Error updating document quality: {str(e)}")
    
    def _extract_new_knowledge(self, query: str, response: str):
        """Extract new knowledge from high-quality interactions"""
        print(f"🎓 Extracting potential new knowledge...")
        
        # This would use the Groq client to analyze if the response contains
        # new algorithmic insights worth storing. For now, we'll skip this
        # to avoid circular imports and focus on the core RAG functionality.
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive RAG system statistics"""
        stats = {
            "knowledge_base_size": 0,
            "total_interactions": 0,
            "average_quality": 0.0,
            "categories": {},
            "faiss_index_size": 0,
            "system_status": {
                "mongodb_connected": bool(self.mongo_client),
                "faiss_available": bool(self.faiss_index),
                "embedding_model_loaded": bool(self.embedding_model),
                "dependencies_available": DEPENDENCIES_AVAILABLE
            }
        }
        
        if self.mongo_client:
            try:
                stats["knowledge_base_size"] = self.knowledge_collection.count_documents({})
                stats["total_interactions"] = self.interactions_collection.count_documents({})
                
                # Calculate average quality
                pipeline = [
                    {"$group": {"_id": None, "avg_quality": {"$avg": "$quality_score"}}}
                ]
                result = list(self.knowledge_collection.aggregate(pipeline))
                if result:
                    stats["average_quality"] = result[0]["avg_quality"]
                
                # Get category distribution
                pipeline = [
                    {"$group": {"_id": "$category", "count": {"$sum": 1}}}
                ]
                for item in self.knowledge_collection.aggregate(pipeline):
                    stats["categories"][item["_id"]] = item["count"]
                
            except Exception as e:
                print(f"❌ Error getting MongoDB stats: {str(e)}")
        
        if self.faiss_index:
            stats["faiss_index_size"] = self.faiss_index.ntotal
        
        return stats

# Global enhanced RAG instance - will be created when imported
enhanced_rag = None

def initialize_enhanced_rag():
    """Initialize the enhanced RAG system"""
    global enhanced_rag
    if enhanced_rag is None:
        enhanced_rag = EnhancedRAGSystem()
    return enhanced_rag

# Initialize on import
enhanced_rag = initialize_enhanced_rag()
