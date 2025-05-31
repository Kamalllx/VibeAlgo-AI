# backend/ai/enhanced_rag_system.py
import os
import json
import numpy as np
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pymongo import MongoClient
import faiss
from sentence_transformers import SentenceTransformer
from ai.groq_client import groq_client

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
        self.mongodb_uri = mongodb_uri or os.getenv('MONGODB_URI')
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dimension = 384
        
        print(f"🔗 Initializing Enhanced RAG System...")
        print(f"📊 Embedding Model: all-MiniLM-L6-v2 (dim: {self.embedding_dimension})")
        
        # Initialize MongoDB
        self._init_mongodb()
        
        # Initialize FAISS
        self._init_faiss()
        
        # Load trusted knowledge base
        self._populate_knowledge_base()
        
        print(f"✅ Enhanced RAG System initialized successfully!")
    
    def _init_mongodb(self):
        """Initialize MongoDB connection and collections"""
        try:
            self.mongo_client = MongoClient(self.mongodb_uri)
            self.db = self.mongo_client['algorithm_intelligence']
            
            # Collections
            self.knowledge_collection = self.db['knowledge_base']
            self.interactions_collection = self.db['rag_interactions'] 
            self.embeddings_collection = self.db['embeddings']
            
            # Create indexes for better performance
            self.knowledge_collection.create_index([("category", 1), ("subcategory", 1)])
            self.knowledge_collection.create_index([("tags", 1)])
            self.knowledge_collection.create_index([("quality_score", -1)])
            
            print(f"✅ MongoDB connected successfully")
            print(f"📚 Collections: knowledge_base, rag_interactions, embeddings")
            
        except Exception as e:
            print(f"❌ MongoDB connection failed: {str(e)}")
            print(f"⚠️ Falling back to local storage mode")
            self.mongo_client = None
    
    def _init_faiss(self):
        """Initialize FAISS index for fast similarity search"""
        try:
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
        print(f"\n📚 Populating knowledge base with trusted algorithmic resources...")
        
        # Comprehensive algorithmic knowledge from trusted sources
        trusted_knowledge = [
            # Time Complexity Fundamentals
            {
                "title": "Big O Notation Fundamentals",
                "content": """
Big O notation describes the upper bound of algorithm performance as input size grows.
- O(1): Constant time - performance doesn't change with input size
- O(log n): Logarithmic time - performance grows logarithmically 
- O(n): Linear time - performance grows linearly with input size
- O(n log n): Linearithmic time - common in efficient sorting algorithms
- O(n²): Quadratic time - performance grows quadratically, often from nested loops
- O(2^n): Exponential time - performance doubles with each additional input

Best Case < Average Case < Worst Case complexity analysis is crucial for understanding algorithm behavior.
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
Bubble Sort repeatedly steps through the list, compares adjacent elements and swaps them if wrong order.
Time Complexity: O(n²) in worst and average case, O(n) best case (already sorted)
Space Complexity: O(1) - in-place sorting algorithm

Algorithm steps:
1. Compare adjacent elements
2. Swap if they are in wrong order  
3. Repeat until no swaps needed

Optimizations:
- Early termination when no swaps occur
- Reduce comparison range in each pass

Use cases: Educational purposes, very small datasets
Better alternatives: Quick Sort, Merge Sort, Heap Sort
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
Quick Sort is a divide-and-conquer algorithm that picks a pivot and partitions array around it.
Time Complexity: O(n log n) average case, O(n²) worst case
Space Complexity: O(log n) average case due to recursion stack

Algorithm steps:
1. Choose a pivot element
2. Partition array so elements < pivot go left, elements > pivot go right
3. Recursively sort subarrays

Pivot selection strategies:
- First/last element (simple but can lead to worst case)
- Random element (good average performance) 
- Median-of-three (better worst case avoidance)

Optimizations:
- Three-way partitioning for duplicate elements
- Hybrid with insertion sort for small subarrays
- Iterative implementation to avoid stack overflow

Use cases: General purpose sorting, when average case performance matters
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
Merge Sort is a stable divide-and-conquer sorting algorithm with guaranteed O(n log n) performance.
Time Complexity: O(n log n) in all cases (best, average, worst)
Space Complexity: O(n) - requires additional array for merging

Algorithm steps:
1. Divide array into two halves
2. Recursively sort both halves
3. Merge the sorted halves

Key properties:
- Stable: maintains relative order of equal elements
- Predictable: always O(n log n) regardless of input
- External: can sort data larger than memory

Variations:
- Bottom-up merge sort (iterative)
- Natural merge sort (takes advantage of existing runs)
- In-place merge sort (complex but O(1) space)

Use cases: When stability is required, external sorting, guaranteed performance needed
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
Binary Search efficiently finds target in sorted array by repeatedly dividing search space in half.
Time Complexity: O(log n) 
Space Complexity: O(1) iterative, O(log n) recursive

Algorithm steps:
1. Compare target with middle element
2. If equal, return index
3. If target < middle, search left half
4. If target > middle, search right half
5. Repeat until found or search space empty

Preconditions:
- Array must be sorted
- Random access to elements required

Variations:
- Lower bound: find first occurrence
- Upper bound: find last occurrence  
- Approximate search: find closest element

Applications:
- Dictionary lookups
- Database indexing
- Finding insertion point
- Peak finding problems

Implementation tips:
- Use (left + right) // 2 or left + (right - left) // 2 to avoid overflow
- Be careful with boundary conditions
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
Arrays provide constant-time access to elements by index with contiguous memory layout.
Access: O(1) - direct indexing
Search: O(n) - linear scan required for unsorted
Insertion: O(n) - may require shifting elements
Deletion: O(n) - may require shifting elements

Memory characteristics:
- Contiguous memory allocation
- Cache-friendly due to spatial locality
- Fixed size in many languages (dynamic in Python)

Advantages:
- Fast random access by index
- Memory efficient (no extra pointers)
- Cache performance benefits

Disadvantages:
- Fixed size (in static arrays)
- Expensive insertion/deletion in middle
- Memory waste if not fully utilized

Common operations analysis:
- Finding minimum/maximum: O(n)
- Sorting: O(n log n) with optimal algorithms
- Reversing: O(n)
- Rotation: O(n)

Use cases: When fast access by index needed, mathematical computations, implementing other data structures
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
Hash Tables provide average O(1) access time using hash functions to map keys to array indices.
Average Case: O(1) for search, insertion, deletion
Worst Case: O(n) when all keys hash to same bucket
Space Complexity: O(n)

Core components:
- Hash function: maps keys to array indices
- Collision resolution: handles multiple keys mapping to same index
- Load factor: ratio of filled slots to total slots

Hash function properties:
- Deterministic: same key always produces same hash
- Uniform distribution: spreads keys evenly
- Fast computation: O(1) hash calculation

Collision resolution strategies:
1. Chaining: store colliding elements in linked lists
2. Open addressing: find alternative slots (linear/quadratic probing, double hashing)

Load factor management:
- Keep load factor < 0.75 for good performance
- Resize and rehash when load factor exceeds threshold
- Dynamic resizing maintains average O(1) performance

Applications:
- Database indexing
- Caching systems
- Symbol tables in compilers
- Set operations
- Frequency counting

Implementation considerations:
- Choose good hash function for key type
- Handle hash collisions properly
- Monitor and manage load factor
                """,
                "category": "data_structures", 
                "subcategory": "hash_based",
                "complexity": "intermediate",
                "tags": ["hash-table", "O(1) average", "collision-resolution", "load-factor"],
                "source": "Introduction to Algorithms (CLRS)"
            },
            
            # Dynamic Programming
            {
                "title": "Dynamic Programming Fundamentals",
                "content": """
Dynamic Programming solves complex problems by breaking them down into simpler overlapping subproblems.
Key principles:
1. Optimal substructure: optimal solution contains optimal solutions to subproblems
2. Overlapping subproblems: same subproblems solved multiple times

Two main approaches:
1. Memoization (Top-down): solve recursively with caching
2. Tabulation (Bottom-up): solve iteratively building table

Time Complexity: Often reduces exponential to polynomial time
Space Complexity: Usually O(n) or O(n²) for storing subproblem solutions

Classic problems:
- Fibonacci sequence: O(n) instead of O(2^n)
- Longest Common Subsequence: O(mn)
- Knapsack problem: O(nW)
- Edit distance: O(mn)
- Coin change: O(amount × coins)

Problem identification patterns:
- Can be broken into subproblems
- Subproblems overlap
- Optimal substructure exists
- Choices lead to subproblems

Implementation steps:
1. Define state/subproblem
2. Find recurrence relation
3. Identify base cases
4. Determine evaluation order
5. Optimize space if possible

Space optimizations:
- Rolling arrays for 1D DP
- Two rows for 2D DP when only previous row needed
                """,
                "category": "algorithms",
                "subcategory": "dynamic_programming", 
                "complexity": "advanced",
                "tags": ["dynamic-programming", "memoization", "tabulation", "optimization"],
                "source": "Dynamic Programming for Coding Interviews"
            },
            
            # Graph Algorithms
            {
                "title": "Depth-First Search (DFS) Analysis",
                "content": """
DFS explores graph by going as deep as possible before backtracking.
Time Complexity: O(V + E) where V = vertices, E = edges
Space Complexity: O(V) for recursion stack or explicit stack

Algorithm approaches:
1. Recursive: natural but limited by stack size
2. Iterative with stack: unlimited depth, explicit control

Applications:
- Topological sorting
- Detecting cycles in directed graphs
- Finding strongly connected components
- Maze solving
- Tree/forest detection

DFS traversal orders:
- Preorder: process vertex before children
- Postorder: process vertex after children

Implementation considerations:
- Track visited vertices to avoid infinite loops
- Choose starting vertex strategically
- Handle disconnected components

Variants:
- DFS on trees: no cycle detection needed
- DFS with timestamps: useful for interval trees
- Iterative deepening: DFS with depth limits

Code structure:
```python
def dfs(graph, start, visited=set()):
visited.add(start)
process(start) # Process current vertex
for neighbor in graph[start]:
if neighbor not in visited:
dfs(graph, neighbor, visited)

Use cases: When you need to explore all paths, detect cycles, or find connected components
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

Algorithm steps:
1. Add starting vertex to queue
2. Mark starting vertex as visited
3. While queue not empty:
   - Dequeue vertex
   - Process vertex
   - Add unvisited neighbors to queue

Key properties:
- Finds shortest path in unweighted graphs
- Explores vertices in order of distance from start
- Guarantees minimum number of edges in path

Applications:
- Shortest path in unweighted graphs
- Level-order tree traversal
- Finding connected components
- Bipartite graph detection
- Web crawling

BFS vs DFS comparison:
- BFS: shortest path, level-by-level, uses queue
- DFS: deeper exploration, uses stack/recursion

Implementation with queue:
```python
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
```

Distance tracking:
- Maintain distance array to track shortest paths
- Parent tracking for path reconstruction

Use cases: Shortest path problems, level-based processing, minimum spanning trees
                """,
                "category": "algorithms",
                "subcategory": "graph_algorithms", 
                "complexity": "intermediate",
                "tags": ["bfs", "shortest-path", "level-order", "queue", "O(V+E)"],
                "source": "Introduction to Algorithms (CLRS)"
            }
        ]
        
        # Add knowledge items to database
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
            
            self._add_knowledge_item(knowledge_item)
        
        print(f"✅ Added {len(trusted_knowledge)} trusted knowledge items")
        print(f"📊 Knowledge base populated successfully!")
    
    def _generate_id(self, text: str) -> str:
        """Generate unique ID for knowledge item"""
        return hashlib.md5(text.encode()).hexdigest()[:12]
    
    def _add_knowledge_item(self, item: KnowledgeItem):
        """Add knowledge item to database and FAISS index"""
        try:
            # Generate embedding
            embedding = self.embedding_model.encode(
                f"{item.title} {item.content}"
            ).tolist()
            item.embedding = embedding
            
            # Add to MongoDB
            if self.mongo_client:
                self.knowledge_collection.insert_one(asdict(item))
            
            # Add to FAISS index
            if self.faiss_index:
                embedding_array = np.array([embedding]).astype('float32')
                self.faiss_index.add(embedding_array)
                self.id_to_doc_mapping[self.doc_counter] = item.id
                self.doc_counter += 1
            
        except Exception as e:
            print(f"❌ Error adding knowledge item: {str(e)}")
    
    def retrieve_relevant_context(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve most relevant knowledge items for query"""
        print(f"\n🔍 RAG KNOWLEDGE RETRIEVAL")
        print(f"{'─'*50}")
        print(f"📝 Query: '{query}'")
        print(f"🎯 Retrieving top {top_k} relevant documents...")
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).astype('float32')
            
            # Search FAISS index
            if self.faiss_index and self.faiss_index.ntotal > 0:
                distances, indices = self.faiss_index.search(query_embedding, min(top_k, self.faiss_index.ntotal))
                
                relevant_docs = []
                for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
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
                                'content': doc['content'][:500] + "...",  # Truncate for display
                                'category': doc['category'],
                                'subcategory': doc['subcategory'],
                                'tags': doc['tags'],
                                'source': doc['source'],
                                'relevance_score': relevance_score,
                                'quality_score': doc['quality_score'],
                                'full_content': doc['content']  # Full content for processing
                            })
                
                # Sort by relevance score
                relevant_docs.sort(key=lambda x: x['relevance_score'], reverse=True)
                
                print(f"📖 Retrieved {len(relevant_docs)} relevant documents:")
                for i, doc in enumerate(relevant_docs, 1):
                    print(f"   {i}. {doc['title']} (relevance: {doc['relevance_score']:.3f})")
                    print(f"      Category: {doc['category']}/{doc['subcategory']}")
                    print(f"      Tags: {', '.join(doc['tags'][:3])}...")
                    print(f"      Source: {doc['source']}")
                
                return relevant_docs
            
            else:
                print(f"⚠️ FAISS index empty or unavailable, using fallback search")
                return self._fallback_search(query, top_k)
                
        except Exception as e:
            print(f"❌ RAG retrieval error: {str(e)}")
            return self._fallback_search(query, top_k)
    
    def _fallback_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Fallback search using simple text matching"""
        print(f"🔄 Using fallback text-based search...")
        
        # Simple keyword matching for demonstration
        query_lower = query.lower()
        relevant_docs = []
        
        # Search in knowledge items
        if self.mongo_client:
            try:
                # Text search in MongoDB
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
                        'content': doc['content'][:500] + "...",
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
        
        return relevant_docs[:top_k]
    
    def _calculate_text_relevance(self, query: str, doc: Dict) -> float:
        """Calculate relevance score based on text matching"""
        score = 0.0
        
        # Title match (higher weight)
        if query in doc['title'].lower():
            score += 0.5
        
        # Content match
        if query in doc['content'].lower():
            score += 0.3
        
        # Tag match
        for tag in doc['tags']:
            if query in tag.lower():
                score += 0.2
        
        return min(score, 1.0)
    
    def learn_from_interaction(self, query: str, retrieved_docs: List[str], 
                             response: str, quality_score: float):
        """Learn from user interactions to improve RAG quality"""
        print(f"\n🧠 REINFORCEMENT LEARNING UPDATE")
        print(f"{'─'*50}")
        print(f"📝 Query: {query[:50]}...")
        print(f"📊 Response Quality Score: {quality_score:.2f}/5.0")
        
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
    
    def _update_document_quality(self, doc_id: str, quality_score: float):
        """Update document quality based on interaction feedback"""
        if not self.mongo_client:
            return
        
        try:
            doc = self.knowledge_collection.find_one({"id": doc_id})
            if doc:
                # Update usage count and success rate using exponential moving average
                new_usage_count = doc['usage_count'] + 1
                alpha = 0.1  # Learning rate
                new_success_rate = (1 - alpha) * doc['success_rate'] + alpha * (quality_score / 5.0)
                
                # Update quality score (weighted average)
                new_quality_score = (doc['quality_score'] * doc['usage_count'] + quality_score) / new_usage_count
                
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
                
                print(f"📈 Updated doc {doc_id}: quality={new_quality_score:.3f}, usage={new_usage_count}")
                
        except Exception as e:
            print(f"❌ Error updating document quality: {str(e)}")
    
    def _extract_new_knowledge(self, query: str, response: str):
        """Extract new knowledge from high-quality interactions"""
        print(f"🎓 Extracting new knowledge from high-quality interaction...")
        
        try:
            # Use AI to identify if response contains new valuable knowledge
            extraction_prompt = f"""
            Analyze this query-response pair and determine if it contains new algorithmic knowledge worth storing:
            
            Query: {query}
            Response: {response[:1000]}...
            
            If this contains new algorithmic insights, patterns, or solutions not commonly found in textbooks, 
            extract the key knowledge and format it as:
            
            TITLE: [Concise title]
            CATEGORY: [algorithm category] 
            CONTENT: [Key insights in structured format]
            TAGS: [Relevant tags]
            
            If not significant enough, respond with "NO_NEW_KNOWLEDGE"
            """
            
            extraction_response = groq_client.chat_completion([
                {"role": "system", "content": "You are an expert at identifying and extracting algorithmic knowledge."},
                {"role": "user", "content": extraction_prompt}
            ])
            
            if extraction_response.success and "NO_NEW_KNOWLEDGE" not in extraction_response.content:
                # Parse and store new knowledge
                new_knowledge = self._parse_extracted_knowledge(extraction_response.content)
                if new_knowledge:
                    self._add_knowledge_item(new_knowledge)
                    print(f"✅ Added new knowledge: {new_knowledge.title}")
            
        except Exception as e:
            print(f"❌ Knowledge extraction failed: {str(e)}")
    
    def _parse_extracted_knowledge(self, content: str) -> Optional[KnowledgeItem]:
        """Parse extracted knowledge from AI response"""
        try:
            lines = content.strip().split('\n')
            title = ""
            category = "extracted"
            knowledge_content = ""
            tags = []
            
            for line in lines:
                if line.startswith("TITLE:"):
                    title = line.replace("TITLE:", "").strip()
                elif line.startswith("CATEGORY:"):
                    category = line.replace("CATEGORY:", "").strip()
                elif line.startswith("CONTENT:"):
                    knowledge_content = line.replace("CONTENT:", "").strip()
                elif line.startswith("TAGS:"):
                    tags = [tag.strip() for tag in line.replace("TAGS:", "").split(",")]
            
            if title and knowledge_content:
                return KnowledgeItem(
                    id=self._generate_id(title),
                    title=title,
                    content=knowledge_content,
                    category=category,
                    subcategory="extracted",
                    complexity="intermediate",
                    tags=tags,
                    source="AI Extracted Knowledge",
                    quality_score=0.8,  # Start with good quality
                    usage_count=0,
                    success_rate=0.8,
                    created_at=datetime.now().isoformat()
                )
        
        except Exception as e:
            print(f"❌ Failed to parse extracted knowledge: {str(e)}")
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics"""
        stats = {
            "knowledge_base_size": 0,
            "total_interactions": 0,
            "average_quality": 0.0,
            "categories": {},
            "faiss_index_size": 0
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
                print(f"❌ Error getting stats: {str(e)}")
        
        if self.faiss_index:
            stats["faiss_index_size"] = self.faiss_index.ntotal
        
        return stats

# Global enhanced RAG instance
enhanced_rag = EnhancedRAGSystem()
