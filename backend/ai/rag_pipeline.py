
# backend/ai/rag_pipeline.py
import json
import os
from typing import Dict, List, Any, Optional
import numpy as np
from datetime import datetime
# backend/ai/rag_pipeline.py (UPDATED)
from ai.enhanced_rag_system import enhanced_rag

class EnhancedRAGPipeline:
    def __init__(self):
        self.rag_system = enhanced_rag
    
    def retrieve_relevant_context(self, query: str, code: str = "") -> List[Dict[str, Any]]:
        """Enhanced context retrieval with quality learning"""
        enhanced_query = f"{query} {code}" if code else query
        return self.rag_system.retrieve_relevant_context(enhanced_query, top_k=5)
    
    def learn_from_feedback(self, query: str, retrieved_docs: List[str], 
                          response: str, quality_score: float):
        """Learn from user feedback to improve future retrievals"""
        self.rag_system.learn_from_interaction(query, retrieved_docs, response, quality_score)
    
    def get_enhanced_prompt(self, query: str, context: List[Dict[str, Any]]) -> str:
        """Generate enhanced prompt with retrieved context"""
        if not context:
            return query
        
        context_text = "Relevant algorithmic knowledge:\n\n"
        for i, doc in enumerate(context[:3], 1):  # Use top 3 most relevant
            context_text += f"{i}. **{doc['title']}** (Quality: {doc['quality_score']:.2f})\n"
            context_text += f"   Source: {doc['source']}\n"
            context_text += f"   {doc['full_content'][:800]}...\n\n"
        
        enhanced_prompt = f"{context_text}\nBased on the above knowledge, {query}"
        return enhanced_prompt

# Global enhanced RAG pipeline
enhanced_rag_pipeline = EnhancedRAGPipeline()

class SimpleRAGPipeline:
    """Simplified RAG pipeline for algorithm knowledge retrieval"""
    
    def __init__(self):
        self.knowledge_base = self._load_algorithm_knowledge()
        self.embeddings_cache = {}
    
    def _load_algorithm_knowledge(self) -> Dict[str, Any]:
        """Load algorithm knowledge base"""
        knowledge = {
            "sorting_algorithms": {
                "bubble_sort": {
                    "description": "Simple comparison-based sorting algorithm",
                    "complexity": "O(n²)",
                    "use_cases": ["Educational purposes", "Small datasets"],
                    "optimizations": ["Early termination", "Cocktail shaker sort"]
                },
                "merge_sort": {
                    "description": "Divide-and-conquer sorting algorithm",
                    "complexity": "O(n log n)",
                    "use_cases": ["Large datasets", "Stable sorting required"],
                    "optimizations": ["In-place merging", "Natural merge sort"]
                },
                "quick_sort": {
                    "description": "Efficient divide-and-conquer algorithm",
                    "complexity": "O(n log n) average, O(n²) worst",
                    "use_cases": ["General purpose sorting", "Cache-friendly"],
                    "optimizations": ["3-way partitioning", "Median-of-three pivot"]
                }
            },
            "search_algorithms": {
                "linear_search": {
                    "description": "Sequential search through elements",
                    "complexity": "O(n)",
                    "use_cases": ["Unsorted data", "Simple implementation"],
                    "optimizations": ["Sentinel search", "Jump search"]
                },
                "binary_search": {
                    "description": "Efficient search on sorted arrays",
                    "complexity": "O(log n)",
                    "use_cases": ["Sorted data", "Large datasets"],
                    "optimizations": ["Interpolation search", "Exponential search"]
                }
            },
            "data_structures": {
                "arrays": {
                    "operations": {"access": "O(1)", "search": "O(n)", "insertion": "O(n)", "deletion": "O(n)"},
                    "use_cases": ["Random access", "Cache locality"],
                    "variants": ["Dynamic arrays", "Circular arrays"]
                },
                "hash_tables": {
                    "operations": {"access": "O(1) avg", "search": "O(1) avg", "insertion": "O(1) avg", "deletion": "O(1) avg"},
                    "use_cases": ["Fast lookups", "Counting", "Caching"],
                    "considerations": ["Hash function quality", "Load factor", "Collision handling"]
                }
            }
        }
        return knowledge
    
    def retrieve_relevant_context(self, query: str, code: str = "") -> List[Dict[str, Any]]:
        """Retrieve relevant context for the query"""
        query_lower = query.lower()
        relevant_docs = []
        
        # Simple keyword matching for demonstration
        for category, items in self.knowledge_base.items():
            for item_name, item_data in items.items():
                # Check if query keywords match item or description
                if (item_name.replace('_', ' ') in query_lower or 
                    any(keyword in query_lower for keyword in item_name.split('_')) or
                    item_data.get('description', '').lower() in query_lower):
                    
                    relevant_docs.append({
                        'category': category,
                        'name': item_name,
                        'data': item_data,
                        'relevance_score': self._calculate_relevance(query_lower, item_name, item_data)
                    })
        
        # Sort by relevance score
        relevant_docs.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return relevant_docs[:5]  # Return top 5 relevant documents
    
    def _calculate_relevance(self, query: str, item_name: str, item_data: Dict[str, Any]) -> float:
        """Calculate relevance score for ranking"""
        score = 0.0
        
        # Name matching
        if item_name.replace('_', ' ') in query:
            score += 2.0
        
        # Keyword matching
        keywords = item_name.split('_')
        for keyword in keywords:
            if keyword in query:
                score += 1.0
        
        # Description matching
        description = item_data.get('description', '').lower()
        if any(word in description for word in query.split()):
            score += 0.5
        
        return score
    
    def generate_enhanced_prompt(self, query: str, code: str = "") -> str:
        """Generate enhanced prompt with RAG context"""
        relevant_context = self.retrieve_relevant_context(query, code)
        
        if not relevant_context:
            return query
        
        context_text = "Relevant Algorithm Knowledge:\n"
        for doc in relevant_context:
            context_text += f"\n{doc['name'].replace('_', ' ').title()}:\n"
            context_text += f"- Description: {doc['data'].get('description', 'N/A')}\n"
            context_text += f"- Complexity: {doc['data'].get('complexity', 'N/A')}\n"
            
            if 'optimizations' in doc['data']:
                context_text += f"- Optimizations: {', '.join(doc['data']['optimizations'])}\n"
        
        enhanced_prompt = f"{context_text}\n\nBased on this knowledge, {query}"
        
        if code:
            enhanced_prompt += f"\n\nCode to analyze:\n``````"
        
        return enhanced_prompt
    
    def search_algorithms(self, complexity_requirement: str) -> List[Dict[str, Any]]:
        """Search for algorithms matching complexity requirements"""
        results = []
        
        for category, items in self.knowledge_base.items():
            for name, data in items.items():
                if complexity_requirement.lower() in data.get('complexity', '').lower():
                    results.append({
                        'name': name,
                        'category': category,
                        'complexity': data.get('complexity'),
                        'description': data.get('description')
                    })
        
        return results

# Global RAG pipeline instance
rag_pipeline = SimpleRAGPipeline()
