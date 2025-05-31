# backend/test_enhanced_rag.py
#!/usr/bin/env python3
"""Test the enhanced RAG system"""

from ai.enhanced_rag_system import enhanced_rag
from ai.rag_pipeline import rag_pipeline

def test_enhanced_rag():
    print("🧪 Testing Enhanced RAG System")
    print("=" * 50)
    
    # Test 1: Basic stats
    stats = enhanced_rag.get_stats()
    print(f"📊 RAG System Stats:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Test 2: Knowledge retrieval
    test_queries = [
        "bubble sort time complexity",
        "binary search algorithm",
        "big o notation fundamentals",
        "hash table collision resolution"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Testing query: '{query}'")
        context = rag_pipeline.retrieve_relevant_context(query)
        print(f"📖 Retrieved {len(context)} documents")
        
        if context:
            print(f"🏆 Top result: {context[0]['title']} (score: {context[0]['relevance_score']:.3f})")
    
    print("\n✅ Enhanced RAG testing complete!")

if __name__ == "__main__":
    test_enhanced_rag()
