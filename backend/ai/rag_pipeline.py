# backend/ai/rag_pipeline.py (FIXED VERSION)
from typing import Dict, List, Any
from ai.enhanced_rag_system import enhanced_rag

class EnhancedRAGPipeline:
    """Enhanced RAG pipeline with MongoDB, FAISS, and reinforcement learning"""
    
    def __init__(self):
        self.rag_system = enhanced_rag
        print(f"🔗 Enhanced RAG Pipeline initialized")
    
    def retrieve_relevant_context(self, query: str, code: str = "") -> List[Dict[str, Any]]:
        """Enhanced context retrieval with quality learning - FIXED METHOD SIGNATURE"""
        # Combine query and code for better context
        enhanced_query = f"{query} {code}" if code else query
        
        # Call the enhanced RAG system with correct parameters
        return self.rag_system.retrieve_relevant_context(enhanced_query, top_k=5)
    
    def generate_enhanced_prompt(self, query: str, context: List[Dict[str, Any]]) -> str:
        """Generate enhanced prompt with retrieved context"""
        if not context:
            return query
        
        context_text = "📚 Relevant Algorithmic Knowledge:\n\n"
        
        for i, doc in enumerate(context[:3], 1):  # Use top 3 most relevant
            context_text += f"{i}. **{doc['title']}** (Quality: {doc['quality_score']:.2f}, Relevance: {doc['relevance_score']:.2f})\n"
            context_text += f"   Source: {doc['source']}\n"
            context_text += f"   Category: {doc['category']}/{doc['subcategory']}\n"
            context_text += f"   {doc['full_content'][:600]}...\n\n"
        
        enhanced_prompt = f"{context_text}\nBased on the above algorithmic knowledge, {query}"
        return enhanced_prompt
    
    def learn_from_feedback(self, query: str, retrieved_docs: List[str], 
                          response: str, quality_score: float):
        """Learn from user feedback to improve future retrievals"""
        self.rag_system.learn_from_interaction(query, retrieved_docs, response, quality_score)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics"""
        return self.rag_system.get_stats()

# Global enhanced RAG pipeline
rag_pipeline = EnhancedRAGPipeline()
