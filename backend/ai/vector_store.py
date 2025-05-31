
# backend/ai/vector_store.py
import json
import os
import hashlib
from typing import Dict, List, Any, Optional
import numpy as np
from datetime import datetime

class SimpleVectorStore:
    """Simplified vector store for caching embeddings and similarity search"""
    
    def __init__(self, store_path: str = "vector_store.json"):
        self.store_path = store_path
        self.vectors = {}
        self.metadata = {}
        self._load_store()
    
    def _load_store(self):
        """Load existing vector store from file"""
        if os.path.exists(self.store_path):
            try:
                with open(self.store_path, 'r') as f:
                    data = json.load(f)
                    self.vectors = data.get('vectors', {})
                    self.metadata = data.get('metadata', {})
            except Exception as e:
                print(f"Warning: Could not load vector store: {e}")
                self.vectors = {}
                self.metadata = {}
    
    def _save_store(self):
        """Save vector store to file"""
        try:
            data = {
                'vectors': self.vectors,
                'metadata': self.metadata,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.store_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save vector store: {e}")
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate simple embedding for text (hash-based for demo)"""
        # This is a simplified embedding - in production, use proper embedding models
        hash_obj = hashlib.md5(text.encode())
        hash_hex = hash_obj.hexdigest()
        
        # Convert hex to vector of floats
        embedding = []
        for i in range(0, len(hash_hex), 2):
            embedding.append(int(hash_hex[i:i+2], 16) / 255.0)
        
        # Pad or truncate to fixed size (16 dimensions)
        while len(embedding) < 16:
            embedding.append(0.0)
        embedding = embedding[:16]
        
        return embedding
    
    def add_document(self, doc_id: str, text: str, metadata: Dict[str, Any] = None):
        """Add document to vector store"""
        embedding = self._generate_embedding(text)
        
        self.vectors[doc_id] = embedding
        self.metadata[doc_id] = {
            'text': text,
            'created_at': datetime.now().isoformat(),
            **(metadata or {})
        }
        
        self._save_store()
    
    def similarity_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find similar documents using cosine similarity"""
        if not self.vectors:
            return []
        
        query_embedding = self._generate_embedding(query)
        similarities = []
        
        for doc_id, doc_embedding in self.vectors.items():
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            similarities.append({
                'doc_id': doc_id,
                'similarity': similarity,
                'metadata': self.metadata.get(doc_id, {})
            })
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = sum(a * a for a in vec1) ** 0.5
            magnitude2 = sum(b * b for b in vec2) ** 0.5
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            return dot_product / (magnitude1 * magnitude2)
        except:
            return 0.0
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID"""
        if doc_id in self.metadata:
            return {
                'doc_id': doc_id,
                'embedding': self.vectors.get(doc_id),
                'metadata': self.metadata[doc_id]
            }
        return None
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete document from store"""
        if doc_id in self.vectors:
            del self.vectors[doc_id]
            del self.metadata[doc_id]
            self._save_store()
            return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        return {
            'total_documents': len(self.vectors),
            'store_size_kb': os.path.getsize(self.store_path) / 1024 if os.path.exists(self.store_path) else 0,
            'last_updated': self.metadata.get('last_updated', 'Never')
        }

# Global vector store instance
vector_store = SimpleVectorStore()
