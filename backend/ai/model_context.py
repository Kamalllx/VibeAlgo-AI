

# backend/ai/model_context.py
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import json

@dataclass
class MCPContext:
    """Model Context Protocol context container"""
    user_id: str
    session_id: str
    context_type: str
    data: Dict[str, Any]
    timestamp: datetime
    expiry: Optional[datetime] = None

class ModelContextProtocol:
    """Simplified MCP implementation for maintaining context across interactions"""
    
    def __init__(self):
        self.contexts: Dict[str, MCPContext] = {}
        self.session_histories: Dict[str, List[Dict[str, Any]]] = {}
    
    def create_context(self, user_id: str, session_id: str, 
                      context_type: str, data: Dict[str, Any],
                      expiry_minutes: int = 60) -> str:
        """Create new context entry"""
        context_key = f"{user_id}:{session_id}:{context_type}"
        
        expiry = None
        if expiry_minutes > 0:
            from datetime import timedelta
            expiry = datetime.now() + timedelta(minutes=expiry_minutes)
        
        context = MCPContext(
            user_id=user_id,
            session_id=session_id,
            context_type=context_type,
            data=data,
            timestamp=datetime.now(),
            expiry=expiry
        )
        
        self.contexts[context_key] = context
        return context_key
    
    def get_context(self, user_id: str, session_id: str, context_type: str) -> Optional[MCPContext]:
        """Retrieve context if it exists and hasn't expired"""
        context_key = f"{user_id}:{session_id}:{context_type}"
        
        if context_key not in self.contexts:
            return None
        
        context = self.contexts[context_key]
        
        # Check expiry
        if context.expiry and datetime.now() > context.expiry:
            del self.contexts[context_key]
            return None
        
        return context
    
    def update_context(self, user_id: str, session_id: str, 
                      context_type: str, data: Dict[str, Any]) -> bool:
        """Update existing context"""
        context = self.get_context(user_id, session_id, context_type)
        
        if context:
            context.data.update(data)
            context.timestamp = datetime.now()
            return True
        return False
    
    def add_to_session_history(self, user_id: str, session_id: str, 
                              interaction: Dict[str, Any]):
        """Add interaction to session history"""
        session_key = f"{user_id}:{session_id}"
        
        if session_key not in self.session_histories:
            self.session_histories[session_key] = []
        
        interaction['timestamp'] = datetime.now().isoformat()
        self.session_histories[session_key].append(interaction)
        
        # Keep only last 50 interactions per session
        if len(self.session_histories[session_key]) > 50:
            self.session_histories[session_key] = self.session_histories[session_key][-50:]
    
    def get_session_history(self, user_id: str, session_id: str, 
                           limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent session history"""
        session_key = f"{user_id}:{session_id}"
        
        if session_key not in self.session_histories:
            return []
        
        return self.session_histories[session_key][-limit:]
    
    def build_contextual_prompt(self, user_id: str, session_id: str, 
                               current_query: str, context_types: List[str] = None) -> str:
        """Build enriched prompt with context"""
        context_types = context_types or ['code_analysis', 'dsa_progress', 'contest_history']
        
        prompt_parts = ["Based on the following context:\n"]
        
        # Add relevant contexts
        for context_type in context_types:
            context = self.get_context(user_id, session_id, context_type)
            if context:
                prompt_parts.append(f"\n{context_type.title()} Context:")
                prompt_parts.append(json.dumps(context.data, indent=2))
        
        # Add recent session history
        history = self.get_session_history(user_id, session_id, 3)
        if history:
            prompt_parts.append("\nRecent Interactions:")
            for interaction in history:
                prompt_parts.append(f"- {interaction.get('type', 'unknown')}: {interaction.get('summary', '')}")
        
        prompt_parts.append(f"\nCurrent Query: {current_query}")
        prompt_parts.append("\nProvide a contextually aware response based on the above information.")
        
        return "\n".join(prompt_parts)
    
    def cleanup_expired_contexts(self):
        """Clean up expired contexts"""
        current_time = datetime.now()
        expired_keys = []
        
        for key, context in self.contexts.items():
            if context.expiry and current_time > context.expiry:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.contexts[key]
        
        return len(expired_keys)
    
    def get_context_stats(self) -> Dict[str, Any]:
        """Get context store statistics"""
        active_contexts = len(self.contexts)
        total_sessions = len(self.session_histories)
        
        context_types = {}
        for context in self.contexts.values():
            context_type = context.context_type
            context_types[context_type] = context_types.get(context_type, 0) + 1
        
        return {
            'active_contexts': active_contexts,
            'total_sessions': total_sessions,
            'context_types': context_types,
            'memory_usage_estimate': active_contexts * 1024  # Rough estimate in bytes
        }

# Global MCP instance
mcp = ModelContextProtocol()
