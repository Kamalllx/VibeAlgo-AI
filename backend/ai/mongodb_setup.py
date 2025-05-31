# backend/ai/mongodb_setup.py
from pymongo import MongoClient
import os

def setup_mongodb_schema():
    """Setup MongoDB collections and indexes"""
    
    mongodb_uri = "mongodb+srv://kamalkarteek1:rvZSeyVHhgOd2fbE@gbh.iliw2.mongodb.net/"
    if not mongodb_uri:
        print("❌ MONGODB_URI not found in environment variables")
        return False
    
    try:
        client = MongoClient(mongodb_uri)
        db = client['algorithm_intelligence']
        
        # Create collections with validation
        knowledge_schema = {
            "$jsonSchema": {
                "bsonType": "object",
                "required": ["id", "title", "content", "category"],
                "properties": {
                    "id": {"bsonType": "string"},
                    "title": {"bsonType": "string"},
                    "content": {"bsonType": "string"},
                    "category": {"bsonType": "string"},
                    "quality_score": {"bsonType": "double", "minimum": 0, "maximum": 5},
                    "usage_count": {"bsonType": "int", "minimum": 0}
                }
            }
        }
        
        # Create knowledge base collection
        try:
            db.create_collection("knowledge_base", validator=knowledge_schema)
        except:
            pass  # Collection might already exist
        
        # Create indexes
        knowledge_collection = db['knowledge_base']
        knowledge_collection.create_index([("category", 1), ("subcategory", 1)])
        knowledge_collection.create_index([("tags", 1)])
        knowledge_collection.create_index([("quality_score", -1)])
        knowledge_collection.create_index([("id", 1)], unique=True)
        
        # Create interactions collection
        interactions_collection = db['rag_interactions']
        interactions_collection.create_index([("timestamp", -1)])
        interactions_collection.create_index([("response_quality", -1)])
        
        print("✅ MongoDB schema setup completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ MongoDB setup failed: {str(e)}")
        return False

if __name__ == "__main__":
    setup_mongodb_schema()
