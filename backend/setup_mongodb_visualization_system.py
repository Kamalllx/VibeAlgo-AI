# backend/setup_mongodb_visualization_system.py
#!/usr/bin/env python3
"""
Complete setup script for MongoDB-powered visualization system
"""

import os
import sys
from pathlib import Path

def setup_mongodb_system():
    """Setup the complete MongoDB visualization system"""
    
    print("🔧 Setting up MongoDB Visualization System")
    print("=" * 50)
    
    # Get MongoDB connection string
    connection_string = input("Enter MongoDB connection string: ").strip()
    
    if not connection_string:
        connection_string = "mongodb://localhost:27017/"
        print(f"Using default: {connection_string}")
    
    # Initialize MongoDB manager
    from visualization_database.mongodb_manager import initialize_mongodb_manager
    
    try:
        mongo_manager = initialize_mongodb_manager(connection_string)
        print("✅ MongoDB connection established")
        
        # Populate database
        print("\n📊 Populating algorithm database...")
        mongo_manager.populate_algorithms_database()
        
        # Create visualization files
        print("\n🎨 Creating visualization files...")
        from visualization_database.create_all_visualizations import create_all_visualization_files
        create_all_visualization_files()
        
        # Test the system
        print("\n🧪 Testing the system...")
        test_queries = [
            "How does Dijkstra's algorithm work?",
            "Show me binary search",
            "Explain bubble sort",
            "Longest increasing subsequence"
        ]
        
        for query in test_queries:
            print(f"\nTesting: '{query}'")
            matches = mongo_manager.detect_algorithm(query)
            for category, algo_key, name, confidence in matches[:2]:
                print(f"  → {name} (confidence: {confidence:.2f})")
        
        print("\n✅ MongoDB visualization system setup complete!")
        print(f"🗄️ Database: {mongo_manager.algorithms.count_documents({})} algorithms loaded")
        
        return mongo_manager
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return None

if __name__ == "__main__":
    setup_mongodb_system()
