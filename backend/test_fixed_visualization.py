# backend/test_fixed_visualization.py
#!/usr/bin/env python3
import asyncio
from visualization.fixed_visualization_orchestrator import fixed_orchestrator

async def test_all_working_visualizations():
    """Test all working visualizations"""
    print("🧪 Testing Fixed Visualization System")
    print("=" * 60)
    
    scenarios = ["complexity", "performance", "animation", "general"]
    
    for scenario in scenarios:
        print(f"\n🎨 Testing {scenario} visualization...")
        try:
            files = await fixed_orchestrator.create_simple_visualization(scenario)
            print(f"✅ {scenario} successful! Files: {files}")
        except Exception as e:
            print(f"❌ {scenario} failed: {e}")
    
    print("\n" + "=" * 60)
    print("🎨 All visualization tests completed!")
    print("📁 Check the generated_visualizations folder!")

if __name__ == "__main__":
    asyncio.run(test_all_working_visualizations())
