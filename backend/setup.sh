# backend/setup.sh
#!/bin/bash

echo "🚀 Setting up Algorithm Intelligence Backend"
echo "============================================"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Create database
echo "🗄️ Setting up database..."
python -c "from data.database import create_tables; create_tables()"

# Generate demo data
echo "🎲 Generating demo data..."
cd cli && python demo_data.py && cd ..

echo "✅ Setup complete!"
echo ""
echo "🎯 To start the server:"
echo "   source venv/bin/activate"
echo "   python app.py"
echo ""
echo "🧪 To run tests:"
echo "   python cli/test_all_endpoints.py"
