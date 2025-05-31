Write-Host "🚀 Setting up Algorithm Intelligence Backend"
Write-Host "============================================"

# Create virtual environment
Write-Host "📦 Creating virtual environment..."
python -m venv venv

# Activate virtual environment (you may still need to do this manually)
Write-Host "🔁 To activate the virtual environment, run:"
Write-Host "   .\\venv\\Scripts\\Activate.ps1"

# Install dependencies
Write-Host "📥 Installing dependencies..."
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\pip.exe install -r requirements.txt

# Create database
Write-Host "🗄️ Setting up database..."
.\venv\Scripts\python.exe -c "from data.database import create_tables; create_tables()"

# Generate demo data
Write-Host "🎲 Generating demo data..."
Push-Location .\cli
.\venv\Scripts\python.exe demo_data.py
Pop-Location

Write-Host "`n✅ Setup complete!"
Write-Host ""
Write-Host "🎯 To start the server:"
Write-Host "   .\\venv\\Scripts\\Activate.ps1"
Write-Host "   python app.py"
Write-Host ""
Write-Host "🧪 To run tests:"
Write-Host "   python cli/test_all_endpoints.py"
